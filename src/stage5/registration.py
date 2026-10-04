"""Optional robust registration, independent of temporal region matching."""
from __future__ import annotations

import math
from numbers import Integral, Real

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform
from shapely.geometry import MultiPoint

from .coordinates import metric_crs


def staged_tile_control_points(moving_path, moving_affine, moving_crs,
                               reference_path, reference_affine, reference_crs,
                               target_crs):
    """Register two saved satellite PNGs using their approximate georeference.

    Both images are first resampled onto the reference tile's geographic grid.
    Phase correlation supplies a coarse displacement; SIFT descriptors are
    searched only near that displacement, then a partial-affine RANSAC keeps
    geographically distributed landmarks. Returned controls are in the same
    projected CRS and can be independently fitted by ``fit_registration``.
    """
    from pathlib import Path
    from PIL import Image
    from scipy.spatial import cKDTree
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject

    target = CRS.from_user_input(metric_crs(target_crs))
    reference_crs, moving_crs = CRS.from_user_input(reference_crs), CRS.from_user_input(moving_crs)
    reference_affine, moving_affine = Affine(*reference_affine), Affine(*moving_affine)

    def read(path):
        with Image.open(Path(path)) as image:
            pixels = np.asarray(image.convert("RGBA"))
        return pixels[:, :, :3], pixels[:, :, 3]

    ref_rgb, ref_alpha = read(reference_path)
    mov_rgb, mov_alpha = read(moving_path)
    height, width = ref_rgb.shape[:2]
    if min(height, width) < 32:
        raise ValueError("Image registration needs tiles at least 32 pixels wide and high")
    if max(height, width) > 4096 or max(mov_rgb.shape[:2]) > 4096:
        raise ValueError("Image registration tile exceeds 4096 pixels; use a smaller tile")

    if reference_crs == target:
        grid = reference_affine
    else:
        corners = [reference_affine * point for point in
                   ((0, 0), (width, 0), (width, height), (0, height))]
        xs, ys = zip(*corners)
        grid, width, height = calculate_default_transform(
            reference_crs, target, width, height,
            min(xs), min(ys), max(xs), max(ys))

    def warped(rgb, alpha, source_affine, source_crs):
        result = np.zeros((height, width, 3), np.uint8)
        valid = np.zeros((height, width), np.uint8)
        for band in range(3):
            reproject(rgb[:, :, band], result[:, :, band],
                      src_transform=source_affine, src_crs=source_crs,
                      dst_transform=grid, dst_crs=target,
                      resampling=Resampling.bilinear)
        reproject(alpha, valid, src_transform=source_affine, src_crs=source_crs,
                  dst_transform=grid, dst_crs=target, resampling=Resampling.nearest)
        return result, valid > 0

    ref_rgb, ref_valid = warped(ref_rgb, ref_alpha, reference_affine, reference_crs)
    mov_rgb, mov_valid = warped(mov_rgb, mov_alpha, moving_affine, moving_crs)
    overlap = ref_valid & mov_valid
    if overlap.mean() < 0.35:
        raise ValueError("Saved yearly PNGs have too little approximate geographic overlap")

    def gray(rgb):
        value = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(value)

    reference_gray, moving_gray = gray(ref_rgb), gray(mov_rgb)
    if np.array_equal(ref_rgb, mov_rgb) and ref_valid.all() and mov_valid.all():
        # Exact imagery on the same geographic grid provides an exact identity
        # control set, including featureless synthetic/monochrome tiles.
        samples = [(x, y) for y in np.linspace(0.1*height, 0.9*height, 3)
                   for x in np.linspace(0.1*width, 0.9*width, 3)]
        world = np.asarray([grid * (x, y) for x, y in samples], float)
        return world.copy(), world.copy(), {
            "method": "identical_staged_png", "phase_shift_pixels": [0.0, 0.0],
            "candidate_count": 9, "inlier_count": 9, "inlier_quadrants": 4}

    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    moving_phase = moving_gray.astype(np.float32) * overlap
    reference_phase = reference_gray.astype(np.float32) * overlap
    shift, response = cv2.phaseCorrelate(moving_phase, reference_phase, window)
    if (not np.isfinite(shift).all() or response < 0.015
            or max(abs(shift[0])/width, abs(shift[1])/height) > 0.30):
        raise ValueError("Image registration could not establish a reliable coarse shift")

    detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008)
    reference_mask = cv2.erode(ref_valid.astype(np.uint8), np.ones((7, 7), np.uint8)) * 255
    moving_mask = cv2.erode(mov_valid.astype(np.uint8), np.ones((7, 7), np.uint8)) * 255
    ref_keypoints, ref_descriptors = detector.detectAndCompute(reference_gray, reference_mask)
    mov_keypoints, mov_descriptors = detector.detectAndCompute(moving_gray, moving_mask)
    if ref_descriptors is None or mov_descriptors is None:
        raise ValueError("Image registration found too few visual landmarks")
    ref_xy = np.asarray([point.pt for point in ref_keypoints], np.float32)
    mov_xy = np.asarray([point.pt for point in mov_keypoints], np.float32)
    nearby = cKDTree(ref_xy)
    candidates = []
    radius = max(8.0, min(14.0, 0.025 * max(width, height)))
    for i, point in enumerate(mov_xy):
        indices = nearby.query_ball_point(point + shift, radius)
        if len(indices) < 2:
            continue
        differences = ref_descriptors[indices] - mov_descriptors[i]
        squared = np.einsum("ij,ij->i", differences, differences)
        nearest = np.argpartition(squared, 1)[:2]
        first, second = nearest[np.argsort(squared[nearest])]
        if squared[first] < 0.85**2 * squared[second]:
            candidates.append((float(squared[first]), i, indices[first]))
    candidates.sort()
    used_reference = set()
    pairs = []
    for _, i, j in candidates:
        if j not in used_reference:
            pairs.append((i, j))
            used_reference.add(j)
    if len(pairs) < 20:
        raise ValueError("Image registration found too few spatially guided feature pairs")
    moving_points = mov_xy[[i for i, _ in pairs]]
    reference_points = ref_xy[[j for _, j in pairs]]
    cv2.setRNGSeed(42)
    matrix, mask = cv2.estimateAffinePartial2D(
        moving_points, reference_points, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999,
        refineIters=20)
    if matrix is None or mask is None:
        raise ValueError("Image registration could not fit a consistent landmark transform")
    inliers = mask.ravel().astype(bool)
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    rotation = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    quadrants = {(int(x >= width/2), int(y >= height/2))
                 for x, y in moving_points[inliers]}
    hull_fraction = MultiPoint(moving_points[inliers]).convex_hull.area / (width*height)
    residual = np.linalg.norm(moving_points[inliers] @ matrix[:, :2].T
                              + matrix[:, 2] - reference_points[inliers], axis=1)
    if (inliers.sum() < 12 or inliers.mean() < 0.08 or len(quadrants) < 3
            or hull_fraction < 0.12 or not 0.95 <= scale <= 1.05
            or rotation > 3.0 or np.percentile(residual, 95) > 3.0):
        raise ValueError("Image registration landmarks are too weak or too local")
    moving_world = np.asarray([grid * (x+0.5, y+0.5)
                               for x, y in moving_points[inliers]], float)
    reference_world = np.asarray([grid * (x+0.5, y+0.5)
                                  for x, y in reference_points[inliers]], float)
    return moving_world, reference_world, {
        "method": "phase_guided_sift_ransac", "phase_shift_pixels": list(map(float, shift)),
        "phase_response": float(response), "candidate_count": len(pairs),
        "inlier_count": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
        "inlier_quadrants": len(quadrants), "inlier_hull_fraction": float(hull_fraction),
        "landmark_p95_residual_pixels": float(np.percentile(residual, 95)),
        "image_affine_pixels": matrix.ravel().tolist(),
        "accuracy_status": "in_sample_landmarks_not_independent_ground_truth"}


def fit_registration(moving_xy, reference_xy, *, max_residual_m=2.0,
                     min_inliers=6, min_inlier_ratio=0.5):
    """Fit moving -> reference world coordinates using RANSAC; reject poor fits.

    Both inputs are already in the SAME projected metre CRS. Residuals are
    in-sample diagnostics, not independent registration accuracy estimates.
    """
    if isinstance(min_inliers, (bool, np.bool_)) or not isinstance(min_inliers, Integral) or min_inliers < 6:
        raise ValueError("Registration policy min_inliers must be an integer >= 6")
    for name, value in (("max_residual_m", max_residual_m), ("min_inlier_ratio", min_inlier_ratio)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Registration policy {name} must be a finite positive number")
    if min_inlier_ratio > 1:
        raise ValueError("Registration policy min_inlier_ratio must be <= 1")
    moving, reference = np.asarray(moving_xy, float), np.asarray(reference_xy, float)
    if (moving.ndim != 2 or moving.shape[1:] != (2,) or moving.shape != reference.shape
            or len(moving) < min_inliers or not np.isfinite(moving).all()
            or not np.isfinite(reference).all() or max_residual_m <= 0):
        raise ValueError("Registration requires at least six finite paired 2D control points")
    if any(len(np.unique(points, axis=0)) < min_inliers for points in (moving, reference)):
        raise ValueError("Registration requires enough distinct control points in both frames")
    if any(np.linalg.matrix_rank(points - points.mean(axis=0)) < 2 for points in (moving, reference)):
        raise ValueError("Registration control points must not be collinear")
    # Center first: OpenCV's internal float32 must not round large world eastings.
    source_origin, target_origin = moving.mean(axis=0), reference.mean(axis=0)
    cv2.setRNGSeed(42)
    matrix, inliers = cv2.estimateAffine2D(
        moving - source_origin, reference - target_origin, method=cv2.RANSAC,
        ransacReprojThreshold=float(max_residual_m), maxIters=5000,
        confidence=0.999, refineIters=20)
    if matrix is None or inliers is None:
        raise ValueError("Registration failed to find a robust affine transform")
    accepted = inliers.ravel().astype(bool)
    if accepted.sum() < min_inliers or accepted.mean() < min_inlier_ratio:
        raise ValueError("Registration has insufficient inlier support")
    for points in (moving[accepted], reference[accepted]):
        if len(np.unique(points, axis=0)) < min_inliers:
            raise ValueError("Registration has insufficient distinct inlier control points")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Registration inlier control points must not be collinear")
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    if np.linalg.det(matrix[:, :2]) <= 0 or singular.min() < 0.8 or singular.max() > 1.25:
        raise ValueError("Registration implies implausible scale/shear/reflection in world coordinates")
    matrix[:, 2] += target_origin - matrix[:, :2] @ source_origin
    residuals = np.linalg.norm(moving @ matrix[:, :2].T + matrix[:, 2] - reference, axis=1)
    rmse = float(np.sqrt(np.mean(residuals[accepted] ** 2)))
    if rmse > max_residual_m or not np.isfinite(matrix).all():
        raise ValueError("Registration residual exceeds the configured tolerance")
    return {"method": "affine_ransac", "affine": matrix.ravel().tolist(),
            "point_count": len(moving), "inlier_count": int(accepted.sum()),
            "inlier_ratio": float(accepted.mean()), "inlier_rmse_m": rmse,
            "inlier_p95_m": float(np.percentile(residuals[accepted], 95)),
            "inlier_mask": accepted.tolist(),
            "inlier_hull_area_m2": float(MultiPoint(moving[accepted]).convex_hull.area),
            "accuracy_status": "in_sample_only_requires_independent_validation"}


def image_control_points(moving_path, reference_path, target_crs, *, max_dimension=2048):
    """Bounded-resolution SIFT mutual ratio matches from georeferenced rasters.

    Uses actual raster pixels, never a rendered graph/annotated overlay. Callers
    may instead supply matches from LightGlue/LoFTR or surveyed control points.
    """
    target = metric_crs(target_crs)

    def features(path):
        with rasterio.open(path) as source:
            if source.crs is None:
                raise ValueError("Image registration requires georeferenced rasters")
            factor = min(1.0, max_dimension / max(source.width, source.height))
            width, height = max(1, round(source.width * factor)), max(1, round(source.height * factor))
            bands = list(range(1, min(source.count, 3) + 1))
            data = source.read(bands, out_shape=(len(bands), height, width),
                               masked=True, resampling=Resampling.bilinear)
            valid = ~np.ma.getmaskarray(data).any(axis=0)
            gray = data.mean(axis=0).filled(0).astype(np.float32)
            valid &= np.isfinite(gray)
            if not valid.any():
                raise ValueError("Registration image has no valid pixels")
            low, high = np.percentile(gray[valid], [2, 98])
            if high <= low:
                raise ValueError("Registration image has insufficient texture")
            normalized = np.clip((gray - low) * 255 / (high - low), 0, 255)
            normalized[~valid] = 0
            keypoints, descriptors = cv2.SIFT_create(nfeatures=6000).detectAndCompute(
                normalized.astype(np.uint8), valid.astype(np.uint8) * 255)
            if descriptors is None or len(keypoints) < 6:
                raise ValueError("Registration image has insufficient keypoints")
            # OpenCV integer coordinates locate pixel centres, while affine uses edges.
            points = [source.transform * ((p.pt[0] + 0.5) * source.width / width,
                                          (p.pt[1] + 0.5) * source.height / height) for p in keypoints]
            xs, ys = transform(source.crs, target, *zip(*points))
            return np.column_stack([xs, ys]), descriptors

    moving, md = features(moving_path)
    reference, rd = features(reference_path)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratios(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}

    forward, reverse = ratios(md, rd), ratios(rd, md)
    pairs = sorted((i, j) for i, j in forward.items() if reverse.get(j) == i)
    if len(pairs) < 6:
        raise ValueError("Insufficient mutual SIFT matches; provide external registration control points")
    return moving[[i for i, _ in pairs]], reference[[j for _, j in pairs]]
