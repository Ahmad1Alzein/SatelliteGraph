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
