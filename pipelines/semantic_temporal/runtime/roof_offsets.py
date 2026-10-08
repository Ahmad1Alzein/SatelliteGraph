"""Current-imagery roof offsets, never historical shift samples."""
import numpy as np

def mask_iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)

def roof_offset_field(b13, b23, seg=1024, search=12, smooth_sigma=1.0):
    """Regional roof displacement (2x px) of 2013 buildings relative to 2023 along the strip:
    best mask-overlap shift per `seg`-px segment, then smoothed and linearly interpolated per column.
    Co-registration aligns the ground; roofs additionally lean by view angle x height."""
    centers, dys, dxs = [], [], []
    if min(b23.shape)<=32: return np.zeros(b23.shape[1]),np.zeros(b23.shape[1]),[]
    W = b23.shape[1]
    for x0 in range(0, W - seg // 2, seg // 2):
        s13, s23 = b13[:, x0:x0 + seg], b23[:, x0:x0 + seg]
        if s23.sum() < 500:
            continue
        best = max(((mask_iou(np.roll(np.roll(s13, dy, 0), dx, 1)[16:-16, 16:-16], s23[16:-16, 16:-16]), dy, dx)
                    for dy in range(-search, search + 1) for dx in range(-search, search + 1)))
        centers.append(x0 + min(seg, W - x0) / 2); dys.append(best[1]); dxs.append(best[2])
    if not centers: return np.zeros(W),np.zeros(W),[]
    from scipy.ndimage import gaussian_filter1d
    dys = gaussian_filter1d(np.array(dys, float), smooth_sigma); dxs = gaussian_filter1d(np.array(dxs, float), smooth_sigma)
    cols = np.arange(W)
    return np.interp(cols, centers, dys), np.interp(cols, centers, dxs), list(zip(centers, dys.round(2), dxs.round(2)))

def offset_building_labels(lab13, bids13, b23, fdy, fdx, local=4):
    """Move every 2013 building by the regional roof offset at its centroid, refined per building
    within +-`local` px to maximise overlap with the 2023 building mask."""
    from scipy import ndimage
    out = np.zeros_like(lab13)
    H, W = lab13.shape
    objs = ndimage.find_objects(lab13)
    shifts = {}
    for bid in sorted(bids13, key=lambda b: -(lab13 == b).sum() if False else 0):
        sl = objs[bid - 1] if bid - 1 < len(objs) else None
        if sl is None:
            continue
        ys, xs = np.nonzero(lab13[sl] == bid)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        cx = int(round(xs.mean()))
        base_dy, base_dx = int(round(fdy[cx])), int(round(fdx[cx]))
        # maximise overlap; on ties prefer the smallest departure from the regional offset
        # (otherwise a building next to another can slide onto its neighbour with an equal score)
        best, bdy, bdx = (-1, 0.0), base_dy, base_dx
        for dy in range(base_dy - local, base_dy + local + 1):
            for dx in range(base_dx - local, base_dx + local + 1):
                yy, xx = ys + dy, xs + dx
                ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                score = (int(b23[yy[ok], xx[ok]].sum()), -float(np.hypot(dy - base_dy, dx - base_dx)))
                if score > best:
                    best, bdy, bdx = score, dy, dx
        yy, xx = ys + bdy, xs + bdx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        out[yy[ok], xx[ok]] = bid
        shifts[bid] = (bdy, bdx)
    return out, shifts
