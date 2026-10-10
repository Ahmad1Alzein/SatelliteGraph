"""Masked paired harmonization on owned mosaics, before either date is segmented."""
from dataclasses import replace
from pathlib import Path
import math
import cv2
import numpy as np
from affine import Affine
from scipy.ndimage import binary_erosion

from .contracts import GridSpec
from .storage import atomic_json, read_json, save_manifest, verify_manifest, write_raster


def _blur(image, valid, sigma):
    if sigma <= 0:
        return image.copy()
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    numerator = cv2.GaussianBlur(image.astype(np.float32) * valid[..., None], (0, 0), sigma)
    result = np.divide(numerator, weight[..., None], out=np.zeros_like(numerator),
                       where=weight[..., None] > 1e-8)
    result[~valid] = 0
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _sharpness(image, valid):
    # Gradients use only pixels whose entire 3x3 neighborhood is observed.
    inside = binary_erosion(valid, structure=np.ones((3, 3)))
    if not inside.any():
        return 0.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy)[inside].mean())


def harmonize_images(reference, moving, reference_valid, moving_valid):
    """Original histogram/blur method with shared observed, non-shadow statistics."""
    ref, mov = reference.copy(), moving.copy()
    common = reference_valid & moving_valid
    common &= (cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) >= 20)
    common &= (cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY) >= 20)
    info = {'common_pixels': int(common.sum()), 'tolerance': .02, 'max_sigma': 3.,
            'bisection_steps': 14, 'blur_sigma_reference': 0., 'blur_sigma_moving': 0.}
    if not common.any():
        info['status'] = 'no_common_nonshadow_pixels'
        return ref, mov, info
    lookup = []
    for channel in range(3):
        values, counts = np.unique(mov[..., channel][common], return_counts=True)
        targets, frequencies = np.unique(ref[..., channel][common], return_counts=True)
        mapped = np.interp(np.cumsum(counts)/counts.sum(),
                           np.cumsum(frequencies)/frequencies.sum(), targets)
        lut = np.clip(np.rint(np.interp(np.arange(256), values, mapped)), 0, 255).astype(np.uint8)
        mov[..., channel] = lut[mov[..., channel]]
        lookup.append(lut.tolist())
    # Shadow pixels cannot affect blur or statistics at their boundaries.
    sr, sm = _sharpness(ref, common), _sharpness(mov, common)
    info.update(sharpness_reference=sr, sharpness_moving=sm, histogram_luts=lookup, status='complete')
    if sr > sm * 1.02 or sm > sr * 1.02:
        name = 'reference' if sr > sm else 'moving'
        image = ref if sr > sm else mov
        target = min(sr, sm)
        low, high = 0., 3.
        for _ in range(14):
            sigma = (low + high) / 2
            if _sharpness(_blur(image, common, sigma), common) > target:
                low = sigma
            else:
                high = sigma
        changed = _blur(image, common, high)
        # Outside the fitted common domain retain original per-year image pixels.
        image[common] = changed[common]
        info['blur_sigma_' + name] = high
        info['sharpness_after_' + name] = _sharpness(image, common)
    ref[~reference_valid] = 0; mov[~moving_valid] = 0
    return ref, mov, info


def prepare_model_inputs(context):
    from .registration import prepare_tile
    if not context.options.get('harmonize_images'):
        return context
    if len(context.years) != 2:
        raise ValueError('--harmonize-images requires two chronological input years')
    directory = context.output / 'prepared/harmonization'
    record = directory / 'inputs.json'
    if record.is_file():
        verify_manifest(directory)
        value = read_json(record)
        if value['input_fingerprint'] != context.fingerprint:
            raise ValueError('Prepared input settings changed; choose a fresh output folder')
        return replace(context, options={**context.options, 'prepared_inputs': value['tiles']})
    # All inputs already have a shared axis-aligned resolution and anchor.
    plain = replace(context, options={k:v for k,v in context.options.items() if k != 'prepared_inputs'})
    prepared = [(tile, *prepare_tile(tile, plain)) for year in context.years for tile in year.tiles]
    resolution = context.options['shared_resolution']
    anchor_x, anchor_y = context.options['shared_anchor']
    sign = -1 if prepared[0][3].crs else 1
    positions = []
    for tile, rgb, valid, grid in prepared:
        affine = Affine(*grid.affine)
        x = round((affine.c-anchor_x)/resolution)
        y = round((affine.f-anchor_y)/(sign*resolution))
        positions.append((x, y))
    x0 = min(x for x,y in positions); y0 = min(y for x,y in positions)
    width = max(x+rgb.shape[1] for (tile,rgb,valid,grid),(x,y) in zip(prepared,positions))-x0
    height = max(y+rgb.shape[0] for (tile,rgb,valid,grid),(x,y) in zip(prepared,positions))-y0
    budget = int(context.options.get('max_comparison_pixels') or 4_000_000)
    if width*height > budget:
        raise ValueError(f'Paired preprocessing requires {width*height} pixels; raise --max-comparison-pixels')
    mosaics = {}; masks = {}
    for year in context.years:
        mosaic = np.zeros((height,width,3), np.uint8); observed = np.zeros((height,width), bool)
        for (tile,rgb,valid,grid),(x,y) in zip(prepared,positions):
            if tile.year != year.year:
                continue
            x -= x0; y -= y0; h,w = valid.shape
            use = valid & ~observed[y:y+h,x:x+w]
            mosaic[y:y+h,x:x+w][use] = rgb[use]
            observed[y:y+h,x:x+w] |= valid
        mosaics[year.year] = mosaic; masks[year.year] = observed
    earlier,later = [year.year for year in context.years]
    ref,mov,info = harmonize_images(mosaics[later],mosaics[earlier],masks[later],masks[earlier])
    model = {earlier:mov, later:ref}
    paths = {}; mapping = {}
    for (tile,rgb,valid,grid),(x,y) in zip(prepared,positions):
        x -= x0; y -= y0; h,w = valid.shape
        base = directory / str(tile.year) / tile.image.stem
        base.mkdir(parents=True,exist_ok=True)
        entry = {'grid':grid, 'original_rgb':base/'original_rgb.tif',
                 'model_rgb':base/'model_rgb.tif', 'valid':base/'valid.tif'}
        for name in ('original_rgb','model_rgb'):
            data = mosaics[tile.year] if name=='original_rgb' else model[tile.year]
            piece = data[y:y+h,x:x+w].copy(); piece[~valid] = 0
            write_raster(entry[name],piece.transpose(2,0,1),grid)
            paths[tile.key+'/'+name] = entry[name]
        write_raster(entry['valid'],valid.astype(np.uint8),grid,nodata=0)
        paths[tile.key+'/valid'] = entry['valid']; mapping[tile.key] = entry
    atomic_json(record, {'input_fingerprint':context.fingerprint,'tiles':mapping,'statistics':info})
    paths['inputs'] = record; save_manifest(directory,paths,method='paired-harmonization.v1')
    result = replace(context,options={**context.options,'prepared_inputs':read_json(record)['tiles']})
    atomic_json(context.output/'context.json',result)
    from .context import seal_prepared_context
    seal_prepared_context(result)
    return result
