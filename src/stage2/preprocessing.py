from __future__ import annotations

from typing import Any

import numpy as np


def prepare_image(image: np.ndarray, metadata: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if image.ndim == 2:
        image = image[..., None]
    bands = [int(value) - 1 for value in config["rgb_bands"]]
    if max(bands) >= image.shape[2]:
        raise ValueError(f"RGB mapping {config['rgb_bands']} exceeds available {image.shape[2]} bands")
    selected = image[..., bands].astype(np.float64, copy=True)
    finite = np.isfinite(selected).all(axis=2)
    valid = finite.copy()
    nodata = metadata.get("nodata")
    alpha_index = metadata.get("alpha_band_index")
    if alpha_index is not None:
        valid &= image[...,int(alpha_index)]>0
    elif nodata is not None:
        valid &= ~np.any(selected == nodata, axis=2)
    if not valid.any():
        raise ValueError("Tile contains no valid pixels after nodata/finite checks")
    mode = config["normalization"]
    operations: list[dict[str, Any]] = []
    rgb = np.empty_like(selected)
    global_values = selected[valid]
    for channel in range(3):
        values = selected[..., channel][valid]
        if mode == "none":
            low, high = 0.0, 255.0
            if selected.dtype != np.uint8 and (values.min() < 0 or values.max() > 255):
                raise ValueError("normalization=none requires values in [0,255]")
        elif mode == "fixed_range":
            fixed = config.get("fixed_range")
            if not fixed or len(fixed) != 2:
                raise ValueError("fixed_range normalization requires fixed_range: [low, high]")
            low, high = map(float, fixed)
        elif mode == "min_max":
            low, high = float(global_values.min()), float(global_values.max())
        elif mode == "percentile":
            low, high = np.percentile(global_values, [config["lower_percentile"], config["upper_percentile"]])
        elif mode == "per_band_percentile":
            low, high = np.percentile(values, [config["lower_percentile"], config["upper_percentile"]])
        else:
            raise ValueError(f"Unsupported normalization: {mode}")
        if not high > low:
            raise ValueError(f"Contrast range collapsed for channel {channel + 1}: {low}..{high}")
        rgb[..., channel] = np.clip((selected[..., channel] - low) * 255.0 / (high - low), 0, 255)
        operations.append({"channel": channel + 1, "source_band": bands[channel] + 1, "low": float(low), "high": float(high)})
    fill = int(config["nodata_fill"])
    rgb[~valid] = fill
    result = np.rint(rgb).astype(np.uint8)
    report = {"normalization": mode, "rgb_bands": config["rgb_bands"], "lower_percentile": config["lower_percentile"], "upper_percentile": config["upper_percentile"], "nodata_fill": fill, "valid_pixel_count": int(valid.sum()), "invalid_pixel_count": int(valid.size - valid.sum()), "operations": operations, "input_dtype": str(image.dtype), "output_dtype": str(result.dtype)}
    return result, valid.astype(bool), report
