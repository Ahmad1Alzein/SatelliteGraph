from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_json, atomic_npz


def save_masks(npz_path: Path, json_path: Path, masks: list[np.ndarray], records: list[dict[str, Any]]) -> None:
    if len(masks) != len(records):
        raise ValueError("Mask and metadata counts differ")
    if masks:
        shape = masks[0].shape
        if any(mask.shape != shape for mask in masks):
            raise ValueError("All masks must have the same shape")
        dense = np.stack([np.asarray(mask, bool) for mask in masks])
        packed = np.packbits(dense.reshape(len(masks), -1), axis=1)
    else:
        shape, packed = (0, 0), np.empty((0, 0), np.uint8)
    atomic_npz(npz_path, packed_masks=packed, shape=np.asarray(shape, np.int64), count=np.asarray(len(masks), np.int64))
    clean = []
    for index, record in enumerate(records):
        value = {key: val for key, val in record.items() if key not in {"mask", "segmentation"}}
        value["mask_index"] = index
        clean.append(value)
    atomic_json(json_path, {"mask_storage": npz_path.name, "count": len(clean), "records": clean})


def load_masks(npz_path: Path, json_path: Path) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    with np.load(npz_path, allow_pickle=False) as data:
        count = int(data["count"])
        shape = tuple(int(v) for v in data["shape"])
        packed = data["packed_masks"]
        masks = [np.unpackbits(packed[i])[: shape[0] * shape[1]].reshape(shape).astype(bool) for i in range(count)]
    import json
    records = json.loads(json_path.read_text(encoding="utf-8"))["records"]
    if len(records) != count:
        raise ValueError(f"Corrupt mask artifact: {npz_path} and {json_path} counts differ")
    return masks, records
