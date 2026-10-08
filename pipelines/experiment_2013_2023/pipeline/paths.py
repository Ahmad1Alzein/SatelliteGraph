"""Where the pipeline finds its inputs (everything it writes is relative to the folder it runs in).

run_experiment.py sets REPRO_DATA / REPRO_CHECKPOINTS; the defaults are <package>/data and <package>/checkpoints.
"""
import os
from pathlib import Path

PACKAGE = Path(os.environ.get("REPRO_PACKAGE", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("REPRO_DATA", PACKAGE / "data"))
CHECKPOINTS = Path(os.environ.get("REPRO_CHECKPOINTS", PACKAGE / "checkpoints"))
REFERENCE = PACKAGE / "reference"

DATASET = DATA / "dataset"                     # dataset/2013, dataset/2023: 7 PNG tiles + metadata.csv each
ALIGNED_FULL = DATA / "aligned"                # full 1.5 m rasters (margin for co-registration context)
META_2023 = DATASET / "2023" / "metadata.csv"  # tile grid (x0, y0, affine) shared by both years
SEGFORMER_DIR = CHECKPOINTS / "satlens-segformer"
DELANY_CKPT = CHECKPOINTS / "DelineateAnythingv2.pt"
