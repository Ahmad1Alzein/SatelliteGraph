from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

TILES_DIR = DATA_DIR / "tiles"

METADATA_DIR = DATA_DIR / "metadata"

# ==========================
# Tiling Parameters
# ==========================

TILE_SIZE = 512

OVERLAP = 50

STRIDE = TILE_SIZE - OVERLAP