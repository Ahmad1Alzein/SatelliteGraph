"""Font files for figure labels: Arial on Windows (as in the original experiment), else matplotlib's DejaVu Sans.
Fonts only change the rendered PNG/JPG figures, never the data outputs."""
from pathlib import Path

import matplotlib

_WIN, _MPL = Path("C:/Windows/Fonts"), Path(matplotlib.get_data_path()) / "fonts" / "ttf"
REGULAR = str(_WIN / "arial.ttf") if (_WIN / "arial.ttf").exists() else str(_MPL / "DejaVuSans.ttf")
BOLD = str(_WIN / "arialbd.ttf") if (_WIN / "arialbd.ttf").exists() else str(_MPL / "DejaVuSans-Bold.ttf")
