"""Thesis dashboard: every page renders without exceptions on the real artifacts (skipped when they are absent),
the original image-comparison tool stays mounted unchanged, and evidence labelling is applied."""
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard import data as D  # noqa: E402
from dashboard import ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "imagery", "segmentation", "graphs", "temporal", "gnn", "changes", "evaluation", "reproduce"]
HAVE_BUNDLE = (ROOT / "data" / "legacy_building_stgnn" / "strip" / "2013" / "segment_ids.tif").is_file()


def test_every_evidence_type_has_label_colour_and_explanation():
    assert {"pseudo", "visual", "osm", "external", "none", "descriptive"} <= set(ui.EVIDENCE)
    for label, fg, bg, tip in ui.EVIDENCE.values():
        assert label and fg.startswith("#") and bg.startswith("#") and len(tip) > 20
    assert "Pseudo-label agreement" in ui.badge("pseudo")


def test_class_palette_matches_original_study():
    assert D.COLORS[1] == (230, 57, 70) and D.CLASSES[7] == "paved_developed" and D.hexcolor("building") == "#e63946"


def test_dashboard_mounts_original_tool_unchanged():
    text = (ROOT / "thesis_dashboard.py").read_text(encoding="utf-8")
    assert 'st.Page("streamlit_app.py"' in text
    for page in PAGES:
        assert (ROOT / "dashboard" / "pages" / f"{page}.py").is_file()


@pytest.mark.skipif(not HAVE_BUNDLE, reason="frozen study bundle not available")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(ROOT / "thesis_dashboard.py"), default_timeout=300)
    at.run()
    at.switch_page(f"dashboard/pages/{page}.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page has no title"


@pytest.mark.skipif(not HAVE_BUNDLE, reason="frozen study bundle not available")
def test_overview_labels_metrics_with_evidence():
    at = AppTest.from_file(str(ROOT / "thesis_dashboard.py"), default_timeout=300)
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert "Independent ground truth: not yet available" in html
    if D.available()["legacy_run"]:
        assert "Pseudo-label agreement" in html and "AP 0.53" in html


@pytest.mark.skipif(not HAVE_BUNDLE, reason="frozen study bundle not available")
def test_segmentation_object_selection_outlines_without_error():
    n = D.nodes("2013")
    lo, hi = D.tile_window("tile_00002")
    obj = int(n[(n.px >= lo) & (n.px < hi)].sort_values("area_m2").segment_id.iloc[-1])
    at = AppTest.from_file(str(ROOT / "thesis_dashboard.py"), default_timeout=300)
    at.run()
    at.switch_page("dashboard/pages/segmentation.py")
    at.session_state["seg_obj"] = obj
    at.run()
    assert not at.exception
    assert any(f"Selected object {obj}" in c.value for c in at.caption)


@pytest.mark.skipif(not HAVE_BUNDLE, reason="frozen study bundle not available")
def test_tile_windows_cover_the_strip_in_order():
    t = D.tiles()
    assert list(t.tile) == D.TILES and t.col0.is_monotonic_increasing and int(t.width.iloc[0]) == 1024
    assert (t.col0.iloc[-1] + t.width.iloc[-1]) == D.strip_segments("2013").shape[1]
