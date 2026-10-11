"""Shared presentation helpers: evidence badges, cards, image comparison, overlays and plotly map figures."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from dashboard import data as D

# Evidence types. Every metric on the dashboard carries one of these.
EVIDENCE = {
    "pseudo": ("Pseudo-label agreement", "#b45309", "#fef3c7",
               "Agreement with labels produced by our own rule layer. Not accuracy against ground truth."),
    "visual": ("Visual audit (not independent)", "#6d28d9", "#ede9fe",
               "Visual judgements made during development (assistant or author). Not an independent human reference."),
    "osm": ("OSM 2023 reference (partial)", "#0369a1", "#e0f2fe",
            "OpenStreetMap building footprints, 2023 only, incomplete. Independent of our pipeline, but not complete ground truth."),
    "external": ("External annotated benchmark", "#047857", "#d1fae5",
                 "Professionally annotated external dataset (LsSCD-Ex). Does not measure accuracy on Al Shouf imagery."),
    "none": ("Independent ground truth: not yet available", "#b91c1c", "#fee2e2",
             "No independent human annotation of Al Shouf change exists yet; accuracy on the study area is unmeasured."),
    "descriptive": ("Descriptive (no accuracy claim)", "#374151", "#f3f4f6",
                    "Counts and areas of pipeline outputs. Describe what the pipeline produced, not whether it is correct."),
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.sg-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;
           border:1px solid; margin:0 6px 4px 0; white-space:nowrap;}
.sg-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 16px; height:100%;}
.sg-card .v {font-size:1.6rem; font-weight:700; line-height:1.2;}
.sg-card .l {font-size:.85rem; opacity:.75; margin-bottom:4px;}
.sg-card .sg-badge {font-size:.68rem; white-space:normal; padding:1px 8px;}
.sg-flow {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:.4rem 0 1rem 0;}
.sg-step {padding:6px 12px; border-radius:8px; background:rgba(37,99,235,.10); border:1px solid rgba(37,99,235,.35);
          font-weight:600; font-size:.88rem;}
.sg-arrow {opacity:.5;}
.sg-legend span {display:inline-block; margin-right:14px; font-size:.85rem;}
.sg-legend i {display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:5px; vertical-align:-1px;}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str) -> str:
    label, fg, bg, tip = EVIDENCE[kind]
    return f'<span class="sg-badge" title="{tip}" style="color:{fg};background:{bg};border-color:{fg}">{label}</span>'


def badges(*kinds: str, note: str | None = None):
    html = "".join(badge(k) for k in kinds)
    st.markdown(html + (f'<span style="font-size:.82rem;opacity:.75">{note}</span>' if note else ""), unsafe_allow_html=True)


def card(value, label: str, evidence: str | None = None):
    b = badge(evidence) if evidence else ""
    st.markdown(f'<div class="sg-card"><div class="v">{value}</div><div class="l">{label}</div>{b}</div>', unsafe_allow_html=True)


def flow(steps: list[str]):
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="sg-step">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="sg-arrow">→</span>')
    st.markdown(f'<div class="sg-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def class_legend(class_ids=None):
    ids = class_ids or list(D.CLASSES)
    items = "".join(f'<span><i style="background:{D.hexcolor(i)}"></i>{D.LABELS[D.CLASSES[i]]}</span>' for i in ids)
    st.markdown(f'<div class="sg-legend">{items}</div>', unsafe_allow_html=True)


def legend(items: dict):
    html = "".join(f'<span><i style="background:{c}"></i>{k}</span>' for k, c in items.items())
    st.markdown(f'<div class="sg-legend">{html}</div>', unsafe_allow_html=True)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _b64(arr: np.ndarray, quality=88) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def compare(left: np.ndarray, right: np.ndarray, labels=("2013", "2023"), mode="swipe", height=640, key="cmp"):
    """Interactive before/after viewer (swipe slider or blink) rendered client-side; images stay local."""
    a, b = _b64(left), _b64(right)
    h, w = left.shape[:2]
    if mode == "swipe":
        body = f"""
<div id="{key}" style="position:relative;width:100%;max-width:{height}px;aspect-ratio:{w}/{h};user-select:none;margin:auto">
  <img src="data:image/jpeg;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">
  <div id="{key}c" style="position:absolute;inset:0;overflow:hidden;width:50%">
    <img src="data:image/jpeg;base64,{a}" style="position:absolute;top:0;left:0;height:100%;width:auto;max-width:none" id="{key}a"></div>
  <div id="{key}h" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:#fff;box-shadow:0 0 6px #000;cursor:ew-resize"></div>
  <span style="position:absolute;left:8px;top:8px;background:#000a;color:#fff;padding:2px 8px;border-radius:6px;font:600 13px sans-serif">{labels[0]}</span>
  <span style="position:absolute;right:8px;top:8px;background:#000a;color:#fff;padding:2px 8px;border-radius:6px;font:600 13px sans-serif">{labels[1]}</span>
</div>
<script>
const box=document.getElementById("{key}"),clip=document.getElementById("{key}c"),hd=document.getElementById("{key}h"),ia=document.getElementById("{key}a");
function size(){{ia.style.width=box.clientWidth+"px";}} size(); window.addEventListener("resize",size);
function mv(x){{const r=box.getBoundingClientRect();let p=Math.min(Math.max((x-r.left)/r.width,0),1)*100;clip.style.width=p+"%";hd.style.left=p+"%";}}
let drag=false; box.addEventListener("mousedown",e=>{{drag=true;mv(e.clientX)}}); window.addEventListener("mouseup",()=>drag=false);
window.addEventListener("mousemove",e=>{{if(drag)mv(e.clientX)}}); box.addEventListener("touchmove",e=>mv(e.touches[0].clientX));
</script>"""
    else:
        body = f"""
<div style="position:relative;width:100%;max-width:{height}px;aspect-ratio:{w}/{h};margin:auto">
  <img id="{key}b" src="data:image/jpeg;base64,{a}" style="position:absolute;inset:0;width:100%;height:100%">
  <span id="{key}t" style="position:absolute;left:8px;top:8px;background:#000a;color:#fff;padding:2px 8px;border-radius:6px;font:600 13px sans-serif">{labels[0]}</span>
</div>
<script>
let s=0;const im=document.getElementById("{key}b"),t=document.getElementById("{key}t");
setInterval(()=>{{s=1-s;im.src=s?"data:image/jpeg;base64,{b}":"data:image/jpeg;base64,{a}";t.textContent=s?"{labels[1]}":"{labels[0]}";}},900);
</script>"""
    components.html(body, height=int(height * h / w) + 12 if w >= h else height + 12)


def colorize(classes: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for k, c in D.COLORS.items():
        lut[k] = c
    return lut[classes]


def overlay(rgb: np.ndarray, classes: np.ndarray, alpha: float, keep=None, boundaries=None) -> np.ndarray:
    col = colorize(classes)
    mask = classes > 0 if keep is None else np.isin(classes, keep)
    out = rgb.copy()
    out[mask] = (rgb[mask] * (1 - alpha) + col[mask] * alpha).astype(np.uint8)
    if boundaries is not None:
        b = (np.diff(boundaries, axis=0, prepend=boundaries[:1]) != 0) | (np.diff(boundaries, axis=1, prepend=boundaries[:, :1]) != 0)
        out[b] = (255, 255, 255)
    return out


def outline(rgb: np.ndarray, mask: np.ndarray, color=(255, 220, 0), width=2) -> np.ndarray:
    out = rgb.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, color, width)
    return out


def image_figure(rgb: np.ndarray, height=650, scale=0.5):
    """Plotly figure with the image as a pixel-space background (y down), downsampled for transfer."""
    import plotly.graph_objects as go
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    fig = go.Figure()
    fig.add_layout_image(dict(source=Image.fromarray(small), xref="x", yref="y", x=0, y=0, sizex=w, sizey=h,
                              sizing="stretch", layer="below"))
    fig.update_xaxes(range=[0, w], visible=False, constrain="domain")
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02),
                      plot_bgcolor="rgba(0,0,0,0)", dragmode="pan")
    return fig


def tile_picker(key: str, default="tile_00002", label="Tile") -> str:
    opts = D.TILES
    return st.selectbox(label, opts, index=opts.index(default), key=key)


def need(flag: str, what: str) -> bool:
    if not D.available()[flag]:
        st.warning(f"{what} is not available on this machine (see the Reproduce page for the command that creates it).")
        return False
    return True
