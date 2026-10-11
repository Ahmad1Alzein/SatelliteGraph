import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("1 · Satellite images")
st.caption("Co-registered and radiometrically harmonised RGB imagery of the frozen study (2013 GeoEye / 2023), "
           "EPSG:32636, 1.5 m source resampled to the 0.75 m segmentation grid.")
if not ui.need("bundle", "The frozen study bundle"):
    st.stop()

c = st.columns([2, 2, 3])
with c[0]:
    tile = ui.tile_picker("img_tile")
mode = c[1].radio("Comparison", ["Swipe", "Blink", "Side by side"], horizontal=True, key="img_mode")
x0, x1 = D.tile_window(tile)
a, b = D.strip_image("2013")[:, x0:x1], D.strip_image("2023")[:, x0:x1]

if mode == "Side by side":
    l, r = st.columns(2)
    l.image(a, caption=f"{tile} · 2013", width="stretch")
    r.image(b, caption=f"{tile} · 2023", width="stretch")
else:
    st.caption("Drag the white handle to compare." if mode == "Swipe" else "The view alternates between 2013 and 2023.")
    ui.compare(a, b, mode=mode.lower(), height=760, key=f"cmp_{tile}_{mode}")

meta = D.tiles().set_index("tile").loc[tile]
st.dataframe({"Property": ["Bounds (EPSG:32636)", "Size", "Pixel size", "Centre (WGS84)"],
              "Value": [f"{meta.left:.0f}, {meta.bottom:.0f} – {meta.right:.0f}, {meta.top:.0f} m",
                        f"{(meta.right - meta.left):.0f} × {(meta.top - meta.bottom):.0f} m",
                        "0.75 m (resampled from 1.5 m)", f"{meta.lat:.5f} N, {meta.lon:.5f} E"]},
             hide_index=True, width="stretch")
d1, d2 = st.columns(2)
d1.download_button("Download 2013 tile (PNG)", ui.png_bytes(a), file_name=f"{tile}_2013.png", mime="image/png")
d2.download_button("Download 2023 tile (PNG)", ui.png_bytes(b), file_name=f"{tile}_2023.png", mime="image/png")
st.info("Spring 2013 vs dry-summer 2023: vegetation appearance differs by season, and buildings lean differently "
        "(different view angles). Both create apparent changes that are not land-cover change.")
