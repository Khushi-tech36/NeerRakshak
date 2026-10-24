"""NeerRakshak - Rapid Flood Mapping & Settlement Impact (Streamlit dashboard)."""
import datetime as dt
import json
import os

import ee
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import flood_engine as fe
import report
import settlements as stl

st.set_page_config(page_title="NeerRakshak - Rapid Flood Mapping", page_icon="🌊", layout="wide")

STATUS_COLORS = {fe.STATUS_INSIDE: "#d62728", fe.STATUS_NEAR: "#ff9f1c", fe.STATUS_SAFE: "#2ca02c"}
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

EMPTY_DF = pd.DataFrame(columns=[
    "priority", "name", "type", "lat", "lon", "inner_frac", "near_frac",
    "pop_nearby", "exposed_pop_est", "status", "risk_score"])


def load_settlements(cfg, bbox, flood, scale, upload=None, on_status=None):
    """
    OPTIONAL step. Never raises, so the flood analysis is never lost.
    Returns dict(df, warning, detail, note).
    """
    out = {"df": EMPTY_DF.copy(), "warning": None, "detail": None, "note": None}
    truncated = False
    try:
        if cfg["src"].startswith("Open"):
            r = stl.get_osm_settlements(bbox, tuple(cfg["types"] or ["city", "town", "village"]),
                                        on_status=on_status)
            out["note"] = r.note
            if r.df is None:
                out["warning"], out["detail"] = r.warning, r.detail
                return out
            sdf, truncated = r.df, r.truncated
        elif cfg["src"].startswith("Upload"):
            if upload is None:
                raise ValueError("no settlements file uploaded")
            sdf, truncated = stl.from_upload(upload, bbox)
        else:
            sdf, truncated = stl.from_sample(bbox)
    except Exception as ex:
        out["warning"], out["detail"] = stl.UNAVAILABLE_MSG, f"{type(ex).__name__}: {ex}"
        return out

    if len(sdf) == 0:
        out["note"] = "No settlements were found inside this study box."
        return out
    if truncated:
        out["note"] = ((out["note"] + " ") if out["note"] else "") + \
            "More than 3000 settlements - only the largest places were kept."
    try:
        out["df"] = fe.assess_settlements(sdf, flood, cfg["inner_m"], cfg["near_m"], scale, cfg["use_pop"])
    except Exception as ex:
        out["warning"] = "Settlement impact check failed. Flood analysis is still available."
        out["detail"] = f"{type(ex).__name__}: {ex}"
    return out


# ----------------------------------------------------------------------------
# Earth Engine init
# ----------------------------------------------------------------------------
def init_ee(project: str):
    if st.session_state.get("ee_ok"):
        return True, ""
    try:
        sa = None
        try:
            sa = st.secrets.get("gee_service_account", None)
        except Exception:
            sa = None
        if sa:
            info = dict(sa)
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
            ee.Initialize(creds, project=info.get("project_id") or project or None)
        else:
            ee.Initialize(project=project or None)
        st.session_state["ee_ok"] = True
        return True, ""
    except Exception as ex:
        return False, str(ex)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🌊 NeerRakshak")
st.sidebar.caption("Sentinel-1 radar flood mapping + village impact")

project = st.sidebar.text_input("Google Earth Engine project ID", value=os.getenv("GEE_PROJECT", ""),
                                help="Cloud project registered for Earth Engine.")

st.sidebar.subheader("1 · Area")
choice = st.sidebar.selectbox("Study area", list(fe.PRESETS) + ["Custom bounding box"],
                              index=len(fe.PRESETS) - 1)
if choice == "Custom bounding box":
    c1, c2 = st.sidebar.columns(2)
    west = c1.number_input("West", value=87.0, format="%.4f")
    south = c2.number_input("South", value=26.3, format="%.4f")
    east = c1.number_input("East", value=87.6, format="%.4f")
    north = c2.number_input("North", value=26.8, format="%.4f")
    bbox = (west, south, east, north)
else:
    bbox = fe.PRESETS[choice]
st.sidebar.caption(f"Box (W,S,E,N): {bbox}")

st.sidebar.subheader("2 · Dates")
event = st.sidebar.date_input("Flood event date", value=dt.date(2024, 9, 30))
after_days = st.sidebar.slider("Search post-flood images for (days)", 3, 24, 12)
base_days = st.sidebar.slider("Baseline window length (days)", 10, 60, 30)
gap_days = st.sidebar.slider("Gap: baseline end → event (days)", 1, 30, 7)
after_start, after_end = event, event + dt.timedelta(days=after_days)
before_end = event - dt.timedelta(days=gap_days)
before_start = before_end - dt.timedelta(days=base_days)
st.sidebar.caption(f"AFTER: {after_start} → {after_end}\n\nBASELINE: {before_start} → {before_end}")

with st.sidebar.expander("Detection settings"):
    orbit = st.selectbox("Orbit pass", ["AUTO", "ASCENDING", "DESCENDING", "BOTH"])
    water_db = st.slider("Water threshold, VV (dB)", -22.0, -10.0, -16.0, 0.5)
    drop_db = st.slider("Min. drop vs baseline (dB)", 1.0, 8.0, 3.0, 0.5)
    smooth = st.slider("Speckle filter radius (m)", 20, 100, 50, 10)
    min_pix = st.slider("Min. blob size (pixels)", 1, 40, 8)
    max_slope = st.slider("Max. terrain slope (°)", 2.0, 15.0, 5.0, 0.5)
    perm_occ = st.slider("Permanent-water cutoff (JRC % occurrence)", 50, 100, 80)
    use_hand = st.checkbox("Mask high ground with HAND", value=True)
    hand_max = st.slider("HAND limit (m)", 5.0, 40.0, 15.0, 1.0, disabled=not use_hand)

st.sidebar.subheader("3 · Settlements")
src = st.sidebar.radio("Source", ["OpenStreetMap (live)", "Upload CSV / GeoJSON", "Built-in sample (offline demo)"])
osm_types = st.sidebar.multiselect("Place types", ["city", "town", "village", "hamlet"],
                                   default=["city", "town", "village"]) if src.startswith("Open") else []
upload = st.sidebar.file_uploader("CSV (name,lat,lon) or GeoJSON", type=["csv", "geojson", "json"]) \
    if src.startswith("Upload") else None
inner_m = st.sidebar.slider("'Inside flood' radius (m)", 50, 500, 150, 50)
near_m = st.sidebar.slider("'Near flood' radius (m)", 500, 3000, 1000, 100)
use_pop = st.sidebar.checkbox("Estimate exposed population (WorldPop)", value=True)

run = st.sidebar.button("▶ Run flood analysis", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------------
st.title("NeerRakshak · Rapid Flood Mapping")
st.warning("**Decision support only.** Always check image dates and verify on the ground before acting.", icon="⚠️")

if run:
    ok, msg = init_ee(project)
    if not ok:
        st.error("Could not start Google Earth Engine.")
        st.code(msg)
        st.info("Fix: run `earthengine authenticate` in a terminal, enter your Cloud project ID in the sidebar, "
                "or add a service account under `[gee_service_account]` in `.streamlit/secrets.toml`.")
        st.stop()

    if (bbox[2] - bbox[0]) > 1.6 or (bbox[3] - bbox[1]) > 1.6:
        st.error("Study box is too large (> 1.6°). Choose a smaller area for a fast same-day run.")
        st.stop()

    params = fe.FloodParams(
        bbox=bbox, after_start=after_start, after_end=after_end,
        before_start=before_start, before_end=before_end, orbit_pass=orbit,
        water_db=water_db, drop_db=drop_db, smooth_m=smooth, min_pixels=min_pix,
        max_slope=max_slope, perm_occurrence=perm_occ, use_hand=use_hand, hand_max=hand_max,
    )
    # ---- Stage 1: flood analysis (required) ---------------------------------
    try:
        with st.status("Running flood analysis…", expanded=True) as status:
            st.write("🛰️ Finding Sentinel-1 images and detecting floodwater…")
            det = fe.detect_flood(params)
            st.write("📐 Measuring flooded area…")
            area = fe.flood_area_km2(det["flood"], det["aoi"], params.scale)
            aoi_km2 = fe.aoi_area_km2(det["aoi"])
            pop_total = fe.exposed_population(det["flood"], det["aoi"]) if use_pop else None
            st.write("🗺️ Preparing map layers…")
            tiles = fe.tile_urls(det)
            status.update(label="Flood analysis complete", state="complete", expanded=False)
    except Exception as ex:
        st.error(f"Analysis failed: {ex}")
        st.stop()

    cfg = dict(src=src, types=list(osm_types), inner_m=inner_m, near_m=near_m, use_pop=use_pop)
    # Save flood results immediately - they must survive any settlement problem.
    st.session_state["res"] = dict(
        params=params, region=choice, bbox=bbox, flood=det["flood"], aoi=det["aoi"], tiles=tiles,
        area_km2=area, aoi_km2=aoi_km2, pop_total=pop_total, df=EMPTY_DF.copy(), orbit=det["orbit"],
        after_dates=det["after_dates"], before_dates=det["before_dates"],
        inner_m=inner_m, near_m=near_m, settle_cfg=cfg,
        settle_warning=None, settle_detail=None, settle_note=None,
    )

    # ---- Stage 2: settlements (optional, isolated) --------------------------
    with st.status("Loading settlements (optional)…", expanded=True) as status2:
        out = load_settlements(cfg, bbox, det["flood"], params.scale, upload=upload, on_status=st.write)
        st.session_state["res"].update(df=out["df"], settle_warning=out["warning"],
                                       settle_detail=out["detail"], settle_note=out["note"])
        if out["warning"]:
            status2.update(label="Settlements unavailable - flood results are ready", state="error", expanded=False)
        else:
            status2.update(label=f"Settlements checked ({len(out['df'])})", state="complete", expanded=False)

res = st.session_state.get("res")

if not res:
    st.markdown(
        """
### How it works
1. **Pick** a Terai study area and the flood date in the sidebar.
2. **Sentinel-1 radar** (works through monsoon clouds) is compared before vs. after the flood.
3. Permanent water, steep slopes, high ground and tiny noise blobs are removed.
4. Villages and towns are checked against the flood map and **ranked by where to check first**.

Press **▶ Run flood analysis** to start. Tip: use *Built-in sample* for a quick offline-settlements demo.
"""
    )
    st.stop()

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if res.get("settle_warning"):
    st.warning(res["settle_warning"], icon="🏘️")
    if res.get("settle_detail"):
        st.caption(f"Reason: {res['settle_detail']}")
    if res["settle_cfg"]["src"].startswith("Open"):
        left = stl.cooldown_remaining()
        if left > 0:
            st.caption(f"OpenStreetMap is rate-limiting requests. You can retry in ~{int(left) + 1}s.")
        if st.button("🔄 Retry settlement data only", disabled=left > 0):
            with st.status("Retrying settlements…", expanded=True):
                out = load_settlements(res["settle_cfg"], res["bbox"], res["flood"],
                                       res["params"].scale, on_status=st.write)
            res.update(df=out["df"], settle_warning=out["warning"],
                       settle_detail=out["detail"], settle_note=out["note"])
            st.rerun()
if res.get("settle_note"):
    st.info(res["settle_note"], icon="ℹ️")

df: pd.DataFrame = res["df"]
has_settle = len(df) > 0
n_inside = int((df["status"] == fe.STATUS_INSIDE).sum()) if len(df) else 0
n_near = int((df["status"] == fe.STATUS_NEAR).sum()) if len(df) else 0
pct = 100 * res["area_km2"] / res["aoi_km2"] if res["aoi_km2"] else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Flooded area", f"{res['area_km2']:.1f} km²", f"{pct:.1f}% of study box", delta_color="off")
k2.metric("Settlements inside flood", n_inside if has_settle else "n/a")
k3.metric("Settlements near flood", n_near if has_settle else "n/a")
k4.metric("Exposed population (approx.)",
          f"{int(res['pop_total']):,}" if res["pop_total"] is not None else "n/a")
st.caption(
    f"🛰️ AFTER image dates: **{', '.join(res['after_dates'])}** · BASELINE: **{', '.join(res['before_dates'])}** "
    f"· Orbit: **{res['orbit']}** · Region: {res['region']}"
)

tab_map, tab_tbl, tab_rep, tab_val, tab_about = st.tabs(
    ["🗺️ Map", "🏘️ Affected settlements", "📄 Report & export", "✅ Validation", "ℹ️ Method & limits"]
)

with tab_map:
    w, s, e, n = res["bbox"]
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(tiles=ESRI, attr="Esri World Imagery", name="Satellite (Esri)").add_to(m)
    for name, url, show in res["tiles"]:
        folium.TileLayer(tiles=url, attr="Google Earth Engine · Copernicus Sentinel-1", name=name,
                         overlay=True, show=show, opacity=0.75 if "Flood" in name else 1.0).add_to(m)
    folium.Rectangle([[s, w], [n, e]], color="#555", weight=1, fill=False, name="Study box").add_to(m)
    for status_name, color in STATUS_COLORS.items():
        fg = folium.FeatureGroup(name=f"Settlements · {status_name}", show=True)
        for r in df[df["status"] == status_name].itertuples():
            folium.CircleMarker(
                [r.lat, r.lon], radius=6 if status_name != fe.STATUS_SAFE else 4,
                color="#222", weight=1, fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=f"#{r.priority} {r.name} - {status_name}",
                popup=folium.Popup(
                    f"<b>{r.name}</b> ({r.type})<br>{status_name}<br>"
                    f"Flooded within {res['inner_m']} m: {r.inner_frac*100:.0f}%<br>"
                    f"Flooded within {res['near_m']} m: {r.near_frac*100:.0f}%<br>"
                    f"Exposed pop (est.): {int(r.exposed_pop_est):,}", max_width=260),
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[s, w], [n, e]])
    st_folium(m, height=640, use_container_width=True, returned_objects=[])
    st.caption("Toggle *Sentinel-1 BEFORE / AFTER* in the layer control to compare. "
               "🔴 inside flood · 🟠 near flood · 🟢 not affected")

with tab_tbl:
    if df.empty:
        st.info("No settlements to show.")
    else:
        sel = st.multiselect("Show", [fe.STATUS_INSIDE, fe.STATUS_NEAR, fe.STATUS_SAFE],
                             default=[fe.STATUS_INSIDE, fe.STATUS_NEAR])
        view = df[df["status"].isin(sel)].copy()
        view["flooded_inner_%"] = (view["inner_frac"] * 100).round(1)
        view["flooded_near_%"] = (view["near_frac"] * 100).round(1)
        st.dataframe(
            view[["priority", "name", "type", "status", "flooded_inner_%", "flooded_near_%",
                  "exposed_pop_est", "risk_score", "lat", "lon"]],
            use_container_width=True, hide_index=True,
            column_config={
                "flooded_inner_%": st.column_config.ProgressColumn("Flooded % (inner)", min_value=0, max_value=100),
                "flooded_near_%": st.column_config.ProgressColumn("Flooded % (near)", min_value=0, max_value=100),
                "exposed_pop_est": st.column_config.NumberColumn("Exposed pop (est.)", format="%d"),
            },
        )
        st.caption("Priority = Inside → Near → Not affected, then by risk score "
                   "(50% inner flooded share, 30% near flooded share, 20% exposed population).")

with tab_rep:
    p = res["params"]
    summary = dict(
        region=res["region"], after_dates=res["after_dates"], before_dates=res["before_dates"],
        orbit=res["orbit"], area_km2=res["area_km2"], area_pct=pct, n_total=len(df),
        n_inside=n_inside, n_near=n_near,
        pop_text=f"{int(res['pop_total']):,}" if res["pop_total"] is not None else "n/a",
        settings=(f"VV < {p.water_db} dB, drop >= {p.drop_db} dB, slope < {p.max_slope} deg, "
                  f"JRC occ <= {p.perm_occurrence}%, HAND < {p.hand_max if p.use_hand else 'off'}, "
                  f"min blob {p.min_pixels} px, {p.scale} m"),
    )
    st.markdown("Download the settlement list and a one-click impact report.")
    c1, c2, c3 = st.columns(3)
    if has_settle:
        c1.download_button("⬇ Settlements CSV", df.to_csv(index=False).encode("utf-8"),
                           "affected_settlements.csv", "text/csv", use_container_width=True)
        c2.download_button("⬇ Settlements GeoJSON", report.to_geojson(df),
                           "affected_settlements.geojson", "application/geo+json", use_container_width=True)
    else:
        c1.caption("Settlement exports unavailable (no settlement data).")
    c3.download_button("⬇ PDF report", report.build_pdf(summary, df),
                       "flood_impact_report.pdf", "application/pdf", use_container_width=True)
    st.json({k: v for k, v in summary.items() if k != "settings"} | {"settings": summary["settings"]}, expanded=False)

with tab_val:
    st.markdown("Compare the detected flood with a **reference flood layer** (e.g. Copernicus EMS / UNOSAT "
                "polygons exported as GeoJSON). Choose a study box that matches the reference extent.")
    ref = st.file_uploader("Reference flood GeoJSON", type=["geojson", "json"], key="ref")
    if ref and st.button("Compute accuracy"):
        try:
            with st.spinner("Comparing…"):
                v = fe.validate(res["flood"], res["aoi"], json.loads(ref.getvalue().decode("utf-8")),
                                res["params"].scale)
            a, b, c = st.columns(3)
            a.metric("IoU", f"{v['IoU']:.2f}")
            b.metric("Precision", f"{v['Precision']:.2f}")
            c.metric("Recall", f"{v['Recall']:.2f}")
            st.caption(f"Predicted {v['Predicted km2']:.1f} km² vs reference {v['Reference km2']:.1f} km².")
        except Exception as ex:
            st.error(f"Validation failed: {ex}")

with tab_about:
    st.markdown(
        """
**Method**
- Data: Sentinel-1 GRD (IW, VV) from Google Earth Engine; medians of the baseline and post-flood windows,
  same orbit pass, speckle-filtered.
- Flood = dark after (VV below threshold) **and** not dark before **and** clear drop vs. baseline.
- Masks: JRC permanent water, SRTM slope (radar shadow), MERIT HAND (high ground), minimum blob size.
- Impact: share of flooded pixels within an *inside* and *near* radius of each settlement point;
  exposed population = WorldPop 100 m population in the near buffer × flooded share (rough estimate).

**Limits (be honest in the demo)**
- "Same-day" depends on when Sentinel-1 passes; always read the image dates above.
- Radar can miss flooding under dense vegetation / inside built-up areas and may flag smooth dark surfaces (sand, tarmac).
- Settlement points are approximate; missing villages in OSM are not analysed.
- Results are decision support, not a rescue instruction.
"""
    )
