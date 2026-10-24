"""
flood_engine.py
Sentinel-1 SAR flood detection + settlement impact assessment on Google Earth Engine.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import ee
import pandas as pd

# (west, south, east, north) - approximate study boxes across the Terai
PRESETS = {
    "Kailali - Tikapur / Dhangadhi": (80.5, 28.4, 81.3, 28.9),
    "Banke / Bardiya - Nepalgunj / Gulariya": (81.2, 28.0, 81.9, 28.5),
    "Rupandehi / Kapilvastu - Lumbini": (83.0, 27.4, 83.6, 27.8),
    "Chitwan / Nawalpur": (83.8, 27.4, 84.7, 27.8),
    "Parsa / Bara - Birgunj": (84.6, 26.9, 85.2, 27.3),
    "Dhanusha / Mahottari - Janakpur": (85.5, 26.6, 86.2, 27.0),
    "Saptari / Siraha - Rajbiraj": (86.4, 26.4, 87.1, 26.9),
    "Sunsari / Morang - Koshi / Biratnagar": (87.0, 26.3, 87.6, 26.8),
}

STATUS_INSIDE = "Inside flood"
STATUS_NEAR = "Near flood"
STATUS_SAFE = "Not affected"


@dataclass
class FloodParams:
    bbox: tuple  # (west, south, east, north)
    after_start: dt.date
    after_end: dt.date
    before_start: dt.date
    before_end: dt.date
    orbit_pass: str = "AUTO"      # AUTO | ASCENDING | DESCENDING | BOTH
    water_db: float = -16.0       # VV below this = water
    drop_db: float = 3.0          # must have dropped at least this much vs. baseline
    smooth_m: int = 50            # speckle filter radius (metres)
    min_pixels: int = 8           # remove isolated blobs smaller than this
    max_slope: float = 5.0        # degrees; steeper = radar shadow, not flood
    perm_occurrence: int = 80     # JRC % occurrence above which water is "permanent"
    use_hand: bool = True         # height-above-nearest-drainage mask
    hand_max: float = 15.0        # metres
    scale: int = 30               # analysis resolution (metres)

    def aoi(self) -> ee.Geometry:
        return ee.Geometry.Rectangle(list(self.bbox), proj="EPSG:4326", geodesic=False)


# --------------------------------------------------------------------------- #
# Sentinel-1 helpers
# --------------------------------------------------------------------------- #
def _collection(aoi, start: dt.date, end: dt.date, orbit: str):
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )
    if orbit in ("ASCENDING", "DESCENDING"):
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))
    return col


def _dates(col) -> list[str]:
    ms = col.aggregate_array("system:time_start").getInfo() or []
    return sorted({dt.datetime.utcfromtimestamp(m / 1000).strftime("%Y-%m-%d") for m in ms})


def _pick_orbit(p: FloodParams, aoi) -> str:
    """Choose the pass direction that has images in BOTH windows (consistent geometry)."""
    if p.orbit_pass != "AUTO":
        return p.orbit_pass
    best, best_score = "BOTH", -1
    for orbit in ("ASCENDING", "DESCENDING"):
        na = _collection(aoi, p.after_start, p.after_end, orbit).size().getInfo()
        nb = _collection(aoi, p.before_start, p.before_end, orbit).size().getInfo()
        if na > 0 and nb > 0 and (na + nb) > best_score:
            best, best_score = orbit, na + nb
    return best


def _composite(col, aoi, smooth_m):
    return col.select("VV").median().focalMedian(smooth_m, "circle", "meters").clip(aoi)


# --------------------------------------------------------------------------- #
# Flood detection
# --------------------------------------------------------------------------- #
def detect_flood(p: FloodParams) -> dict:
    aoi = p.aoi()
    orbit = _pick_orbit(p, aoi)

    col_after = _collection(aoi, p.after_start, p.after_end, orbit)
    col_before = _collection(aoi, p.before_start, p.before_end, orbit)
    after_dates, before_dates = _dates(col_after), _dates(col_before)

    if not after_dates:
        raise ValueError(
            "No Sentinel-1 image found in the AFTER window. Widen the window or move the event date."
        )
    if not before_dates:
        raise ValueError(
            "No Sentinel-1 image found in the BASELINE window. Increase baseline length or gap."
        )

    after = _composite(col_after, aoi, p.smooth_m)
    before = _composite(col_before, aoi, p.smooth_m)
    diff = after.subtract(before)

    # 1) change detection: dark now, was not dark before, and clearly darker than before
    flood = after.lt(p.water_db).And(before.gt(p.water_db)).And(diff.lt(-p.drop_db))

    # 2) remove permanent / seasonal river & lake water (JRC Global Surface Water)
    occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    flood = flood.And(occurrence.lte(p.perm_occurrence))

    # 3) remove steep terrain (radar shadow looks like water)
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    flood = flood.And(slope.lt(p.max_slope))

    # 4) remove high ground that cannot flood (height above nearest drainage)
    if p.use_hand:
        hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
        flood = flood.And(hand.lt(p.hand_max))

    # 5) fix analysis grid, then drop tiny isolated blobs (speckle leftovers)
    proj = ee.Projection("EPSG:4326").atScale(p.scale)
    flood = flood.selfMask().reproject(proj)
    connected = flood.connectedPixelCount(100, True)
    flood = flood.updateMask(connected.gte(p.min_pixels)).rename("flood")

    return {
        "aoi": aoi,
        "flood": flood,
        "before": before,
        "after": after,
        "diff": diff,
        "orbit": orbit,
        "after_dates": after_dates,
        "before_dates": before_dates,
    }


def flood_area_km2(flood, aoi, scale) -> float:
    r = (
        ee.Image.pixelArea()
        .updateMask(flood)
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e10, bestEffort=True, tileScale=4,
        )
        .get("area")
        .getInfo()
    )
    return float(r or 0) / 1e6


def aoi_area_km2(aoi) -> float:
    return float(aoi.area(1).getInfo()) / 1e6


def tile_urls(det: dict) -> list[tuple[str, str, bool]]:
    """Return [(layer name, tile url, shown_by_default)] for the map."""
    sar_vis = {"min": -25, "max": 0}
    out = []
    for name, img, vis, show in [
        ("Sentinel-1 BEFORE (VV dB)", det["before"], sar_vis, False),
        ("Sentinel-1 AFTER (VV dB)", det["after"], sar_vis, False),
        ("Flood extent", det["flood"], {"min": 0, "max": 1, "palette": ["1e6fff", "1e6fff"]}, True),
    ]:
        mid = img.getMapId(vis)
        out.append((name, mid["tile_fetcher"].url_format, show))
    return out


# --------------------------------------------------------------------------- #
# Population (WorldPop) - optional
# --------------------------------------------------------------------------- #
def _population():
    return (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "NPL"))
        .filter(ee.Filter.eq("year", 2020))
        .mosaic()
        .select("population")
    )


def exposed_population(flood, aoi) -> float | None:
    """Approximate people living on flooded pixels (WorldPop 100 m x flooded fraction)."""
    try:
        pop = _population()
        proj100 = ee.Projection("EPSG:4326").atScale(100)
        frac = (
            flood.unmask(0)
            .reduceResolution(ee.Reducer.mean(), True, 1024)
            .reproject(proj100)
        )
        v = (
            pop.multiply(frac)
            .reduceRegion(ee.Reducer.sum(), aoi, 100, maxPixels=1e10, bestEffort=True, tileScale=4)
            .get("population")
            .getInfo()
        )
        return float(v or 0)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Settlement impact
# --------------------------------------------------------------------------- #
def assess_settlements(df: pd.DataFrame, flood, inner_m: int, near_m: int,
                       scale: int, use_pop: bool = True, chunk: int = 400) -> pd.DataFrame:
    """
    For each settlement compute the share of flooded pixels within
      * inner_m  -> 'Inside flood' if > 0   (point accuracy buffer)
      * near_m   -> 'Near flood'   if > 0
    and an approximate exposed population.
    """
    df = df.copy().reset_index(drop=True)
    flood01 = flood.unmask(0).rename("f")
    pop = None
    if use_pop:
        try:
            pop = _population()
        except Exception:
            pop = None

    def buffered(m):
        return lambda f: f.setGeometry(f.geometry().buffer(m))

    inner_vals, near_vals, pop_vals = {}, {}, {}
    for s in range(0, len(df), chunk):
        part = df.iloc[s:s + chunk]
        pts = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"idx": int(r.Index)})
            for r in part.itertuples()
        ])

        def run(img, reducer, m, sc, prop):
            fc = img.reduceRegions(pts.map(buffered(m)), reducer.setOutputs([prop]), sc)
            feats = fc.select(["idx", prop], None, False).getInfo()["features"]
            return {f["properties"]["idx"]: f["properties"].get(prop) for f in feats}

        inner_vals.update(run(flood01, ee.Reducer.mean(), inner_m, scale, "v"))
        near_vals.update(run(flood01, ee.Reducer.mean(), near_m, scale, "v"))
        if pop is not None:
            try:
                pop_vals.update(run(pop, ee.Reducer.sum(), near_m, 100, "v"))
            except Exception:
                pass

    df["inner_frac"] = df.index.map(lambda i: inner_vals.get(i) or 0.0).astype(float)
    df["near_frac"] = df.index.map(lambda i: near_vals.get(i) or 0.0).astype(float)
    df["pop_nearby"] = df.index.map(lambda i: pop_vals.get(i) or 0.0).astype(float)
    df["exposed_pop_est"] = (df["pop_nearby"] * df["near_frac"]).round(0)

    df["status"] = STATUS_SAFE
    df.loc[df["near_frac"] > 0, "status"] = STATUS_NEAR
    df.loc[df["inner_frac"] > 0, "status"] = STATUS_INSIDE

    # simple transparent priority score (0-100)
    df["risk_score"] = (
        100 * (0.5 * df["inner_frac"].clip(0, 1)
               + 0.3 * df["near_frac"].clip(0, 1)
               + 0.2 * (df["exposed_pop_est"] / 2000).clip(0, 1))
    ).round(1)

    order = {STATUS_INSIDE: 0, STATUS_NEAR: 1, STATUS_SAFE: 2}
    df["_o"] = df["status"].map(order)
    df = df.sort_values(["_o", "risk_score"], ascending=[True, False]).drop(columns="_o")
    df.insert(0, "priority", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validation against a reference flood layer (e.g. Copernicus EMS / UNOSAT)
# --------------------------------------------------------------------------- #
def validate(flood, aoi, geojson: dict, scale: int) -> dict:
    feats = [
        ee.Feature(ee.Geometry(f["geometry"]))
        for f in geojson.get("features", []) if f.get("geometry")
    ]
    if not feats:
        raise ValueError("Reference GeoJSON has no geometries.")
    ref = ee.Image(0).paint(ee.FeatureCollection(feats), 1).rename("ref")
    pred = flood.unmask(0).rename("pred")
    inter = pred.And(ref).rename("inter")
    union = pred.Or(ref).rename("union")
    stack = ee.Image.cat([pred, ref, inter, union]).multiply(ee.Image.pixelArea())
    r = stack.reduceRegion(ee.Reducer.sum(), aoi, scale, maxPixels=1e10,
                           bestEffort=True, tileScale=4).getInfo()
    inter_a, union_a = r.get("inter", 0), r.get("union", 0)
    pred_a, ref_a = r.get("pred", 0), r.get("ref", 0)
    return {
        "IoU": inter_a / union_a if union_a else 0.0,
        "Precision": inter_a / pred_a if pred_a else 0.0,
        "Recall": inter_a / ref_a if ref_a else 0.0,
        "Predicted km2": pred_a / 1e6,
        "Reference km2": ref_a / 1e6,
    }
