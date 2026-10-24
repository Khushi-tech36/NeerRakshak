"""
settlements.py - load village / town points from OSM, uploads or a built-in sample.

Overpass handling (HTTP 429 etc.):
  * one request at a time (process-wide lock) - parallel Streamlit sessions share the result
  * disk + memory cache (24 h) keyed by rounded bbox + place types
  * limited retries (3 attempts) with exponential backoff + jitter; honours Retry-After
  * a cooldown after a failed run so repeated clicks do not hammer the server
  * if live data fails, stale cached data is used when available
  * NEVER raises from get_osm_settlements(): the caller gets a result object with a warning instead
"""
from __future__ import annotations

import email.utils
import hashlib
import io
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

SAMPLE = Path(__file__).parent / "data" / "sample_settlements.csv"
OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
TYPE_RANK = {"city": 0, "town": 1, "village": 2, "hamlet": 3}

UNAVAILABLE_MSG = "Settlement data temporarily unavailable. Flood analysis is still available."

# --- tunables -------------------------------------------------------------
CACHE_DIR = Path(os.getenv("NEERRAKSHAK_CACHE", Path(__file__).parent / ".cache" / "overpass"))
CACHE_TTL_S = 24 * 3600        # reuse results for a day
MAX_ATTEMPTS = 3               # total tries per user action (not per mirror)
BASE_BACKOFF_S = 5.0           # 5 s, 10 s, ... when no Retry-After is given
MAX_SINGLE_WAIT_S = 30.0       # never block the UI longer than this for one wait
MAX_TOTAL_WAIT_S = 60.0        # ... or this across all waits
REQUEST_TIMEOUT_S = 45
DEFAULT_COOLDOWN_S = 60.0      # after a final 429 without Retry-After
FAILURE_COOLDOWN_S = 20.0      # after any other final failure
USER_AGENT = "NeerRakshak-FloodMapping/1.0 (" + os.getenv("NEERRAKSHAK_CONTACT", "hackathon prototype") + ")"

_FETCH_LOCK = threading.Lock()          # only one Overpass request at a time
_STATE_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0
_MEM: dict[str, dict] = {}


def _standardise(df: pd.DataFrame, bbox, max_n: int) -> tuple[pd.DataFrame, bool]:
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    w, s, e, n = bbox
    df = df[(df.lon.between(w, e)) & (df.lat.between(s, n))]
    if "type" not in df:
        df["type"] = "settlement"
    if "name" not in df:
        df["name"] = ""
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    blank = df["name"] == ""
    df.loc[blank, "name"] = "Unnamed " + df.loc[blank, "type"].astype(str)
    df = df.drop_duplicates(subset=["name", "lat", "lon"])
    truncated = len(df) > max_n
    if truncated:
        df = (df.assign(_r=df["type"].map(TYPE_RANK).fillna(9))
                .sort_values("_r").head(max_n).drop(columns="_r"))
    cols = [c for c in ["name", "type", "lat", "lon"] if c in df]
    return df[cols].reset_index(drop=True), truncated


def from_sample(bbox, max_n=3000):
    return _standardise(pd.read_csv(SAMPLE), bbox, max_n)


# --------------------------------------------------------------------------- #
# Overpass: cache, cooldown, retry
# --------------------------------------------------------------------------- #
@dataclass
class SettlementResult:
    df: Optional[pd.DataFrame] = None   # None => unavailable
    truncated: bool = False
    source: str = ""                    # "live" | "cache" | "stale-cache" | ""
    warning: Optional[str] = None       # shown as st.warning when df is None
    detail: Optional[str] = None        # technical reason (caption)
    note: Optional[str] = None          # shown as st.info (e.g. stale cache in use)


class _HttpFail(Exception):
    def __init__(self, status, retry_after=None, msg=""):
        super().__init__(msg or f"HTTP {status}")
        self.status, self.retry_after = status, retry_after
        self.retryable = status in (None, 429, 502, 503, 504)


class SettlementFetchError(Exception):
    pass


def _reset_state():
    """Used by tests."""
    global _COOLDOWN_UNTIL
    with _STATE_LOCK:
        _COOLDOWN_UNTIL = 0.0
        _MEM.clear()


def cooldown_remaining() -> float:
    return max(0.0, _COOLDOWN_UNTIL - time.time())


def _set_cooldown(seconds: float):
    global _COOLDOWN_UNTIL
    with _STATE_LOCK:
        _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.time() + seconds)


def _parse_retry_after(value) -> Optional[float]:
    if not value:
        return None
    value = str(value).strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _cache_key(bbox, types) -> str:
    raw = json.dumps([[round(float(v), 3) for v in bbox], sorted(types)])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _cache_read(key: str) -> Optional[dict]:
    if key in _MEM:
        return _MEM[key]
    try:
        p = CACHE_DIR / f"{key}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            _MEM[key] = data
            return data
    except Exception:
        pass
    return None


def _cache_write(key: str, records: list):
    data = {"ts": time.time(), "records": records}
    _MEM[key] = data
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_DIR / f"{key}.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(CACHE_DIR / f"{key}.json")
    except OSError:
        pass  # read-only filesystem: memory cache still works


def _is_fresh(entry: Optional[dict]) -> bool:
    return bool(entry) and (time.time() - entry["ts"]) < CACHE_TTL_S


def _query(bbox, types, timeout=30) -> str:
    w, s, e, n = bbox
    regex = "|".join(types)
    return (f'[out:json][timeout:{timeout}];'
            f'node["place"~"^({regex})$"]({s},{w},{n},{e});out body qt;')


def _fetch_once(url: str, query: str) -> list:
    try:
        r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT},
                          timeout=REQUEST_TIMEOUT_S)
    except (requests.Timeout, requests.ConnectionError) as ex:
        raise _HttpFail(None, None, f"network error: {type(ex).__name__}")
    if r.status_code != 200:
        raise _HttpFail(r.status_code, _parse_retry_after(r.headers.get("Retry-After")),
                        f"HTTP {r.status_code}")
    try:
        body = r.json()
    except ValueError:
        raise _HttpFail(504, None, "invalid response")
    # Overpass can answer 200 with a 'remark' when the query was killed / overloaded
    if not body.get("elements") and "runtime error" in str(body.get("remark", "")).lower():
        raise _HttpFail(504, None, "Overpass runtime error: " + str(body.get("remark"))[:120])
    return body.get("elements", [])


def _fetch_with_backoff(query: str, say: Callable[[str], None]) -> list:
    total_wait, last = 0.0, None
    for attempt in range(MAX_ATTEMPTS):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            return _fetch_once(url, query)
        except _HttpFail as f:
            last = f
            if not f.retryable or attempt == MAX_ATTEMPTS - 1:
                break
            base = f.retry_after if f.retry_after is not None else BASE_BACKOFF_S * (2 ** attempt)
            wait = base + random.uniform(0.0, 1.0)
            if wait > MAX_SINGLE_WAIT_S or total_wait + wait > MAX_TOTAL_WAIT_S:
                break  # too long to block the UI; cooldown below prevents hammering
            say(f"OpenStreetMap server busy ({f}). Waiting {wait:.0f}s before retry "
                f"{attempt + 2}/{MAX_ATTEMPTS}…")
            time.sleep(wait)
            total_wait += wait
    # all attempts used (or wait too long) -> cooldown, then report
    if last is not None and last.status == 429:
        _set_cooldown(last.retry_after if last.retry_after else DEFAULT_COOLDOWN_S)
    else:
        _set_cooldown(FAILURE_COOLDOWN_S)
    raise SettlementFetchError(str(last) if last else "unknown error")


def _to_df(records: list, bbox, max_n):
    df = pd.DataFrame(records, columns=["name", "type", "lat", "lon"])
    return _standardise(df, bbox, max_n)


def _from_stale(entry, bbox, max_n, reason: str) -> SettlementResult:
    if entry and entry.get("records"):
        df, trunc = _to_df(entry["records"], bbox, max_n)
        age_h = (time.time() - entry["ts"]) / 3600
        return SettlementResult(
            df=df, truncated=trunc, source="stale-cache",
            note=f"Live OpenStreetMap request failed ({reason}). Using saved settlement data "
                 f"from {age_h:.0f} h ago.")
    return SettlementResult(df=None, warning=UNAVAILABLE_MSG, detail=reason)


def get_osm_settlements(bbox, types=("city", "town", "village"), max_n=3000,
                        on_status: Optional[Callable[[str], None]] = None) -> SettlementResult:
    """Fetch settlements from OSM with caching + safe retries. Never raises."""
    say = on_status or (lambda m: None)
    try:
        types = tuple(types) or ("city", "town", "village")
        key = _cache_key(bbox, types)

        entry = _cache_read(key)
        if _is_fresh(entry) and entry.get("records"):
            df, trunc = _to_df(entry["records"], bbox, max_n)
            return SettlementResult(df=df, truncated=trunc, source="cache",
                                    note="Settlements loaded from local cache (no OpenStreetMap request).")

        left = cooldown_remaining()
        if left > 0:
            return _from_stale(entry, bbox, max_n,
                               f"OpenStreetMap rate limit - retry in ~{int(left) + 1}s")

        with _FETCH_LOCK:                       # one Overpass request at a time
            entry = _cache_read(key)            # another session may have just filled it
            if _is_fresh(entry) and entry.get("records"):
                df, trunc = _to_df(entry["records"], bbox, max_n)
                return SettlementResult(df=df, truncated=trunc, source="cache")
            left = cooldown_remaining()
            if left > 0:
                return _from_stale(entry, bbox, max_n,
                                   f"OpenStreetMap rate limit - retry in ~{int(left) + 1}s")
            say("Requesting settlements from OpenStreetMap…")
            try:
                elements = _fetch_with_backoff(_query(bbox, types), say)
            except SettlementFetchError as ex:
                return _from_stale(entry, bbox, max_n, str(ex))

        records = []
        for el in elements:
            t = el.get("tags", {})
            records.append({"name": t.get("name:en") or t.get("name") or "",
                            "type": t.get("place", "settlement"),
                            "lat": el["lat"], "lon": el["lon"]})
        if records:                              # never cache empty answers
            _cache_write(key, records)
        df, trunc = _to_df(records, bbox, max_n)
        return SettlementResult(df=df, truncated=trunc, source="live")
    except Exception as ex:                      # absolute safety net
        return SettlementResult(df=None, warning=UNAVAILABLE_MSG, detail=f"{type(ex).__name__}: {ex}")


def from_osm(bbox, types=("city", "town", "village"), max_n=3000, timeout=90):
    """Backward-compatible wrapper (raises if unavailable). Prefer get_osm_settlements()."""
    r = get_osm_settlements(bbox, types, max_n)
    if r.df is None:
        raise RuntimeError(f"{r.warning} ({r.detail})")
    return r.df, r.truncated


def from_upload(uploaded_file, bbox, max_n=3000):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith((".geojson", ".json")):
        from shapely.geometry import shape
        gj = json.loads(raw.decode("utf-8"))
        rows = []
        for f in gj.get("features", []):
            if not f.get("geometry"):
                continue
            c = shape(f["geometry"]).centroid
            p = f.get("properties") or {}
            rows.append({
                "name": p.get("name") or p.get("name:en") or p.get("NAME") or "",
                "type": p.get("type") or p.get("place") or "settlement",
                "lat": c.y, "lon": c.x,
            })
        df = pd.DataFrame(rows, columns=["name", "type", "lat", "lon"])
    else:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip().lower() for c in df.columns]
        ren = {"latitude": "lat", "y": "lat", "longitude": "lon", "lng": "lon", "long": "lon", "x": "lon"}
        df = df.rename(columns=ren)
        if not {"lat", "lon"} <= set(df.columns):
            raise ValueError("CSV needs columns: name, lat, lon (type optional).")
    return _standardise(df, bbox, max_n)
