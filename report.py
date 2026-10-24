"""report.py - PDF + GeoJSON exports."""
from __future__ import annotations

import datetime as dt
import io
import json
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _latin(s) -> str:
    """Built-in PDF fonts cannot render Devanagari etc. - keep Latin chars only."""
    out = "".join(ch if ord(ch) < 256 else "?" for ch in str(s))
    return escape(out)


def to_geojson(df: pd.DataFrame) -> str:
    feats = []
    for r in df.itertuples():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
            "properties": {
                "priority": r.priority, "name": r.name, "type": r.type, "status": r.status,
                "flooded_pct_inner": round(r.inner_frac * 100, 1),
                "flooded_pct_near": round(r.near_frac * 100, 1),
                "exposed_pop_est": r.exposed_pop_est, "risk_score": r.risk_score,
            },
        })
    return json.dumps({"type": "FeatureCollection", "features": feats})


def build_pdf(summary: dict, df: pd.DataFrame, top_n: int = 40) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=28, rightMargin=28,
                            topMargin=28, bottomMargin=28)
    ss = getSampleStyleSheet()
    el = [
        Paragraph("NeerRakshak - Rapid Flood Impact Report", ss["Title"]),
        Paragraph(f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M}", ss["Normal"]),
        Spacer(1, 8),
    ]
    rows = [
        ["Study area", _latin(summary["region"])],
        ["Sentinel-1 AFTER image dates", _latin(", ".join(summary["after_dates"]))],
        ["Sentinel-1 BASELINE image dates", _latin(", ".join(summary["before_dates"]))],
        ["Orbit pass", summary["orbit"]],
        ["Estimated flooded area", f"{summary['area_km2']:.2f} km2 ({summary['area_pct']:.1f}% of study box)"],
        ["Settlements analysed", str(summary["n_total"]) if summary["n_total"] else
         "not available (settlement data unavailable - flood analysis only)"],
        ["Inside flood / Near flood", f"{summary['n_inside']} / {summary['n_near']}" if summary["n_total"] else "n/a"],
        ["Approx. exposed population", summary["pop_text"]],
        ["Detection settings", _latin(summary["settings"])],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", ss["Normal"]), Paragraph(b, ss["Normal"])] for a, b in rows],
              colWidths=[190, 600])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el += [t, Spacer(1, 12)]
    if len(df):
        el.append(Paragraph(f"Top {top_n} settlements to check first", ss["Heading2"]))
        data = [["#", "Settlement", "Type", "Status", "Flooded % (inner)", "Flooded % (near)",
                 "Exposed pop (est.)", "Lat", "Lon"]]
        for r_ in df.head(top_n).itertuples():
            data.append([r_.priority, _latin(r_.name)[:40], r_.type, r_.status,
                         f"{r_.inner_frac*100:.0f}", f"{r_.near_frac*100:.0f}",
                         f"{int(r_.exposed_pop_est):,}", f"{r_.lat:.4f}", f"{r_.lon:.4f}"])
        tb = Table(data, repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a6b")),
                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTSIZE", (0, 0), (-1, -1), 8),
                 ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]
        for i, r_ in enumerate(df.head(top_n).itertuples(), start=1):
            c = {"Inside flood": "#fde2e2", "Near flood": "#fff0d9"}.get(r_.status)
            if c:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(c)))
        tb.setStyle(TableStyle(style))
        el += [tb, Spacer(1, 12)]
    else:
        el += [Paragraph("Settlement data was temporarily unavailable. The flood extent and area above "
                         "are still valid; re-run the settlement check later.", ss["Normal"]),
               Spacer(1, 12)]
    el += [Paragraph(
        "<b>Decision support only.</b> This map is derived from satellite radar and open data. "
        "It can miss flooding under dense vegetation/buildings and may flag non-flood dark surfaces. "
        "Verify on the ground before operational decisions.", ss["Normal"])]
    doc.build(el)
    return buf.getvalue()
