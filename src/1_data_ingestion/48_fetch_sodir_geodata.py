"""
Script 48 — Hent felt-polygoner, lisensblokker og funn fra Sodir REST API

KILDE: https://factmaps.sodir.no/api/rest/services/Factmaps/FactMapsWGS84/MapServer

Lag som hentes:
  502 — Field by status                  → felt-polygoner med operatør, status
  503 — Discovery, active by HC type    → aktive funn (Yggdrasil/Hugin/Munin/Fulla)
  616 — Production licence, current     → lisens-polygoner med operatør
  802 — Blocks                           → NCS-blokkrute (referansegrid)

OUTPUT (under data/raw/sodir_geo/):
  fields.geojson
  discoveries.geojson
  licences.geojson
  blocks.geojson

Cacher lokalt — gjenkjør kun ved behov for oppdatering.
"""

from pathlib import Path
import json
import time
import urllib.parse
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR      = PROJECT_ROOT / "data" / "raw" / "sodir_geo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://factmaps.sodir.no/api/rest/services/"
    "Factmaps/FactMapsWGS84/MapServer"
)

LAYERS = {
    "fields":       502,  # Field by status (polygoner)
    "discoveries":  503,  # Aktive funn (under utbygging)
    "licences":     616,  # Production licence current (med polygon)
    "blocks":       802,  # NCS-blokker (referanse-grid)
}

PAGE_SIZE = 1000   # ArcGIS REST cap


def fetch_layer(layer_id: int, name: str) -> dict:
    """Pagine gjennom et lag og samle alle features til ett FeatureCollection."""
    print(f"\n→ Henter lag {layer_id} ({name})...")
    all_features = []
    offset = 0
    while True:
        params = {
            "where":             "1=1",
            "outFields":         "*",
            "f":                 "geojson",
            "resultOffset":      offset,
            "resultRecordCount": PAGE_SIZE,
        }
        url = f"{BASE_URL}/{layer_id}/query?{urllib.parse.urlencode(params)}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        all_features.extend(feats)
        print(f"   sidet {offset:>5} → {offset + len(feats):>5} ({len(feats)} features)")
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)   # Vær snill mot Sodir-API-et

    fc = {
        "type":     "FeatureCollection",
        "features": all_features,
    }
    return fc


def save(fc: dict, name: str) -> None:
    path = OUT_DIR / f"{name}.geojson"
    path.write_text(json.dumps(fc, ensure_ascii=False))
    n = len(fc.get("features", []))
    size_mb = path.stat().st_size / 1e6
    print(f"   ✓ Lagret {n} features → {path.name} ({size_mb:.1f} MB)")


def summarize_fields(fc: dict) -> None:
    """Print en oversikt over feltene per status / operatør."""
    feats = fc["features"]
    status_count = {}
    operator_count = {}
    for f in feats:
        p = f.get("properties", {})
        s = p.get("fldCurrentActivitySatus") or p.get("fldCurrentActivityStatus") or "Unknown"
        o = p.get("cmpLongName") or "Unknown operator"
        status_count[s] = status_count.get(s, 0) + 1
        operator_count[o] = operator_count.get(o, 0) + 1

    print(f"\n   Felt-status-fordeling:")
    for s, n in sorted(status_count.items(), key=lambda x: -x[1]):
        print(f"     {s:<30} {n:>4}")

    print(f"\n   Top 10 operatører (etter antall felt):")
    for o, n in sorted(operator_count.items(), key=lambda x: -x[1])[:10]:
        print(f"     {o:<40} {n:>4}")


def summarize_discoveries(fc: dict) -> None:
    feats = fc["features"]
    print(f"\n   Antall aktive funn: {len(feats)}")
    # Funn knyttet til Yggdrasil-hubene
    ygg = [
        f for f in feats
        if (f.get("properties", {}).get("dscName") or "").upper() in
           {"25/2-10 S HUGIN", "25/2-7 MUNIN", "25/2-13 S FULLA",
            "HUGIN", "MUNIN", "FULLA"}
        or (f.get("properties", {}).get("fldName") or "").upper() in
           {"HUGIN", "MUNIN", "FULLA"}
    ]
    print(f"   Yggdrasil-relaterte funn (Hugin/Munin/Fulla): {len(ygg)}")
    for f in ygg[:5]:
        p = f["properties"]
        print(f"     {p.get('dscName','?'):<30} | felt: {p.get('fldName','?'):<10} "
              f"| status: {p.get('dscCurrentActivityStatus','?')}")


def main() -> None:
    print("=" * 70)
    print("  SCRIPT 48: Hent felt/lisens/funn-geodata fra Sodir REST")
    print("=" * 70)

    for name, layer_id in LAYERS.items():
        out_path = OUT_DIR / f"{name}.geojson"
        if out_path.exists():
            print(f"\n→ {name}.geojson finnes allerede ({out_path.stat().st_size / 1e6:.1f} MB)")
            print(f"   Slett filen for å re-hente. Hopper over.")
            continue
        fc = fetch_layer(layer_id, name)
        save(fc, name)

    # Oppsummering
    fields_fc      = json.loads((OUT_DIR / "fields.geojson").read_text())
    discoveries_fc = json.loads((OUT_DIR / "discoveries.geojson").read_text())
    summarize_fields(fields_fc)
    summarize_discoveries(discoveries_fc)

    print(f"\n  Ferdig. Filer i {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
