"""
Script 99 — Sentralisert data-refresh for hele Oljepris-prosjektet.

OVERSIKT:
  Identifiserer hvilke data-kilder som er utdaterte, henter ny data via
  eksisterende fetcher-scripts, og rebuilder downstream-pipeline ved behov.

BRUK:
  python scripts/99_refresh_data.py                # Vis status (dry-run)
  python scripts/99_refresh_data.py --refresh-stale # Hent kun utdatert data
  python scripts/99_refresh_data.py --refresh-all   # Hent alt på nytt
  python scripts/99_refresh_data.py --only steo,sodir-geo  # Spesifikke kilder
  python scripts/99_refresh_data.py --rebuild       # Re-bygg panel + modeller etter fetch

DATAKILDER:
  Markedsdata:    Brent, crack spreads, VIX, FX (daily/weekly)
  Fundamentaler:  EIA stocks/refinery util, STEO product demand (monthly)
  Produksjon:     Sodir månedlig per felt (monthly)
  Normpris:       Petroleumsprisrådet (quarterly)
  Geometri:       Sodir felt/lisens-polygoner (annual)
  Assays:         Equinor/ExxonMobil/TotalEnergies XLSX (annual+)
"""

from pathlib import Path
import argparse
import datetime as dt
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"


# ── DATA SOURCES REGISTRY ───────────────────────────────────────────────────
# Hver entry beskriver:
#   name:           kort identifier
#   description:    forklaring til brukeren
#   fetcher:        scriptnavn (i scripts/)
#   outputs:        liste med output-filer (for å sjekke om den finnes)
#   max_age_days:   etter dette anses dataen som utdatert
#   frequency:      'daily', 'weekly', 'monthly', 'quarterly', 'annual'
#   group:          for filtrering: 'market', 'production', 'fundamentals', etc.

DATA_SOURCES = [
    # ── MARKEDSDATA (oppdateres daglig/ukentlig) ──────────────────────────
    {
        "name":        "brent",
        "description": "Brent spot- og forward-pris (EIA)",
        "fetcher":     "02_fetch_brent.py",
        "outputs":     ["data/raw/brent_spot_eia.csv"],
        "max_age_days": 7,
        "frequency":   "daily",
        "group":       "market",
    },
    {
        "name":        "global-prices",
        "description": "Globale crude-priser (WTI, Dubai, Mars Blend etc.)",
        "fetcher":     "23_fetch_global_prices.py",
        "outputs":     ["data/processed/global_differentials_monthly.csv"],
        "max_age_days": 14,
        "frequency":   "weekly",
        "group":       "market",
    },
    {
        "name":        "market-controls",
        "description": "VIX, FX, makro-controls",
        "fetcher":     "24_fetch_market_controls.py",
        "outputs":     ["data/processed/market_controls_monthly.csv"],
        "max_age_days": 7,
        "frequency":   "daily",
        "group":       "market",
    },
    {
        "name":        "crack-spreads",
        "description": "Gasoline/diesel/jet crack-spreads vs. Brent",
        "fetcher":     "29_fetch_crack_spreads.py",
        "outputs":     ["data/raw/eia_products"],
        "max_age_days": 14,
        "frequency":   "weekly",
        "group":       "market",
    },
    {
        "name":        "forward-curve",
        "description": "Brent forward-kurve + contango/backwardation-dummies",
        "fetcher":     "32_forward_curve_and_logistics.py",
        "outputs":     ["data/raw/grade_logistics.csv"],
        "max_age_days": 14,
        "frequency":   "weekly",
        "group":       "market",
    },

    # ── FUNDAMENTALER (EIA, månedlig/ukentlig) ────────────────────────────
    {
        "name":        "eia-fundamentals",
        "description": "EIA US crude stocks, refinery util, exports",
        "fetcher":     "31_fetch_eia_fundamentals.py",
        "outputs":     ["data/raw/eia_fundamentals/us_crude_stocks_kbbl.xls",
                        "data/raw/eia_fundamentals/us_refinery_util_pct.xls"],
        "max_age_days": 14,
        "frequency":   "weekly",
        "group":       "fundamentals",
    },
    {
        "name":        "eia-imports",
        "description": "EIA monthly landed costs per crude grade",
        "fetcher":     "27_fetch_eia_imports.py",
        "outputs":     ["data/raw/eia_imports"],
        "max_age_days": 30,
        "frequency":   "monthly",
        "group":       "fundamentals",
    },
    {
        "name":        "steo-demand",
        "description": "EIA STEO product demand + 2-års forward forecasts",
        "fetcher":     "53_fetch_product_demand.py",
        "outputs":     ["data/raw/eia_demand/steo_full.xlsx",
                        "data/processed/53_product_demand.csv"],
        "max_age_days": 30,
        "frequency":   "monthly",
        "group":       "fundamentals",
    },

    # ── NCS-SPESIFIKKE KILDER ─────────────────────────────────────────────
    {
        "name":        "sodir-production",
        "description": "Sodir månedlig produksjon per felt",
        "fetcher":     "09_dno_sodir_production.py",
        "outputs":     ["data/raw/sodir/sodir_field_production_monthly.csv"],
        "max_age_days": 60,
        "frequency":   "monthly",
        "group":       "production",
    },
    {
        "name":        "normpris",
        "description": "Petroleumsprisrådet normpris-differensialer (kvartalsvis)",
        "fetcher":     "03_fetch_normpris_differentials.py",
        "outputs":     ["data/processed/normpris_differentials_long.csv"],
        "max_age_days": 100,
        "frequency":   "quarterly",
        "group":       "production",
    },
    {
        "name":        "sodir-geodata",
        "description": "Sodir felt/lisens/funn-polygoner (statisk)",
        "fetcher":     "48_fetch_sodir_geodata.py",
        "outputs":     ["data/raw/sodir_geo/fields.geojson",
                        "data/raw/sodir_geo/discoveries.geojson",
                        "data/raw/sodir_geo/licences.geojson"],
        "max_age_days": 365,
        "frequency":   "annual",
        "group":       "production",
    },

    # ── ASSAY-DATABASER (operatørene oppdaterer årlig+) ──────────────────
    {
        "name":        "assays-equinor",
        "description": "Equinor crude assays (XLSX) — offisielle lab-data",
        "fetcher":     "35_fetch_equinor_assays.py",
        "outputs":     ["data/raw/equinor_assays", "data/raw/equinor_assay_index.json"],
        "max_age_days": 365,
        "frequency":   "annual",
        "group":       "assays",
    },
    {
        "name":        "assays-exxonmobil",
        "description": "ExxonMobil crude assays",
        "fetcher":     "37_fetch_exxonmobil_assays.py",
        "outputs":     ["data/raw/exxonmobil_assays", "data/raw/exxonmobil_assays_parsed.csv"],
        "max_age_days": 365,
        "frequency":   "annual",
        "group":       "assays",
    },
    {
        "name":        "assays-totalenergies",
        "description": "TotalEnergies crude assays",
        "fetcher":     "40_fetch_totalenergies_assays.py",
        "outputs":     ["data/raw/totalenergies_assays", "data/raw/totalenergies_assays_parsed.csv"],
        "max_age_days": 365,
        "frequency":   "annual",
        "group":       "assays",
    },
]


# Downstream rebuild scripts (kjøres etter raw data er hentet)
REBUILD_PIPELINE = [
    ("25_assemble_panel.py",       "Re-bygg regression panel fra rådata"),
    ("47_two_model_system.py",     "Re-tren Modell A + B"),
    ("50_product_demand_interactions.py", "Re-tren med yield × crack-interaksjoner"),
    ("62_finalize_model_v3.py",    "Finaliser modell v3 (is_fpso)"),
    ("42_akrbp_realized_price_decomposition.py", "Re-kjør AKRBP-dekomponering"),
    ("43_akrbp_forward_prediction.py",          "Re-kjør forward forecast"),
    ("45b_field_comparison_presentation.py",     "Re-generer normpris-graf"),
    ("49_interactive_ncs_map.py",                "Re-generer interaktivt kart"),
    ("55_combined_forward_forecast.py",         "Re-kjør Stage1+Stage2 forecast"),
]


# ────────────────────────────────────────────────────────────────────────────
# STATUS-SJEKK
# ────────────────────────────────────────────────────────────────────────────

def file_age_days(path: Path) -> float | None:
    """Returnerer alder i dager, eller None hvis filen ikke finnes."""
    if not path.exists():
        return None
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return (dt.datetime.now() - mtime).total_seconds() / 86400


def check_source_status(src: dict) -> dict:
    """Sjekk om en datakilde er fresh / stale / missing."""
    output_files = [PROJECT_ROOT / p for p in src["outputs"]]
    ages = [file_age_days(p) for p in output_files]

    if all(a is None for a in ages):
        status = "MISSING"
        oldest = None
    elif any(a is None for a in ages):
        status = "PARTIAL"
        oldest = max(a for a in ages if a is not None)
    else:
        oldest = max(ages)
        status = "FRESH" if oldest <= src["max_age_days"] else "STALE"

    return {
        "name":      src["name"],
        "status":    status,
        "age_days":  round(oldest, 1) if oldest is not None else None,
        "max_age":   src["max_age_days"],
        "frequency": src["frequency"],
        "group":     src["group"],
    }


def print_status(sources_status: list[dict]) -> None:
    """Print pretty status-tabell."""
    groups = ["market", "fundamentals", "production", "assays"]
    print(f"\n  {'KILDE':<22} {'STATUS':<10} {'ALDER':>10} {'MAKS':>8} {'FREKV':>10}")
    print(f"  {'-' * 70}")
    for g in groups:
        print(f"\n  ── {g.upper()} ──")
        for s in sources_status:
            if s["group"] != g:
                continue
            age = f"{s['age_days']:.1f} d" if s["age_days"] is not None else "-"
            color = {
                "FRESH":   "✓",
                "STALE":   "⚠",
                "MISSING": "✗",
                "PARTIAL": "△",
            }.get(s["status"], "?")
            print(f"  {s['name']:<22} {color} {s['status']:<8} {age:>10} "
                  f"{s['max_age']:>5} d {s['frequency']:>10}")


# ────────────────────────────────────────────────────────────────────────────
# FETCHING
# ────────────────────────────────────────────────────────────────────────────

def run_fetcher(src: dict) -> bool:
    """Kjør fetcher-scriptet for én datakilde. Returnerer True hvis vellykket."""
    script_path = SCRIPTS_DIR / src["fetcher"]
    if not script_path.exists():
        print(f"  ⚠ Fetcher ikke funnet: {src['fetcher']}")
        return False

    print(f"\n  → Henter {src['name']} ({src['fetcher']})...")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,   # 10 min max per fetcher
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  ✓ {src['name']} ferdig på {elapsed:.0f}s")
            return True
        else:
            print(f"  ✗ {src['name']} feilet (exit {result.returncode})")
            err_tail = result.stderr.strip().split("\n")[-3:] if result.stderr else []
            for line in err_tail:
                print(f"     {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ {src['name']} timeout etter 10 min")
        return False
    except Exception as e:
        print(f"  ✗ {src['name']} feilet: {e}")
        return False


def run_rebuild_step(script: str, desc: str) -> bool:
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        print(f"  · {script}: ikke funnet, hopper over")
        return False
    print(f"\n  → {desc} ({script})...")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=1200,
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  ✓ Ferdig på {elapsed:.0f}s")
            return True
        else:
            print(f"  ✗ Feilet — siste output:")
            for line in (result.stderr or "").strip().split("\n")[-5:]:
                print(f"     {line}")
            return False
    except Exception as e:
        print(f"  ✗ {e}")
        return False


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Refresh data sources for NCS oil model")
    ap.add_argument("--refresh-stale", action="store_true",
                     help="Hent kun utdaterte kilder (default = bare vis status)")
    ap.add_argument("--refresh-all", action="store_true",
                     help="Hent ALT på nytt (force)")
    ap.add_argument("--only", type=str, default="",
                     help="Komma-separert liste av kilde-navn eller grupper")
    ap.add_argument("--rebuild", action="store_true",
                     help="Kjør downstream-pipeline (panel, modeller, outputs) etter fetch")
    ap.add_argument("--rebuild-only", action="store_true",
                     help="Kun rebuild — ikke hent ny rådata")
    args = ap.parse_args()

    print("=" * 75)
    print("  NCS Oil Data Refresh — Script 99")
    print(f"  {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 75)

    # Filter sources
    if args.only:
        filters = [s.strip() for s in args.only.split(",")]
        filtered = [s for s in DATA_SOURCES
                     if s["name"] in filters or s["group"] in filters]
    else:
        filtered = DATA_SOURCES

    # Status
    status_list = [check_source_status(s) for s in filtered]
    print(f"\n[1] STATUS for {len(filtered)} datakilder:")
    print_status(status_list)

    # Hvis rebuild-only, skip fetch
    if args.rebuild_only:
        print(f"\n[2] --rebuild-only flagg satt — hopper over fetch")
    elif args.refresh_all or args.refresh_stale:
        print(f"\n[2] FETCHING...")
        if args.refresh_all:
            to_fetch = filtered
            print(f"  --refresh-all: henter alle {len(to_fetch)} kilder")
        else:
            to_fetch = [s for s, st in zip(filtered, status_list)
                         if st["status"] in ("STALE", "MISSING", "PARTIAL")]
            print(f"  --refresh-stale: henter {len(to_fetch)} utdaterte kilder")

        if not to_fetch:
            print(f"  Ingenting å hente — alle er ferske.")
        else:
            n_ok = 0
            for src in to_fetch:
                if run_fetcher(src):
                    n_ok += 1
            print(f"\n  → Resultat: {n_ok}/{len(to_fetch)} kilder hentet vellykket")
    else:
        print(f"\n[2] Ingen --refresh-stale eller --refresh-all flagg — kun status vist")
        print(f"  Bruk --refresh-stale for å hente kun utdaterte kilder.")
        return

    # Rebuild pipeline
    if args.rebuild or args.rebuild_only:
        print(f"\n[3] REBUILDING pipeline ({len(REBUILD_PIPELINE)} steg)...")
        n_ok = 0
        for script, desc in REBUILD_PIPELINE:
            if run_rebuild_step(script, desc):
                n_ok += 1
        print(f"\n  → Rebuild: {n_ok}/{len(REBUILD_PIPELINE)} steg vellykket")
    else:
        print(f"\n[3] Skip rebuild (bruk --rebuild for å re-trene modeller etterpå)")


if __name__ == "__main__":
    main()
