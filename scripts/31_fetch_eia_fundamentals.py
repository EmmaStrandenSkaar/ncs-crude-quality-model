"""
Hent EIA weekly petroleum-fundamentals: lagre, raffinerikapasitetsbruk,
import, eksport, produksjon, dager-i-supply. Aggreger til måned.

Disse variablene fanger den tidsvarierende balansen mellom tilbud og
etterspørsel for crude — som ofte forklarer hvorfor differensialer flytter
seg selv når kvaliteten er konstant.

Hvorfor disse er viktige:
  - Refinery utilization > 95% = "ta hva som helst" → smale rabatter
  - Refinery utilization < 85% = picky refineries → brede rabatter
  - Cushing inventories = WTI-spesifikk pris-press
  - OECD/US inventories = global tightness
  - Crude exports = US-dollar dominans i lett-sweet markedet
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "eia_fundamentals"

# EIA weekly XLS-URL-er (alle bekreftet å eksistere)
EIA_SERIES = {
    "us_crude_stocks_kbbl":          "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls",
    "cushing_stocks_kbbl":           "https://www.eia.gov/dnav/pet/hist_xls/W_EPC0_SAX_YCUOK_MBBLw.xls",
    "spr_stocks_kbbl":               "https://www.eia.gov/dnav/pet/hist_xls/WCSSTUS1w.xls",
    "us_refinery_util_pct":          "https://www.eia.gov/dnav/pet/hist_xls/WPULEUS3w.xls",
    "us_refinery_inputs_kbpd":       "https://www.eia.gov/dnav/pet/hist_xls/WCRRIUS2w.xls",
    "us_crude_imports_kbpd":         "https://www.eia.gov/dnav/pet/hist_xls/WCEIMUS2w.xls",
    "us_crude_exports_kbpd":         "https://www.eia.gov/dnav/pet/hist_xls/WCREXUS2w.xls",
    "us_crude_production_kbpd":      "https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls",
    "us_days_supply_crude":          "https://www.eia.gov/dnav/pet/hist_xls/W_EPC0_VSD_NUS_DAYSw.xls",
}


def download(name: str, url: str) -> Path | None:
    dest = CACHE_DIR / f"{name}.xls"
    if dest.exists():
        return dest
    try:
        print(f"  Henter {name} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        print(f"    Feilet ({url}): {e}")
        return None


def parse_weekly_xls(path: Path, value_col_name: str) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(path, engine="xlrd")
    except Exception as e:
        print(f"  Parsing feilet: {e}")
        return pd.DataFrame()

    data_sheets = [s for s in xls.sheet_names if "data" in s.lower()]
    if not data_sheets:
        return pd.DataFrame()

    df = pd.read_excel(xls, sheet_name=data_sheets[0], header=2)
    if df.shape[1] < 2:
        return pd.DataFrame()

    df = df.iloc[:, :2].copy()
    df.columns = ["date", value_col_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")
    df = df.dropna()
    return df


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Henter EIA weekly fundamentals ===\n")
    weekly = {}
    for name, url in EIA_SERIES.items():
        path = download(name, url)
        if path:
            df = parse_weekly_xls(path, name)
            if not df.empty:
                weekly[name] = df
                print(f"  {name:35s}: {len(df):4d} obs ({df['date'].min():%Y-%m} – {df['date'].max():%Y-%m})")

    if not weekly:
        print("Ingen data hentet — avbryter.")
        return

    # === Aggregér til måned ===
    print("\n=== Aggregerer ukentlige data til måned ===")
    monthly_pieces = []
    for name, df in weekly.items():
        df["year_month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("year_month")[name].mean().reset_index()
        monthly_pieces.append(monthly)

    monthly_combined = monthly_pieces[0]
    for piece in monthly_pieces[1:]:
        monthly_combined = monthly_combined.merge(piece, on="year_month", how="outer")

    monthly_combined = monthly_combined.sort_values("year_month").reset_index(drop=True)

    # === Avledede variabler ===
    print("\n=== Beregner avledede variabler ===")
    # Lagrene som forskjell fra 5-års rullende snitt (relativ tightness)
    for col in ["us_crude_stocks_kbbl", "cushing_stocks_kbbl"]:
        if col in monthly_combined.columns:
            rolling = monthly_combined[col].rolling(60, min_periods=24).mean()
            monthly_combined[f"{col}_dev_5y"] = monthly_combined[col] - rolling
            monthly_combined[f"{col}_dev_5y_pct"] = (
                (monthly_combined[col] - rolling) / rolling * 100
            )

    # Refinery utilization endring 3 måneder
    if "us_refinery_util_pct" in monthly_combined.columns:
        monthly_combined["refinery_util_3m_change"] = (
            monthly_combined["us_refinery_util_pct"]
            - monthly_combined["us_refinery_util_pct"].shift(3)
        )
        # Boolean: høy utilization-regime
        monthly_combined["d_refinery_tight"] = (
            monthly_combined["us_refinery_util_pct"] > 92
        ).astype(int)
        monthly_combined["d_refinery_slack"] = (
            monthly_combined["us_refinery_util_pct"] < 85
        ).astype(int)

    # Net imports = imports - exports
    if all(c in monthly_combined.columns for c in ["us_crude_imports_kbpd", "us_crude_exports_kbpd"]):
        monthly_combined["us_net_crude_imports_kbpd"] = (
            monthly_combined["us_crude_imports_kbpd"] - monthly_combined["us_crude_exports_kbpd"]
        )

    out_csv = PROCESSED_DIR / "eia_fundamentals_monthly.csv"
    monthly_combined.to_csv(out_csv, index=False)

    print(f"\n=== Lagret: {out_csv} ===")
    print(f"Periode: {monthly_combined['year_month'].min()} – {monthly_combined['year_month'].max()}")
    print(f"Kolonner: {len(monthly_combined.columns)}")
    print(f"Rader: {len(monthly_combined)}")

    # Statistikk
    print("\n=== Statistikk ===")
    for col in monthly_combined.columns:
        if col == "year_month":
            continue
        n = monthly_combined[col].notna().sum()
        if n > 50:
            mean = monthly_combined[col].mean()
            std = monthly_combined[col].std()
            print(f"  {col:45s}: N={n:4d}, mean={mean:>10.1f}, std={std:>9.1f}")


if __name__ == "__main__":
    main()
