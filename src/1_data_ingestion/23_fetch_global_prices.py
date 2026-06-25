"""
Hent månedlige råoljepriser fra flere offentlige kilder og beregn
differensialer mot Brent for alle tilgjengelige crude grades.

Datakilder:
  1. FRED (Federal Reserve) - Brent, WTI, Dubai (IMF-serier)
  2. EIA - Brent, WTI, Mars, LLS m.fl. (direkte XLS-nedlasting)
  3. Norske normpris-differensialer (allerede hentet, leses inn)
  4. World Bank Commodity Prices - Brent, Dubai, WTI

Output:
  - data/processed/global_crude_prices_monthly.csv  (priser)
  - data/processed/global_differentials_monthly.csv  (differensialer mot Brent)
"""

from pathlib import Path
from io import StringIO
import time
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = RAW_DIR / "price_cache"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES = {
    "MCOILBRENTEU": "Brent Blend",
    "MCOILWTICO": "WTI",
    "POILDUBUSDM": "Dubai Fateh",
}

EIA_XLS_SERIES = {
    "RBRTEm": "Brent Blend",
    "RWTCm": "WTI",
}

EIA_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/{series}.xls"


def fetch_fred_series(series_id: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Hent en månedsserie fra FRED som CSV (ingen API-nøkkel nødvendig)."""
    url = f"{FRED_CSV_URL}?id={series_id}&cosd={start}&coed=2026-12-31"
    print(f"  FRED: henter {series_id} ...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        return df
    except Exception as e:
        print(f"    FRED feilet for {series_id}: {e}")
        return pd.DataFrame()


def fetch_eia_xls(series_key: str) -> pd.DataFrame:
    """Hent månedlige spotpriser fra EIA direkte XLS-nedlasting."""
    url = EIA_XLS_URL.format(series=series_key)
    cache_file = CACHE_DIR / f"eia_{series_key}.xls"
    print(f"  EIA: henter {series_key} ...")

    if not cache_file.exists():
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(resp.content)
        except Exception as e:
            print(f"    EIA nedlasting feilet: {e}")
            return pd.DataFrame()

    try:
        df = pd.read_excel(cache_file, sheet_name=1, header=2, engine="xlrd")
        date_col = [c for c in df.columns if "date" in c.lower() or "Date" in str(c)]
        if not date_col:
            date_col = [df.columns[0]]
        val_col = [c for c in df.columns if c not in date_col]
        if not val_col:
            return pd.DataFrame()
        df = df[[date_col[0], val_col[0]]].copy()
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna()
        return df
    except Exception as e:
        print(f"    EIA parsing feilet: {e}")
        return pd.DataFrame()


def fetch_world_bank_commodities() -> dict[str, pd.DataFrame]:
    """Hent World Bank Commodity Price Data (månedlig).
    Returnerer dict med grade-navn -> DataFrame(date, price)."""
    url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
    cache_file = CACHE_DIR / "worldbank_cmo_monthly.xlsx"

    if not cache_file.exists():
        print("  World Bank: laster ned CMO-data ...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(resp.content)
        except Exception as e:
            print(f"    World Bank nedlasting feilet: {e}")
            return {}

    results = {}
    try:
        xls = pd.ExcelFile(cache_file, engine="openpyxl")
        for sheet in xls.sheet_names:
            if "monthly" in sheet.lower() or "prices" in sheet.lower():
                df = pd.read_excel(xls, sheet_name=sheet)
                oil_cols = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if "brent" in col_lower:
                        oil_cols[col] = "Brent Blend"
                    elif "dubai" in col_lower:
                        oil_cols[col] = "Dubai Fateh"
                    elif "wti" in col_lower:
                        oil_cols[col] = "WTI"

                if oil_cols:
                    date_col = df.columns[0]
                    for col, grade in oil_cols.items():
                        sub = df[[date_col, col]].copy()
                        sub.columns = ["date", "price"]
                        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                        sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
                        sub = sub.dropna()
                        if not sub.empty and grade not in results:
                            results[grade] = sub
                    break
    except Exception as e:
        print(f"    World Bank parsing feilet: {e}")

    return results


def load_norwegian_differentials() -> pd.DataFrame:
    """Les inn allerede hentede norske normpris-differensialer."""
    csv_path = PROCESSED_DIR / "normpris_differentials_long.csv"
    if not csv_path.exists():
        print("  ADVARSEL: normpris_differentials_long.csv finnes ikke!")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.rename(columns={"field": "grade", "differential_usd": "differential"})
    df["source"] = "normpris"
    return df[["grade", "date", "differential", "source"]]


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_prices: list[pd.DataFrame] = []

    # === 1. FRED-serier ===
    print("\n=== FRED (Federal Reserve) ===")
    for series_id, grade_name in FRED_SERIES.items():
        df = fetch_fred_series(series_id)
        if not df.empty:
            df["grade"] = grade_name
            df["source"] = "FRED"
            all_prices.append(df)
            print(f"    {grade_name}: {len(df)} månedlige obs ({df['date'].min():%Y-%m} – {df['date'].max():%Y-%m})")
        time.sleep(0.5)

    # === 2. EIA-serier ===
    print("\n=== EIA (US Energy Information Administration) ===")
    for series_key, grade_name in EIA_XLS_SERIES.items():
        df = fetch_eia_xls(series_key)
        if not df.empty:
            df["grade"] = grade_name
            df["source"] = "EIA"
            all_prices.append(df)
            print(f"    {grade_name}: {len(df)} månedlige obs ({df['date'].min():%Y-%m} – {df['date'].max():%Y-%m})")
        time.sleep(0.5)

    # === 3. World Bank ===
    print("\n=== World Bank Commodity Prices ===")
    wb_data = fetch_world_bank_commodities()
    for grade_name, df in wb_data.items():
        df["grade"] = grade_name
        df["source"] = "WorldBank"
        all_prices.append(df)
        print(f"    {grade_name}: {len(df)} månedlige obs ({df['date'].min():%Y-%m} – {df['date'].max():%Y-%m})")

    if not all_prices:
        print("\nIngen prisdata hentet — avbryter.")
        return

    # === Kombiner alle priser ===
    prices = pd.concat(all_prices, ignore_index=True)

    # Prioriter kilder: FRED > EIA > WorldBank (ved duplikater)
    source_priority = {"FRED": 0, "EIA": 1, "WorldBank": 2}
    prices["_prio"] = prices["source"].map(source_priority)
    prices["year_month"] = prices["date"].dt.to_period("M")
    prices = prices.sort_values(["grade", "year_month", "_prio"])
    prices = prices.drop_duplicates(subset=["grade", "year_month"], keep="first")
    prices = prices.drop(columns=["_prio", "year_month"])

    out_prices = PROCESSED_DIR / "global_crude_prices_monthly.csv"
    prices.to_csv(out_prices, index=False)
    print(f"\n=== Priser lagret: {out_prices} ===")
    print(f"Totalt: {len(prices):,} observasjoner, {prices['grade'].nunique()} grades")
    print(f"Periode: {prices['date'].min():%Y-%m} – {prices['date'].max():%Y-%m}")

    # === Beregn differensialer mot Brent ===
    brent = prices[prices["grade"] == "Brent Blend"][["date", "price"]].copy()
    brent = brent.rename(columns={"price": "brent_price"})
    brent["year_month"] = brent["date"].dt.to_period("M")
    brent = brent.drop_duplicates(subset=["year_month"], keep="first")

    non_brent = prices[prices["grade"] != "Brent Blend"].copy()
    non_brent["year_month"] = non_brent["date"].dt.to_period("M")

    merged = non_brent.merge(brent[["year_month", "brent_price"]], on="year_month", how="inner")
    merged["differential"] = merged["price"] - merged["brent_price"]

    intl_diffs = merged[["grade", "date", "differential", "source"]].copy()

    # === Legg til norske differensialer ===
    nor_diffs = load_norwegian_differentials()
    all_diffs = pd.concat([intl_diffs, nor_diffs], ignore_index=True)
    all_diffs = all_diffs.sort_values(["grade", "date"]).reset_index(drop=True)

    out_diffs = PROCESSED_DIR / "global_differentials_monthly.csv"
    all_diffs.to_csv(out_diffs, index=False)

    print(f"\n=== Differensialer lagret: {out_diffs} ===")
    print(f"Totalt: {len(all_diffs):,} observasjoner, {all_diffs['grade'].nunique()} grades")
    print(f"Periode: {all_diffs['date'].min()} – {all_diffs['date'].max()}")

    print("\n=== Differensialer per grade ===")
    summary = all_diffs.groupby("grade").agg(
        n_obs=("differential", "size"),
        mean_diff=("differential", "mean"),
        std_diff=("differential", "std"),
        min_date=("date", "min"),
        max_date=("date", "max"),
    ).sort_values("n_obs", ascending=False)
    print(summary.to_string(float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
