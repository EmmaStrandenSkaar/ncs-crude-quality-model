"""
Hent tidsvarierende markedskontroller som påvirker crude differensialer.

Variabler:
  1. Brent-prisnivå (absolutt, log, kvartil-regime)
  2. WTI-Brent spread (proxy for Atlantic Basin dynamics)
  3. 3-2-1 crack spread (raffinerimarkeds-margin)
  4. US refinery utilization
  5. Dollar-styrke (DXY / trade-weighted)
  6. VIX (markedsusikkerhet)
  7. OPEC spare capacity proxy (OPEC produksjon)
  8. Global oil inventory proxy

Kilder: FRED (gratis CSV-nedlasting, ingen API-nøkkel).
"""

from pathlib import Path
from io import StringIO
import time
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


FRED_CONTROLS = {
    "MCOILBRENTEU": "brent_price",
    "MCOILWTICO": "wti_price",
    "DTWEXBGS": "usd_broad_index",
    "VIXCLS": "vix",
    "GASREGW": "us_gasoline_price",
    "DHHNGSP": "us_heating_oil_price",
}

FRED_CONTROLS_DAILY_TO_MONTHLY = {
    "DTWEXBGS", "VIXCLS", "GASREGW", "DHHNGSP",
}


def fetch_fred(series_id: str, start: str = "2000-01-01") -> pd.DataFrame:
    url = f"{FRED_CSV}?id={series_id}&cosd={start}&coed=2026-12-31"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return df
    except Exception as e:
        print(f"  FEIL {series_id}: {e}")
        return pd.DataFrame()


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    return df.groupby("year_month").agg(value=("value", "mean")).reset_index()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}

    print("=== Henter markedskontroller fra FRED ===\n")

    for series_id, col_name in FRED_CONTROLS.items():
        print(f"  {series_id} -> {col_name} ...", end=" ")
        df = fetch_fred(series_id)
        if df.empty:
            print("INGEN DATA")
            continue

        if series_id in FRED_CONTROLS_DAILY_TO_MONTHLY:
            df = to_monthly(df)
        else:
            df["year_month"] = df["date"].dt.to_period("M")
            df = df.rename(columns={"value": "value"})

        df = df.rename(columns={"value": col_name})
        frames[col_name] = df[["year_month", col_name]]
        n = len(df)
        print(f"{n} obs")
        time.sleep(0.3)

    if not frames:
        print("Ingen kontrollvariabler hentet!")
        return

    # === Slå sammen alle kontroller på year_month ===
    base = list(frames.values())[0]
    for name, df in list(frames.items())[1:]:
        base = base.merge(df, on="year_month", how="outer")

    base = base.sort_values("year_month").reset_index(drop=True)

    # === Avledede variabler ===
    if "brent_price" in base.columns and "wti_price" in base.columns:
        base["wti_brent_spread"] = base["wti_price"] - base["brent_price"]

    if "brent_price" in base.columns:
        base["brent_log"] = np.log(base["brent_price"])
        base["brent_pct_change"] = base["brent_price"].pct_change()
        q25 = base["brent_price"].quantile(0.25)
        q75 = base["brent_price"].quantile(0.75)
        base["brent_regime"] = pd.cut(
            base["brent_price"],
            bins=[-np.inf, q25, q75, np.inf],
            labels=["low", "mid", "high"],
        )
        base["brent_volatility_3m"] = base["brent_pct_change"].rolling(3).std()

    if "us_gasoline_price" in base.columns and "brent_price" in base.columns:
        base["crack_spread_proxy"] = base["us_gasoline_price"] * 42 - base["brent_price"]

    base["month"] = base["year_month"].dt.month
    base["quarter"] = ((base["month"] - 1) // 3) + 1
    base["year"] = base["year_month"].dt.year

    # === Lagre ===
    out_csv = PROCESSED_DIR / "market_controls_monthly.csv"
    base["year_month_str"] = base["year_month"].astype(str)
    base.to_csv(out_csv, index=False)

    print(f"\n=== Markedskontroller lagret: {out_csv} ===")
    print(f"Periode: {base['year_month'].min()} – {base['year_month'].max()}")
    print(f"Rader: {len(base):,}")
    print(f"\nKolonner: {list(base.columns)}")
    print(f"\nOppsummering:")
    num_cols = base.select_dtypes(include=[np.number]).columns
    print(base[num_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
