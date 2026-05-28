"""
Hent raffinerimarginer (crack spreads) og legg til hendelsesdummies.

Crack spreads er forskjellen mellom produktpris og råoljepris:
  - Gasoline crack = gasoline-pris × 42 - crude pris
  - Diesel crack = diesel-pris × 42 - crude pris
  - 3-2-1 = (2×gasoline + 1×diesel) × 42 / 3 - crude

Disse er kritiske fordi:
  - Når dieselmarginen er høy verdsettes middeldestillat-rike crudes mer
  - Når bensinmarginen er høy verdsettes lette/nafta-rike crudes mer
  - Dette skifter rabatt/premium for ulike kvaliteter over tid

Hendelsesdummies:
  - Russland-sanksjoner (feb 2022+)
  - US shale boom (2011-2015)
  - OPEC+ cuts (2017+, 2020+, 2023+)
  - COVID demand crash (mar-mai 2020)
"""

from pathlib import Path
from io import StringIO
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "eia_products"

EIA_PRODUCT_URLS = {
    "gasoline_nyharbor_conv": "https://www.eia.gov/dnav/pet/hist_xls/EER_EPMRU_PF4_Y35NY_DPGm.xls",
    "gasoline_usgulf_conv": "https://www.eia.gov/dnav/pet/hist_xls/EER_EPMRU_PF4_Y44MB_DPGm.xls",
    "gasoline_la_conv": "https://www.eia.gov/dnav/pet/hist_xls/EER_EPMRU_PF4_Y05LA_DPGm.xls",
    "ulsd_nyharbor": "https://www.eia.gov/dnav/pet/hist_xls/EER_EPD2DXL0_PF4_Y35NY_DPGm.xls",
    "jet_gulfcoast": "https://www.eia.gov/dnav/pet/hist_xls/EER_EPJK_PF4_RGC_DPGm.xls",
    "heating_oil_nyharbor": "https://www.eia.gov/dnav/pet/hist_xls/EER_EPD2F_PF4_Y35NY_DPGm.xls",
}


def download_eia(name: str, url: str) -> Path | None:
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


def parse_eia_product_xls(path: Path) -> pd.DataFrame:
    """Parse EIA spotpris-XLS. Returnerer (date, price_usd_per_gallon)."""
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
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna()
    return df


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Henter produktpriser fra EIA ===\n")
    product_prices = {}
    for name, url in EIA_PRODUCT_URLS.items():
        path = download_eia(name, url)
        if path:
            df = parse_eia_product_xls(path)
            if not df.empty:
                product_prices[name] = df
                print(f"  {name}: {len(df)} obs ({df['date'].min():%Y-%m} – {df['date'].max():%Y-%m})")

    if not product_prices:
        print("  Ingen produktpriser hentet — bruker fallback")

    # === Les eksisterende kontroller ===
    controls = pd.read_csv(PROCESSED_DIR / "market_controls_monthly.csv")
    controls["year_month"] = pd.PeriodIndex(controls["year_month_str"], freq="M")

    # === Beregn crack spreads ===
    print("\n=== Beregner crack spreads ===")
    brent_col = controls.set_index("year_month")["brent_price"]
    wti_col = controls.set_index("year_month")["wti_price"]

    for name, df in product_prices.items():
        df["year_month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("year_month")["price"].mean()
        controls = controls.merge(
            monthly.rename(f"{name}_usd_gal").reset_index(),
            on="year_month",
            how="left",
        )

    # Konverter gallon-priser til crack spreads ($/bbl)
    # 1 fat = 42 US gallons.
    # Velg første tilgjengelige gasoline-serie (Gulf Coast prio, så NY Harbor, så LA).
    gasoline_col = None
    for cand in ["gasoline_usgulf_conv_usd_gal", "gasoline_nyharbor_conv_usd_gal", "gasoline_la_conv_usd_gal"]:
        if cand in controls.columns and controls[cand].notna().sum() > 100:
            gasoline_col = cand
            print(f"  Bruker {cand} for gasoline crack")
            break

    if gasoline_col:
        controls["gasoline_crack_brent"] = controls[gasoline_col] * 42 - controls["brent_price"]
        controls["gasoline_crack_wti"] = controls[gasoline_col] * 42 - controls["wti_price"]

    # Velg diesel-serie: ULSD prioritert, heating oil som fallback (lengre historikk)
    diesel_col = None
    for cand in ["ulsd_nyharbor_usd_gal", "heating_oil_nyharbor_usd_gal"]:
        if cand in controls.columns and controls[cand].notna().sum() > 100:
            diesel_col = cand
            print(f"  Bruker {cand} for diesel crack")
            break

    if diesel_col:
        controls["diesel_crack_brent"] = controls[diesel_col] * 42 - controls["brent_price"]
        controls["diesel_crack_wti"] = controls[diesel_col] * 42 - controls["wti_price"]

    if "jet_gulfcoast_usd_gal" in controls.columns:
        controls["jet_crack_brent"] = (
            controls["jet_gulfcoast_usd_gal"] * 42 - controls["brent_price"]
        )

    # 3-2-1 Brent crack: (2 gasoline + 1 diesel) / 3
    if gasoline_col and diesel_col:
        controls["crack_321_brent"] = (
            (2 * controls[gasoline_col] + controls[diesel_col]) / 3 * 42
            - controls["brent_price"]
        )
        controls["crack_321_wti"] = (
            (2 * controls[gasoline_col] + controls[diesel_col]) / 3 * 42
            - controls["wti_price"]
        )

    # Middle-distillate strength: høyt diesel-crack vs gasoline-crack betyr
    # at middeldestillat-tunge crudes (Heidrun, Mars, Arab Heavy) verdsettes mer
    if "diesel_crack_brent" in controls.columns and "gasoline_crack_brent" in controls.columns:
        controls["diesel_minus_gasoline_crack"] = (
            controls["diesel_crack_brent"] - controls["gasoline_crack_brent"]
        )

    # Light-Heavy spread proxy: forskjellen mellom Brent og Dubai sier noe om
    # hvor mye lett-sweet vs medium-sour verdsettes
    if "brent_price" in controls.columns:
        prices = pd.read_csv(PROCESSED_DIR / "global_crude_prices_monthly.csv")
        dubai = prices[prices["grade"] == "Dubai Fateh"][["date", "price"]].copy()
        dubai["date"] = pd.to_datetime(dubai["date"])
        dubai["year_month"] = dubai["date"].dt.to_period("M")
        dubai_monthly = dubai.groupby("year_month")["price"].mean()
        controls = controls.merge(
            dubai_monthly.rename("dubai_price").reset_index(),
            on="year_month", how="left",
        )
        controls["brent_dubai_spread"] = controls["brent_price"] - controls["dubai_price"]

    # === Hendelsesdummies ===
    print("\n=== Legger til hendelsesdummies ===")
    controls["year_month_period"] = controls["year_month"]
    controls["d_russia_sanctions"] = (controls["year_month"] >= pd.Period("2022-02")).astype(int)
    controls["d_iran_sanctions_v1"] = (
        (controls["year_month"] >= pd.Period("2012-01")) &
        (controls["year_month"] <= pd.Period("2015-12"))
    ).astype(int)
    controls["d_iran_sanctions_v2"] = (controls["year_month"] >= pd.Period("2018-05")).astype(int)
    controls["d_venezuela_sanctions"] = (controls["year_month"] >= pd.Period("2019-01")).astype(int)
    controls["d_us_shale_boom"] = (
        (controls["year_month"] >= pd.Period("2011-01")) &
        (controls["year_month"] <= pd.Period("2015-06"))
    ).astype(int)
    controls["d_opec_cuts_2017"] = (
        (controls["year_month"] >= pd.Period("2017-01")) &
        (controls["year_month"] <= pd.Period("2018-12"))
    ).astype(int)
    controls["d_covid"] = (
        (controls["year_month"] >= pd.Period("2020-03")) &
        (controls["year_month"] <= pd.Period("2020-08"))
    ).astype(int)
    controls["d_opec_plus_cuts_2023"] = (controls["year_month"] >= pd.Period("2023-04")).astype(int)

    # === Lagre ===
    controls = controls.drop(columns=["year_month_period"], errors="ignore")
    out_csv = PROCESSED_DIR / "market_controls_monthly.csv"
    controls.to_csv(out_csv, index=False)

    print(f"\n=== Markedskontroller oppdatert: {out_csv} ===")
    print(f"Nye kolonner:")
    new_cols = [c for c in controls.columns if any(x in c for x in
                ["crack", "_usd_gal", "d_russia", "d_iran", "d_venezuela",
                 "d_us_shale", "d_opec", "d_covid", "diesel_minus"])]
    for c in new_cols:
        n_valid = controls[c].notna().sum()
        if controls[c].dtype in [np.float64, np.int64]:
            print(f"  {c:35s}: {n_valid:4d} obs, mean={controls[c].mean():.2f}")
        else:
            print(f"  {c:35s}: {n_valid:4d} obs")

    print(f"\nTotal kolonner: {len(controls.columns)}")
    print(f"Periode: {controls['year_month'].min()} – {controls['year_month'].max()}")


if __name__ == "__main__":
    main()
