"""
Forward curve struktur (contango/backwardation) + logistikk-features.

Forward curve:
  Bruker WTI-futures M1, M3, M6, M12 fra EIA. Når M1 < M12 = contango
  (lagerinsentiv, ofte tegn på overkapasitet). Når M1 > M12 = backwardation
  (knapphet, fysiske premier).

Logistikk:
  - Distance-to-market band per grade (statisk klassifisering)
  - Landlocked-flag (WCS, Cushing-grades) — kan ikke eksporteres med tanker
  - Geografisk discount-flag (Bakken-railroad, Permian-pipeline-konstrains)

Sesongeffekter:
  - sin/cos for måned (kontinuerlige) i stedet for 12 dummies
  - Vinter-flag (Q1+Q4) når diesel-etterspørsel er høyest
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "wti_futures"

# WTI futures contracts ved måned-end (Cushing delivery)
EIA_WTI_FUTURES = {
    "wti_m1":  "https://www.eia.gov/dnav/pet/hist_xls/RCLC1m.xls",
    "wti_m2":  "https://www.eia.gov/dnav/pet/hist_xls/RCLC2m.xls",
    "wti_m3":  "https://www.eia.gov/dnav/pet/hist_xls/RCLC3m.xls",
    "wti_m4":  "https://www.eia.gov/dnav/pet/hist_xls/RCLC4m.xls",
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
        print(f"    Feilet: {e}")
        return None


def parse_monthly_xls(path: Path, col: str) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(path, engine="xlrd")
    except Exception:
        return pd.DataFrame()
    data_sheets = [s for s in xls.sheet_names if "data" in s.lower()]
    if not data_sheets:
        return pd.DataFrame()
    df = pd.read_excel(xls, sheet_name=data_sheets[0], header=2)
    if df.shape[1] < 2:
        return pd.DataFrame()
    df = df.iloc[:, :2].copy()
    df.columns = ["date", col]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # === Forward curve ===
    print("=== Henter WTI futures (forward curve) ===\n")
    futures_pieces = []
    for name, url in EIA_WTI_FUTURES.items():
        path = download(name, url)
        if path:
            df = parse_monthly_xls(path, name)
            if not df.empty:
                df["year_month"] = df["date"].dt.to_period("M")
                monthly = df.groupby("year_month")[name].mean().reset_index()
                futures_pieces.append(monthly)
                print(f"  {name}: {len(monthly):4d} obs ({monthly['year_month'].min()} – {monthly['year_month'].max()})")

    if futures_pieces:
        futures = futures_pieces[0]
        for piece in futures_pieces[1:]:
            futures = futures.merge(piece, on="year_month", how="outer")

        # Forward curve slope: (M4 - M1) i $/fat. Positiv = contango.
        if "wti_m1" in futures.columns and "wti_m4" in futures.columns:
            futures["fc_slope_4m"] = futures["wti_m4"] - futures["wti_m1"]
            # Normaliser til % av M1-pris (skala-uavhengig)
            futures["fc_slope_pct"] = futures["fc_slope_4m"] / futures["wti_m1"] * 100
            futures["d_contango"] = (futures["fc_slope_4m"] > 0).astype(int)
            futures["d_strong_contango"] = (futures["fc_slope_4m"] > 1.5).astype(int)
            futures["d_strong_backwardation"] = (futures["fc_slope_4m"] < -1.5).astype(int)

        out_fc = PROCESSED_DIR / "forward_curve_monthly.csv"
        futures.to_csv(out_fc, index=False)
        print(f"\nForward curve lagret: {out_fc}")
        print(f"Statistikk fc_slope_4m: mean={futures['fc_slope_4m'].mean():.2f}, "
              f"std={futures['fc_slope_4m'].std():.2f}")
        print(f"Contango-andel: {futures['d_contango'].mean():.1%}")

    # === Distance-to-market klassifisering ===
    print("\n=== Bygger distance-to-market classification ===")
    # Per grade: nærmeste store raffineri-marked og distanse-band
    # Korte = innenfor 1-2 dager seiling. Lange = trans-Atlantic / trans-Pacific.
    # Landlocked = kan ikke skipes ut.
    distance_classification = pd.DataFrame([
        # Nordsjø/Norge → Europa: kort
        ("Brent Blend", "NorthSea-Europe", "short", 0, 0),
        ("Forties", "NorthSea-Europe", "short", 0, 0),
        ("Oseberg", "NorthSea-Europe", "short", 0, 0),
        ("Ekofisk", "NorthSea-Europe", "short", 0, 0),
        ("Troll", "NorthSea-Europe", "short", 0, 0),
        ("Johan Sverdrup", "NorthSea-Europe", "short", 0, 0),
        ("Statfjord", "NorthSea-Europe", "short", 0, 0),
        ("Gullfaks", "NorthSea-Europe", "short", 0, 0),
        ("Alvheim", "NorthSea-Europe", "short", 0, 0),
        ("Grane", "NorthSea-Europe", "short", 0, 0),
        ("Heidrun", "NorthSea-Europe", "short", 0, 0),
        ("Norne", "NorthSea-Europe", "short", 0, 0),
        ("Draugen", "NorthSea-Europe", "short", 0, 0),
        ("Skarv", "NorthSea-Europe", "short", 0, 0),
        ("Asgard", "NorthSea-Europe", "short", 0, 0),
        ("Balder", "NorthSea-Europe", "short", 0, 0),
        ("Gina Krog", "NorthSea-Europe", "short", 0, 0),
        ("Goliat", "NorthSea-Europe", "short", 0, 0),
        ("Gudrun", "NorthSea-Europe", "short", 0, 0),
        ("Martin Linge", "NorthSea-Europe", "short", 0, 0),
        ("Njord", "NorthSea-Europe", "short", 0, 0),
        ("Knarr", "NorthSea-Europe", "short", 0, 0),
        ("Yme", "NorthSea-Europe", "short", 0, 0),
        ("Volve", "NorthSea-Europe", "short", 0, 0),
        ("Jotun", "NorthSea-Europe", "short", 0, 0),
        ("Valhall", "NorthSea-Europe", "short", 0, 0),
        ("Ula", "NorthSea-Europe", "short", 0, 0),
        # North America: WTI/Bakken = landlocked, Maya/Mars = USGC seafloating
        ("WTI", "Cushing-USGC", "short", 1, 1),  # LANDLOCKED til 2015
        ("Mars Blend", "USGC-Local", "short", 0, 0),
        ("LLS", "USGC-Local", "short", 0, 0),
        ("WCS", "AB-USGC", "long", 1, 1),  # LANDLOCKED, pipeline-konstrained
        ("Bow River Heavy", "AB-USGC", "long", 1, 1),
        ("Lloydminster", "AB-USGC", "long", 1, 1),
        ("Cold Lake", "AB-USGC", "long", 1, 1),
        ("Canadian Light Sour", "AB-USGC", "long", 1, 1),
        ("Syncrude Sweet", "AB-USGC", "long", 1, 1),
        # Latin Amerika: kort til USGC
        ("Maya", "Mex-USGC", "short", 0, 0),
        ("Isthmus", "Mex-USGC", "short", 0, 0),
        ("Olmeca", "Mex-USGC", "short", 0, 0),
        ("Oriente", "Ecu-USGC", "medium", 0, 0),
        ("Napo", "Ecu-USGC", "medium", 0, 0),
        ("Vasconia", "Col-USGC", "short", 0, 0),
        ("Merey", "Ven-USGC", "short", 0, 0),
        ("Furrial", "Ven-USGC", "short", 0, 0),
        ("Leona", "Ven-USGC", "short", 0, 0),
        ("Marlim", "Brazil-Asia", "long", 0, 0),
        # Midtøsten: lang seiling til Asia/Europa
        ("Arab Light", "MidEast-Asia", "long", 0, 0),
        ("Arab Heavy", "MidEast-Asia", "long", 0, 0),
        ("Arab Medium", "MidEast-Asia", "long", 0, 0),
        ("Arab Extra Light", "MidEast-Asia", "long", 0, 0),
        ("Dubai Fateh", "MidEast-Asia", "long", 0, 0),
        ("Murban", "MidEast-Asia", "long", 0, 0),
        ("Oman", "MidEast-Asia", "long", 0, 0),
        ("Iran Light", "MidEast-Asia", "long", 0, 0),
        ("Iran Heavy", "MidEast-Asia", "long", 0, 0),
        ("Kuwait Export", "MidEast-Asia", "long", 0, 0),
        ("Basrah Light", "MidEast-Asia", "long", 0, 0),
        ("Basrah Heavy", "MidEast-Asia", "long", 0, 0),
        # Vest-Afrika: middels seiling
        ("Bonny Light", "WAF-Europe", "medium", 0, 0),
        ("Qua Iboe", "WAF-Europe", "medium", 0, 0),
        ("Forcados", "WAF-Europe", "medium", 0, 0),
        ("Brass River", "WAF-Europe", "medium", 0, 0),
        ("Girassol", "WAF-Asia", "long", 0, 0),
        ("Cabinda", "WAF-Asia", "long", 0, 0),
        ("Dalia", "WAF-Asia", "long", 0, 0),
        ("Rabi Light", "WAF-Europe", "medium", 0, 0),
        ("Djeno", "WAF-Asia", "long", 0, 0),
        # Nord-Afrika: kort til Europa
        ("Saharan Blend", "NAF-Europe", "short", 0, 0),
        ("Es Sider", "NAF-Europe", "short", 0, 0),
        # FSU: pipeline-bundet (Urals = pipeline+tanker, ESPO = ren tanker)
        ("Urals", "Baltic-Med", "medium", 0, 1),  # PIPELINE-AVHENGIG
        ("ESPO", "RuFE-Asia", "short", 0, 0),
        ("CPC Blend", "Caspian-Med", "medium", 0, 1),  # PIPELINE-AVHENGIG
        ("Azeri Light", "Caspian-Med", "medium", 0, 1),
        ("Tengiz", "Caspian-Med", "medium", 0, 1),
        # Asia-Pacific: lokal
        ("Tapis", "Asia-Local", "short", 0, 0),
        ("Minas", "Asia-Local", "short", 0, 0),
        ("Daqing", "Asia-Local", "short", 0, 0),
        ("Cossack", "Asia-Local", "short", 0, 0),
        ("Bach Ho", "Asia-Local", "short", 0, 0),
        ("Seria Light", "Asia-Local", "short", 0, 0),
    ], columns=["grade", "primary_route", "distance_band", "is_landlocked", "is_pipeline_constrained"])

    out_dist = RAW_DIR / "grade_logistics.csv"
    distance_classification.to_csv(out_dist, index=False)
    print(f"Distance-classification lagret: {out_dist}")
    print(f"  Short: {(distance_classification['distance_band']=='short').sum()} grades")
    print(f"  Medium: {(distance_classification['distance_band']=='medium').sum()} grades")
    print(f"  Long: {(distance_classification['distance_band']=='long').sum()} grades")
    print(f"  Landlocked: {distance_classification['is_landlocked'].sum()} grades")
    print(f"  Pipeline-constrained: {distance_classification['is_pipeline_constrained'].sum()} grades")


if __name__ == "__main__":
    main()
