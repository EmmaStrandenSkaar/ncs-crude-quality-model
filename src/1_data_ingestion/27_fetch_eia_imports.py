"""
Hent EIA-data for US crude oil imports: landingskost per crude grade.

EIA publiserer månedlige landed costs for ~25 spesifikke crude grades (Brent,
Bonny Light, Maya, Arab Light, etc.) helt tilbake til 1980-tallet.

Vi parser grade-navn fra kolonneoverskrifter, beregner differensialer mot Brent,
og kobler til kvalitetsdata fra vår crude quality database.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "eia_imports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EIA_URL = "https://www.eia.gov/dnav/pet/xls/PET_MOVE_LAND2_K_M.xls"

GRADE_MAP = {
    "Algerian Saharan Blend": {"grade": "Saharan Blend", "api": 45.5, "sulfur": 0.09},
    "Angolan Cabinda": {"grade": "Cabinda", "api": 32.0, "sulfur": 0.17},
    "Brazilian Marlim": {"grade": "Marlim", "api": 19.2, "sulfur": 0.78},
    "Canadian Bow River Heavy": {"grade": "Bow River Heavy", "api": 22.0, "sulfur": 2.80},
    "Canadian Light Sour Blend": {"grade": "Canadian Light Sour", "api": 35.0, "sulfur": 1.20},
    "Canadian LLoydminster": {"grade": "Lloydminster", "api": 20.0, "sulfur": 3.50},
    "Ecuadorian Napo": {"grade": "Napo", "api": 19.5, "sulfur": 2.03},
    "Ecuadorian Oriente": {"grade": "Oriente", "api": 24.0, "sulfur": 1.42},
    "Gabon Rabi-Kouanga": {"grade": "Rabi Light", "api": 33.5, "sulfur": 0.07},
    "Iraqi Basrah Light": {"grade": "Basrah Light", "api": 30.5, "sulfur": 2.90},
    "Mexican Mayan": {"grade": "Maya", "api": 21.1, "sulfur": 3.30},
    "Mexican Olmeca": {"grade": "Olmeca", "api": 38.0, "sulfur": 0.80},
    "Nigerian Bonny Light": {"grade": "Bonny Light", "api": 33.4, "sulfur": 0.13},
    "Nigerian Forcados Blend": {"grade": "Forcados", "api": 31.0, "sulfur": 0.18},
    "Nigerian Qua Iboe": {"grade": "Qua Iboe", "api": 36.1, "sulfur": 0.10},
    "Saudi Arabian Berri": {"grade": "Arab Extra Light", "api": 37.2, "sulfur": 1.15},
    "Saudi Arabian Light": {"grade": "Arab Light", "api": 33.0, "sulfur": 1.77},
    "Saudi Arabian Medium": {"grade": "Arab Medium", "api": 30.4, "sulfur": 2.54},
    "United Kingdom Brent": {"grade": "Brent Blend (landed)", "api": 38.3, "sulfur": 0.37},
    "Venezuelan Furrial": {"grade": "Furrial", "api": 30.0, "sulfur": 0.83},
    "Venezuelan Leona": {"grade": "Leona", "api": 24.0, "sulfur": 1.50},
    "Venezuelan Merey": {"grade": "Merey", "api": 16.0, "sulfur": 2.45},
}


def download_if_missing(url: str, dest: Path) -> Path | None:
    if dest.exists():
        print(f"  Bruker cachet: {dest.name}")
        return dest
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        print(f"  Nedlasting feilet: {e}")
        return None


def parse_grade_from_column(col: str) -> str | None:
    m = re.search(r"Landed Costs of (.+?) Crude Oil", col)
    return m.group(1).strip() if m else None


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== EIA Crude Oil Landed Costs ===\n")
    path = download_if_missing(EIA_URL, CACHE_DIR / "landed_costs_by_grade.xls")
    if not path:
        return

    xls = pd.ExcelFile(path, engine="xlrd")
    data_sheet = [s for s in xls.sheet_names if "data" in s.lower()]
    if not data_sheet:
        print("Ingen data-ark funnet")
        return

    df = pd.read_excel(xls, sheet_name=data_sheet[0], header=2)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Les Brent-priser for differensialberegning
    prices = pd.read_csv(PROCESSED_DIR / "global_crude_prices_monthly.csv")
    brent = prices[prices["grade"] == "Brent Blend"][["date", "price"]].copy()
    brent["date"] = pd.to_datetime(brent["date"])
    brent = brent.rename(columns={"price": "brent_price"})
    brent["year_month"] = brent["date"].dt.to_period("M")

    all_rows = []
    grade_stats = {}

    for col in df.columns[1:]:
        raw_grade = parse_grade_from_column(str(col))
        if not raw_grade:
            continue

        info = GRADE_MAP.get(raw_grade)
        if not info:
            print(f"  UKJENT grade: {raw_grade}")
            continue

        if info["grade"] == "Brent Blend (landed)":
            continue

        series = df[[date_col, col]].copy()
        series.columns = ["date", "landed_cost"]
        series["landed_cost"] = pd.to_numeric(series["landed_cost"], errors="coerce")
        series = series.dropna()
        if series.empty:
            continue

        series["year_month"] = series["date"].dt.to_period("M")
        merged = series.merge(brent[["year_month", "brent_price"]], on="year_month", how="inner")
        merged["differential"] = merged["landed_cost"] - merged["brent_price"]
        merged["grade"] = info["grade"]
        merged["api_gravity"] = info["api"]
        merged["sulfur_pct"] = info["sulfur"]
        merged["source"] = "EIA_landed_cost"

        all_rows.append(merged[["grade", "date", "differential", "landed_cost",
                                "brent_price", "api_gravity", "sulfur_pct", "source"]])

        grade_stats[info["grade"]] = {
            "n_obs": len(merged),
            "period": f"{merged['date'].min():%Y-%m} – {merged['date'].max():%Y-%m}",
            "api": info["api"],
            "sulfur": info["sulfur"],
            "mean_diff": merged["differential"].mean(),
        }

    if not all_rows:
        print("Ingen data parset!")
        return

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["grade", "date"]).reset_index(drop=True)

    out_csv = PROCESSED_DIR / "eia_import_differentials.csv"
    result.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"EIA IMPORT-DIFFERENSIALER")
    print(f"{'='*70}")
    print(f"Totalt: {len(result):,} observasjoner, {result['grade'].nunique()} grades")
    print(f"Periode: {result['date'].min():%Y-%m} – {result['date'].max():%Y-%m}")

    print(f"\nPer grade:")
    print(f"{'Grade':25s} {'N':>5s} {'API':>5s} {'S%':>5s} {'Snitt diff':>10s}  Periode")
    print("-" * 75)
    for grade, stats in sorted(grade_stats.items(), key=lambda x: -x[1]["n_obs"]):
        print(f"{grade:25s} {stats['n_obs']:5d} {stats['api']:5.1f} {stats['sulfur']:5.2f} "
              f"{stats['mean_diff']:+10.2f}  {stats['period']}")

    # === Kombiner med eksisterende differensialer ===
    existing = pd.read_csv(PROCESSED_DIR / "global_differentials_monthly.csv")
    eia_for_merge = result[["grade", "date", "differential", "source"]].copy()
    eia_for_merge["date"] = pd.to_datetime(eia_for_merge["date"])
    existing["date"] = pd.to_datetime(existing["date"])

    combined = pd.concat([existing, eia_for_merge], ignore_index=True)
    combined = combined.sort_values(["grade", "date"]).reset_index(drop=True)

    # Fjern duplikater (same grade + month)
    combined["year_month"] = combined["date"].dt.to_period("M")
    combined = combined.drop_duplicates(subset=["grade", "year_month"], keep="first")
    combined = combined.drop(columns=["year_month"])

    out_combined = PROCESSED_DIR / "global_differentials_monthly.csv"
    combined.to_csv(out_combined, index=False)

    print(f"\n{'='*70}")
    print(f"KOMBINERT DIFFERENSIAL-DATASETT")
    print(f"{'='*70}")
    print(f"Totalt: {len(combined):,} observasjoner, {combined['grade'].nunique()} grades")
    print(f"Periode: {combined['date'].min()} – {combined['date'].max()}")

    # Oppdater også kvalitetsdatabasen med nye grades
    quality = pd.read_csv(RAW_DIR / "global_crude_quality.csv")
    existing_grades = set(quality["grade"])
    new_grades = []
    for raw_grade, info in GRADE_MAP.items():
        if info["grade"] not in existing_grades and info["grade"] != "Brent Blend (landed)":
            new_grades.append({
                "grade": info["grade"],
                "country": raw_grade.split()[0],
                "region": "Various",
                "api_gravity": info["api"],
                "sulfur_pct": info["sulfur"],
                "tan_mgkoh": 0.1,
                "viscosity_cst_40c": 10.0,
                "pour_point_c": -10,
                "classification_weight": "light" if info["api"] >= 35 else "medium" if info["api"] >= 25 else "heavy",
                "classification_sulfur": "sweet" if info["sulfur"] < 0.5 else "sour",
                "production_kbpd": 100,
                "is_benchmark": 0,
            })

    if new_grades:
        new_df = pd.DataFrame(new_grades)
        combined_quality = pd.concat([quality, new_df], ignore_index=True)
        combined_quality.to_csv(RAW_DIR / "global_crude_quality.csv", index=False)
        print(f"\n{len(new_grades)} nye grades lagt til kvalitetsdatabasen:")
        for g in new_grades:
            print(f"  {g['grade']:25s} API={g['api_gravity']:5.1f}, S={g['sulfur_pct']:.2f}%")


if __name__ == "__main__":
    main()
