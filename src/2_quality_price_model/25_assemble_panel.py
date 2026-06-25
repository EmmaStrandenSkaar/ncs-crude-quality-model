"""
Assembler komplett paneldatasett for regresjonsanalyse:
  grade × måned med kvalitetsvariable + markedskontroller.

Input:
  - global_crude_quality.csv (statisk kvalitet per grade)
  - global_differentials_monthly.csv (differensialer, tidsvarierende)
  - market_controls_monthly.csv (markedskontroller, tidsvarierende)

Output:
  - regression_panel.csv — ferdig til regresjon
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GRADE_NAME_MAP = {
    "ALVHEIM": "Alvheim",
    "BALDER": "Balder",
    "DRAUGEN": "Draugen",
    "EKOFISK": "Ekofisk",
    "GINA KROG": "Gina Krog",
    "GOLIAT": "Goliat",
    "GRANE": "Grane",
    "GUDRUN": "Gudrun",
    "GULLFAKS": "Gullfaks",
    "HEIDRUN": "Heidrun",
    "JOHAN SVERDRUP": "Johan Sverdrup",
    "JOTUN": "Jotun",
    "KNARR": "Knarr",
    "MARTIN LINGE": "Martin Linge",
    "NJORD": "Njord",
    "NORNE": "Norne",
    "OSEBERG": "Oseberg",
    "SKARV": "Skarv",
    "STATFJORD": "Statfjord",
    "TROLL": "Troll",
    "VOLVE": "Volve",
    "YME": "Yme",
    "ÅSGARD": "Asgard",
    "VALHALL": "Valhall",
    "ULA": "Ula",
}


def normalize_grade(name: str) -> str:
    name = name.strip()
    if name in GRADE_NAME_MAP:
        return GRADE_NAME_MAP[name]
    return name


def main() -> None:
    # === 1. Les unified assay-database (erstatter separate kvalitets- + assay-filer) ===
    unified_path = PROCESSED_DIR / "unified_crude_assays.csv"
    if unified_path.exists():
        unified = pd.read_csv(unified_path)
        print(f"Unified assay-database: {len(unified)} grades (verifiserte data)")
        # Bygg quality-subset med kolonnene panelet trenger
        # Hent country/region/classification fra gammel quality-fil
        old_quality = pd.read_csv(RAW_DIR / "global_crude_quality.csv")
        old_meta = old_quality[["grade", "country", "region", "classification_weight",
                                 "classification_sulfur", "production_kbpd", "is_benchmark"]]
        # Merge unified assay → meta
        quality = unified.merge(old_meta, on="grade", how="left")
        # Rename for kompatibilitet med gammel kode
        if "ccr_pct" in quality.columns and "ccr_wt_pct" not in quality.columns:
            quality["ccr_wt_pct"] = quality["ccr_pct"].fillna(quality.get("mcr_pct", np.nan))
        if "diesel_pct" in quality.columns and "diesel_gasoil_pct" not in quality.columns:
            quality["diesel_gasoil_pct"] = quality["diesel_pct"].fillna(0) + quality.get("heavy_diesel_pct", pd.Series(0)).fillna(0)
        # Bruk unified som assay-kilde også
        assay_cols = ["grade", "naphtha_pct", "kerosene_pct", "diesel_gasoil_pct",
                      "vgo_pct", "vacuum_resid_pct", "ccr_wt_pct",
                      "vanadium_ppm", "nickel_ppm", "nitrogen_ppm",
                      "asphaltenes_pct"]
        assay_cols = [c for c in assay_cols if c in quality.columns]
        assays = quality[assay_cols].copy()
        assays["source_confidence"] = quality.get("confidence", "high")

        # Fallback: for grades uten destillasjonskutt, bruk gammel crude_assays.csv
        old_assay_path = RAW_DIR / "crude_assays.csv"
        if old_assay_path.exists():
            old_assays = pd.read_csv(old_assay_path)
            # Rename old columns to match
            if "diesel_gasoil_pct" in old_assays.columns:
                pass  # allerede riktig
            missing_mask = assays["naphtha_pct"].isna() | assays["vgo_pct"].isna()
            missing_grades = assays.loc[missing_mask, "grade"].tolist()
            if missing_grades:
                for g in missing_grades:
                    old_row = old_assays[old_assays["grade"] == g]
                    if old_row.empty:
                        continue
                    idx = assays[assays["grade"] == g].index
                    for col in ["naphtha_pct", "kerosene_pct", "diesel_gasoil_pct",
                                "vgo_pct", "vacuum_resid_pct", "ccr_wt_pct",
                                "vanadium_ppm", "nickel_ppm", "nitrogen_ppm", "asphaltenes_pct"]:
                        if col in old_row.columns and col in assays.columns:
                            if assays.loc[idx, col].isna().any():
                                assays.loc[idx, col] = old_row[col].values[0]
                    assays.loc[idx, "source_confidence"] = "estimated"
                print(f"  Fallback til estimerte kutt for: {missing_grades}")
    else:
        # Fallback: bruk gamle separate filer
        quality = pd.read_csv(RAW_DIR / "global_crude_quality.csv")
        print(f"Kvalitetsdata: {len(quality)} grades (gammel fil)")
        assays = pd.read_csv(RAW_DIR / "crude_assays.csv")
        print(f"Assay-data: {len(assays)} grades med destillasjonskutt + forurensninger")

    # === 1c. Les grade-logistikk (distance-to-market, landlocked) ===
    logistics = pd.read_csv(RAW_DIR / "grade_logistics.csv")
    print(f"Logistikk-data: {len(logistics)} grades med distance-band + landlocked-flag")

    # === 2. Les differensialer ===
    diffs = pd.read_csv(PROCESSED_DIR / "global_differentials_monthly.csv")
    diffs["date"] = pd.to_datetime(diffs["date"])
    diffs["grade"] = diffs["grade"].apply(normalize_grade)
    diffs["year_month"] = diffs["date"].dt.to_period("M")
    print(f"Differensialer: {len(diffs):,} obs, {diffs['grade'].nunique()} grades")

    # === 3. Les markedskontroller ===
    controls = pd.read_csv(PROCESSED_DIR / "market_controls_monthly.csv")
    controls["year_month"] = pd.PeriodIndex(controls["year_month_str"], freq="M")
    control_cols = [
        "year_month", "brent_price", "wti_price", "wti_brent_spread",
        "brent_log", "brent_pct_change", "brent_regime",
        "brent_volatility_3m", "usd_broad_index", "vix",
        "crack_spread_proxy", "month", "quarter", "year",
        "gasoline_crack_brent", "diesel_crack_brent", "jet_crack_brent",
        "crack_321_brent", "diesel_minus_gasoline_crack", "brent_dubai_spread",
        "d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
        "d_venezuela_sanctions", "d_us_shale_boom", "d_opec_cuts_2017",
        "d_covid", "d_opec_plus_cuts_2023",
    ]
    controls = controls[[c for c in control_cols if c in controls.columns]]
    print(f"Markedskontroller: {len(controls)} måneder, {len(controls.columns)-1} variabler")

    # === 4. Match kvalitet til differensialer ===
    quality_cols = [
        "grade", "country", "region", "api_gravity", "sulfur_pct",
        "tan_mgkoh", "viscosity_cst_40c", "pour_point_c",
        "classification_weight", "classification_sulfur",
        "production_kbpd", "is_benchmark",
    ]
    quality_cols = [c for c in quality_cols if c in quality.columns]
    quality_sub = quality[quality_cols].drop_duplicates(subset="grade", keep="first")

    panel = diffs.merge(quality_sub, on="grade", how="left")

    matched = panel["api_gravity"].notna().sum()
    unmatched_grades = panel.loc[panel["api_gravity"].isna(), "grade"].unique()
    print(f"\nMatching kvalitet: {matched:,}/{len(panel):,} obs")
    if len(unmatched_grades) > 0:
        print(f"  Grades uten kvalitetsdata: {list(unmatched_grades)}")

    panel = panel.dropna(subset=["api_gravity"])

    # === 4b. Slå inn assay-data ===
    panel = panel.merge(assays, on="grade", how="left")
    assay_matched = panel["vacuum_resid_pct"].notna().sum()
    assay_missing_grades = panel.loc[panel["vacuum_resid_pct"].isna(), "grade"].unique()
    print(f"Matching assay: {assay_matched:,}/{len(panel):,} obs")
    if len(assay_missing_grades) > 0:
        print(f"  Grades uten assay-data: {list(assay_missing_grades)}")

    # === 4c. Slå inn logistikk-data ===
    panel = panel.merge(logistics, on="grade", how="left")
    log_matched = panel["distance_band"].notna().sum()
    log_missing_grades = panel.loc[panel["distance_band"].isna(), "grade"].unique()
    print(f"Matching logistikk: {log_matched:,}/{len(panel):,} obs")
    if len(log_missing_grades) > 0:
        print(f"  Grades uten logistikk-data: {list(log_missing_grades)}")

    # === 5. Legg til markedskontroller ===
    panel = panel.merge(controls, on="year_month", how="left")
    has_controls = panel["brent_price"].notna().sum()
    print(f"Etter matching med kontroller: {has_controls:,}/{len(panel):,} obs")

    panel = panel.dropna(subset=["brent_price"])

    # === 5b. Legg til EIA fundamentals (lagre, raffinerikapasitet) ===
    eia_fund_path = PROCESSED_DIR / "eia_fundamentals_monthly.csv"
    if eia_fund_path.exists():
        eia_fund = pd.read_csv(eia_fund_path)
        eia_fund["year_month"] = pd.PeriodIndex(eia_fund["year_month"], freq="M")
        fund_cols = ["year_month", "us_crude_stocks_kbbl_dev_5y_pct",
                     "cushing_stocks_kbbl_dev_5y_pct", "us_refinery_util_pct",
                     "refinery_util_3m_change", "d_refinery_tight", "d_refinery_slack",
                     "us_crude_exports_kbpd", "us_days_supply_crude",
                     "us_net_crude_imports_kbpd"]
        fund_cols = [c for c in fund_cols if c in eia_fund.columns]
        panel = panel.merge(eia_fund[fund_cols], on="year_month", how="left")
        print(f"Slått inn EIA fundamentals: {len(fund_cols)-1} variabler")

    # === 5c. Legg til forward curve (contango/backwardation) ===
    fc_path = PROCESSED_DIR / "forward_curve_monthly.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path)
        fc["year_month"] = pd.PeriodIndex(fc["year_month"], freq="M")
        fc_cols = ["year_month", "fc_slope_4m", "fc_slope_pct", "d_contango",
                   "d_strong_contango", "d_strong_backwardation"]
        fc_cols = [c for c in fc_cols if c in fc.columns]
        panel = panel.merge(fc[fc_cols], on="year_month", how="left")
        print(f"Slått inn forward curve: {len(fc_cols)-1} variabler")

    # === 6. Feature engineering ===
    panel["api2"] = panel["api_gravity"] ** 2
    panel["sulfur2"] = panel["sulfur_pct"] ** 2
    panel["api_x_sulfur"] = panel["api_gravity"] * panel["sulfur_pct"]
    panel["api_x_brent"] = panel["api_gravity"] * panel["brent_price"]
    panel["sulfur_x_brent"] = panel["sulfur_pct"] * panel["brent_price"]
    panel["log_production"] = np.log1p(panel["production_kbpd"])

    panel["is_light"] = (panel["api_gravity"] >= 35).astype(int)
    panel["is_heavy"] = (panel["api_gravity"] < 25).astype(int)
    panel["is_sweet"] = (panel["sulfur_pct"] < 0.5).astype(int)
    panel["is_sour"] = (panel["sulfur_pct"] >= 1.0).astype(int)
    panel["is_condensate"] = (panel["api_gravity"] >= 45).astype(int)

    # === 6b. Assay-baserte features ===
    # Refining-økonomi: hvor mye av fatet er høyverdige produkter?
    panel["light_yield_pct"] = panel["naphtha_pct"] + panel["kerosene_pct"]
    panel["middle_distillate_pct"] = panel["kerosene_pct"] + panel["diesel_gasoil_pct"]
    panel["bottom_of_barrel_pct"] = panel["vgo_pct"] + panel["vacuum_resid_pct"]
    panel["high_value_yield_pct"] = (
        panel["naphtha_pct"] + panel["kerosene_pct"] + panel["diesel_gasoil_pct"]
    )

    # Log-transform metals (svært skjev fordeling)
    panel["log_v_ni"] = np.log1p(panel["vanadium_ppm"] + panel["nickel_ppm"])

    # Interaksjoner: refining-økonomi × crack spreads
    if "diesel_crack_brent" in panel.columns:
        panel["middle_dist_x_diesel_crack"] = (
            panel["middle_distillate_pct"] * panel["diesel_crack_brent"]
        )
        panel["vacuum_resid_x_diesel_crack"] = (
            panel["vacuum_resid_pct"] * panel["diesel_crack_brent"]
        )
    if "gasoline_crack_brent" in panel.columns:
        panel["naphtha_x_gasoline_crack"] = (
            panel["naphtha_pct"] * panel["gasoline_crack_brent"]
        )

    # Vacuum residue × Brent: når Brent er høyt taper du mer på bottom-of-barrel
    panel["vacuum_resid_x_brent"] = panel["vacuum_resid_pct"] * panel["brent_price"]
    panel["ccr_x_brent"] = panel["ccr_wt_pct"] * panel["brent_price"]
    panel["metals_x_brent"] = panel["log_v_ni"] * panel["brent_price"]

    # Russland-spesifikk interaksjon: Urals/ESPO påvirkes ekstremt av sanksjoner
    if "d_russia_sanctions" in panel.columns:
        panel["is_russian"] = (panel["country"].str.contains("Russia", case=False, na=False)).astype(int)
        panel["russia_sanctions_x_russian"] = panel["d_russia_sanctions"] * panel["is_russian"]

    # === 6c. Logistikk-features ===
    if "distance_band" in panel.columns:
        panel["d_distance_medium"] = (panel["distance_band"] == "medium").astype(int)
        panel["d_distance_long"] = (panel["distance_band"] == "long").astype(int)
        # is_landlocked og is_pipeline_constrained er allerede 0/1

        # Interaksjon: landlocked × Cushing stocks (WTI får ekstra rabatt når Cushing fyller seg)
        if "cushing_stocks_kbbl_dev_5y_pct" in panel.columns:
            panel["landlocked_x_cushing_stocks"] = (
                panel["is_landlocked"].fillna(0) * panel["cushing_stocks_kbbl_dev_5y_pct"]
            )

    # === 6d. Forward curve interaksjoner ===
    if "fc_slope_4m" in panel.columns:
        # Contango → lette/sweet crudes får ekstra premie (storage-arbitrage)
        panel["api_x_contango"] = panel["api_gravity"] * panel["fc_slope_4m"]
        # Contango × landlocked: WTI lider mest under contango
        if "is_landlocked" in panel.columns:
            panel["landlocked_x_contango"] = (
                panel["is_landlocked"].fillna(0) * panel["fc_slope_4m"]
            )

    # === 6e. Sesongeffekter (sin/cos) ===
    if "month" in panel.columns:
        panel["sin_month"] = np.sin(2 * np.pi * panel["month"] / 12)
        panel["cos_month"] = np.cos(2 * np.pi * panel["month"] / 12)
        # Vinter-flag (Q1+Q4)
        panel["d_winter"] = panel["month"].isin([12, 1, 2]).astype(int)
        panel["d_summer"] = panel["month"].isin([6, 7, 8]).astype(int)

    # === 6f. Refinery utilization interaksjoner ===
    if "us_refinery_util_pct" in panel.columns:
        # Tight refinery × heavy/sour: når raffinerier kjører fullt, smalner heavy/sour-rabatten
        panel["sulfur_x_refinery_util"] = (
            panel["sulfur_pct"] * panel["us_refinery_util_pct"]
        )
        panel["vacuum_resid_x_refinery_util"] = (
            panel["vacuum_resid_pct"] * panel["us_refinery_util_pct"]
        )

    panel["year_month_str"] = panel["year_month"].astype(str)
    panel["date_str"] = panel["date"].dt.strftime("%Y-%m-%d")

    # === 7. Lagre ===
    out_csv = PROCESSED_DIR / "regression_panel.csv"
    drop_cols = ["year_month", "date"]
    panel.drop(columns=drop_cols, errors="ignore").to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"PANELDATASETT LAGRET: {out_csv}")
    print(f"{'='*70}")
    print(f"Observasjoner:    {len(panel):,}")
    print(f"Grades:           {panel['grade'].nunique()}")
    print(f"Periode:          {panel['year_month'].min()} – {panel['year_month'].max()}")
    print(f"Kolonner:         {panel.shape[1]}")
    print(f"\nGrades i panelet:")
    for grade in sorted(panel["grade"].unique()):
        sub = panel[panel["grade"] == grade]
        print(f"  {grade:25s}: {len(sub):>4} obs, "
              f"API={sub['api_gravity'].iloc[0]:5.1f}, "
              f"S={sub['sulfur_pct'].iloc[0]:5.2f}%, "
              f"diff snitt={sub['differential'].mean():+.2f}")

    print(f"\nAvhengig variabel (differential):")
    print(f"  Snitt:     {panel['differential'].mean():+.2f} USD/fat")
    print(f"  Std:       {panel['differential'].std():.2f}")
    print(f"  Min:       {panel['differential'].min():+.2f}")
    print(f"  Max:       {panel['differential'].max():+.2f}")

    print(f"\nKvalitetsvariable:")
    for col in ["api_gravity", "sulfur_pct", "tan_mgkoh", "production_kbpd"]:
        if col in panel.columns:
            print(f"  {col:20s}: mean={panel[col].mean():.2f}, "
                  f"std={panel[col].std():.2f}, "
                  f"range=[{panel[col].min():.2f}, {panel[col].max():.2f}]")

    print(f"\nMarkedskontroller:")
    for col in ["brent_price", "wti_brent_spread", "vix", "crack_spread_proxy"]:
        if col in panel.columns:
            print(f"  {col:20s}: mean={panel[col].mean():.2f}, "
                  f"std={panel[col].std():.2f}")


if __name__ == "__main__":
    main()
