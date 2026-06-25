"""
Script 55 — Stage 1 + Stage 2: Sammensatt forward grade-pris-forecast

KJEDEN:
  STEO forward demand/stocks
        ↓ (Stage 1: script 54)
  Forward crack-spread forecasts
        ↓ (Stage 2: 34b_brent_model)
  Forward grade-spesifikke differensialer mot Brent
        ↓
  Forward realisert pris per grade (+ Brent-prognose)

SCENARIO-ANALYSE:
  Base:     STEO konsensus-forecast
  Bull:     +5% diesel/jet demand (gjenåpning-historie, økt fly-trafikk)
  Bear:     -3% gasoline demand (EV-adopsjon, energieffektivitet)

OUTPUT:
  data/processed/55_forward_forecast.csv  (per måned per scenario per grade)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

STAGE1_JSON = PROC_DIR / "54_crack_spread_models.json"
STAGE2_JSON = PROC_DIR / "34b_brent_model.json"
ELASTICITY_JSON = PROC_DIR / "57_demand_stocks_elasticity.json"
DEMAND_CSV  = PROC_DIR / "53_product_demand.csv"
PANEL_CSV   = PROC_DIR / "regression_panel.csv"
OUT_CSV     = PROC_DIR / "55_forward_forecast.csv"


# ── Brent-prognose per scenario (USD/bbl) ──────────────────────────────────
BRENT_FORECAST = {
    "Base":   {2026: 72.0, 2027: 71.0},
    "Bull":   {2026: 88.0, 2027: 85.0},   # Iran-eskalering
    "Bear":   {2026: 63.0, 2027: 60.0},   # OPEC+ supply-ramp
}

# Demand-scenarier (% av base-prognose)
DEMAND_SCENARIOS = {
    "Base":          {"gasoline": 1.00, "diesel": 1.00, "jet": 1.00},
    "DieselJet-Up":  {"gasoline": 1.00, "diesel": 1.05, "jet": 1.05},
    "Gasoline-Down": {"gasoline": 0.97, "diesel": 1.00, "jet": 1.00},
    "All-Demand-Up": {"gasoline": 1.03, "diesel": 1.05, "jet": 1.05},
}

# Grades vi forecaster (sentrale for AKRBP og NCS-presentasjon)
FOCUS_GRADES = [
    "Johan Sverdrup", "Alvheim", "Ekofisk", "Grane", "Skarv",
    "Heidrun", "Oseberg", "Statfjord", "Bonny Light", "Arab Light",
    "Goliat", "Norne",
]


def load_stage1() -> dict:
    return json.loads(STAGE1_JSON.read_text())


def load_elasticities() -> dict:
    """
    Hent demand → stocks elastisiteter fra script 57.
    Bruk langsiktig elastisitet (β / (1 - lag)) for scenario-analyse.

    For produkter med svak/usignifikant elastisitet (jet), bruker vi
    distillate som proxy siden de er like raffinerings-økonomisk.
    """
    e = json.loads(ELASTICITY_JSON.read_text())
    return {
        # Bruk LR elastisitet (mbbl stocks per kbpd demand-endring i likevekt)
        "gasoline":  e["motor_gasoline"]["long_run_elasticity"],
        "diesel":    e["distillate_fuel"]["long_run_elasticity"],
        "jet":       e["distillate_fuel"]["long_run_elasticity"],   # proxy
    }


def load_stage2() -> dict:
    return json.loads(STAGE2_JSON.read_text())


def load_forward_drivers() -> pd.DataFrame:
    """STEO-forecast: produkt-demand + stocks per måned 2026-2027."""
    df = pd.read_csv(DEMAND_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_forecast"]].copy()   # kun forward
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def load_panel_baseline_features() -> dict:
    """Sist tilgjengelige verdi for andre Stage 2-features (markedsregime)."""
    panel = pd.read_csv(PANEL_CSV)
    panel["date"] = pd.to_datetime(panel["date_str"])
    last = panel.sort_values("date").iloc[-1]
    return {
        "vacuum_resid_x_brent":            float(last.get("vacuum_resid_x_brent", 0)),
        "us_crude_stocks_kbbl_dev_5y_pct": 0.0,
        "cushing_stocks_kbbl_dev_5y_pct":  0.0,
        "d_refinery_slack":                0,
        "fc_slope_4m":                     0.0,
        "brent_dubai_spread":              1.5,
        "diesel_minus_gasoline_crack":     15.0,
        "us_crude_exports_kbpd":           float(last.get("us_crude_exports_kbpd", 4500)),
    }


def predict_crack_spread(model_obj: dict, drivers: dict) -> float:
    """Stage 1: predikter én crack spread for ett tidspunkt."""
    coefs = model_obj["coefficients"]
    pred  = coefs.get("const", 0)
    for f, c in coefs.items():
        if f == "const":
            continue
        v = drivers.get(f, 0)
        pred += c * v
    return pred


def predict_grade_differential(stage2: dict, grade_assay: dict,
                                  market: dict) -> float:
    """Stage 2: predikter grade-spesifikk differensial gitt assay + market features."""
    coefs    = stage2["coefficients"]
    features = stage2["features"]

    feat = {
        # Statisk grade-kvalitet
        "api_gravity":           grade_assay["api"],
        "sulfur_pct":            grade_assay["sulfur"],
        "api2":                  grade_assay["api"] ** 2,
        "vacuum_resid_pct":      grade_assay["vac_res"],
        "middle_distillate_pct": grade_assay["mid_dist"],
        "ccr_wt_pct":            grade_assay["ccr"],
        "log_v_ni":              np.log1p(grade_assay.get("v_ni", 5.0)),
        # Region dummies (NCS by default — annet håndteres for ikke-NCS grades)
        "reg_NorthSea":          1 if grade_assay["region"] == "NCS" else 0,
        "reg_WestAfrica":        1 if grade_assay["region"] == "WestAfrica" else 0,
        "reg_NorthAfrica":       1 if grade_assay["region"] == "NorthAfrica" else 0,
        # Logistikk
        "d_distance_long":       1 if grade_assay["region"] in ("MiddleEast", "WestAfrica") else 0,
        "is_fpso":               grade_assay.get("is_fpso", 0),    # NY i v3
        # Markedsfeatures (fra crack-forecast og baseline)
        "brent_price":                       market["brent"],
        "gasoline_crack_brent":              market["gasoline_crack"],
        "diesel_crack_brent":                market["diesel_crack"],
        "diesel_minus_gasoline_crack":       market["diesel_crack"] - market["gasoline_crack"],
        "brent_dubai_spread":                market.get("brent_dubai_spread", 1.5),
        "us_refinery_util_pct":              market["us_refinery_util_pct"],
        "us_crude_stocks_kbbl_dev_5y_pct":   market.get("us_crude_stocks_kbbl_dev_5y_pct", 0),
        "cushing_stocks_kbbl_dev_5y_pct":    market.get("cushing_stocks_kbbl_dev_5y_pct", 0),
        "d_refinery_slack":                  market.get("d_refinery_slack", 0),
        "fc_slope_4m":                       market.get("fc_slope_4m", 0),
        "cos_month":                         market["cos_month"],
        # Interaksjoner
        "sulfur_x_brent":         grade_assay["sulfur"] * market["brent"],
        "vacuum_resid_x_brent":   grade_assay["vac_res"] * market["brent"],
        "ccr_x_brent":            grade_assay["ccr"] * market["brent"],
        "api_x_contango":         grade_assay["api"] * 0,
        "sulfur_x_refinery_util": grade_assay["sulfur"] * market["us_refinery_util_pct"],
        # Yield × crack-interaksjoner (NY i modellen!)
        "naphtha_x_gasoline_crack":   grade_assay.get("naphtha", 25) * market["gasoline_crack"],
        "diesel_x_diesel_crack":      grade_assay.get("diesel_gasoil", 30) * market["diesel_crack"],
        "vacuum_resid_x_diesel_crack":grade_assay["vac_res"] * market["diesel_crack"],
        "middle_dist_x_diesel_crack": grade_assay["mid_dist"] * market["diesel_crack"],
        # Politikk-dummies (1 i forward-perioden)
        "d_russia_sanctions":     1,
        "d_iran_sanctions_v1":    0,
        "d_iran_sanctions_v2":    1,
        "d_us_shale_boom":        0,
        "d_covid":                0,
        "d_opec_plus_cuts_2023":  1,
    }

    pred = coefs.get("const", 0)
    for f in features:
        if f in feat and f in coefs:
            pred += coefs[f] * feat[f]
    return pred


def build_grade_assays(panel: pd.DataFrame, stage2_grades: list) -> dict:
    """Bygg per-grade assay-dict for forecast-bruk."""
    last = (panel.sort_values("date_str").groupby("grade").last().reset_index())
    last = last[last["grade"].isin(stage2_grades)]

    region_map = {
        "North Sea": "NCS", "Norwegian Sea": "NCS", "Barents Sea": "NCS",
        "Middle East": "MiddleEast", "West Africa": "WestAfrica",
        "North Africa": "NorthAfrica",
    }

    out = {}
    for _, r in last.iterrows():
        if pd.isna(r["api_gravity"]):
            continue
        out[r["grade"]] = {
            "api":          float(r["api_gravity"]),
            "sulfur":       float(r["sulfur_pct"]),
            "vac_res":      float(r.get("vacuum_resid_pct", 15.0) or 15.0),
            "ccr":          float(r.get("ccr_wt_pct", 2.0) or 2.0),
            "mid_dist":     float(r.get("middle_distillate_pct", 40.0) or 40.0),
            "naphtha":      float(r.get("naphtha_pct", 25.0) or 25.0),
            "diesel_gasoil":float(r.get("diesel_gasoil_pct", 30.0) or 30.0),
            "kerosene":     float(r.get("kerosene_pct", 12.0) or 12.0),
            "v_ni":         float(np.expm1(r.get("log_v_ni", np.log1p(5.0)) or np.log1p(5.0))),
            "region":       region_map.get(r["region"], "Other"),
            "is_fpso":      int(r.get("is_fpso", 0)),   # NY i v3
        }
    return out


def main():
    print("=" * 75)
    print("  SCRIPT 55: Sammensatt Stage 1 + Stage 2 forward forecast")
    print("=" * 75)

    print("\n[1] Laster modeller og forecast-drivere...")
    stage1       = load_stage1()
    stage2       = load_stage2()
    forward      = load_forward_drivers()
    baseline     = load_panel_baseline_features()
    elasticities = load_elasticities()
    panel        = pd.read_csv(PANEL_CSV)
    assays       = build_grade_assays(panel, stage2["grades"])

    print(f"  Stage 1 modeller: {list(stage1.keys())}")
    print(f"  Stage 2 grades:   {len(stage2['grades'])}")
    print(f"  Forecast måneder: {len(forward)} ({forward['date'].min().strftime('%Y-%m')} → "
          f"{forward['date'].max().strftime('%Y-%m')})")
    print(f"  Grades med assay: {len(assays)}")
    print(f"  Demand→Stocks elastisiteter (mbbl per kbpd):")
    for prod, el in elasticities.items():
        print(f"    {prod:<12} {el:+.5f}")

    # ── Bygg scenarier ──────────────────────────────────────────────────────
    print("\n[2] Genererer forecasts per scenario...")
    all_rows = []

    for d_scen_name, d_mult in DEMAND_SCENARIOS.items():
        for b_scen_name, b_path in BRENT_FORECAST.items():
            scen_label = f"{b_scen_name} × {d_scen_name}"

            for _, mo in forward.iterrows():
                # ── Apply demand-scenario multiplier + propagate to stocks ─
                # Demand-shock i kbpd
                gas_dem_shock = mo["motor_gasoline_kbpd"]  * (d_mult["gasoline"] - 1)
                dsl_dem_shock = mo["distillate_fuel_kbpd"] * (d_mult["diesel"]   - 1)
                jet_dem_shock = mo["jet_fuel_kbpd"]         * (d_mult["jet"]     - 1)

                # Stocks-respons via elastisitet (β = mbbl per kbpd, neg = drawdown)
                gas_stocks_shock = elasticities["gasoline"] * gas_dem_shock
                dsl_stocks_shock = elasticities["diesel"]   * dsl_dem_shock
                jet_stocks_shock = elasticities["jet"]      * jet_dem_shock

                drivers_gasoline = {
                    "motor_gasoline_kbpd":         mo["motor_gasoline_kbpd"] + gas_dem_shock,
                    "motor_gasoline_stocks_mbbl":  mo["motor_gasoline_stocks_mbbl"] + gas_stocks_shock,
                    "us_refinery_util_pct":        92.0,
                    "brent_price":                 b_path[mo["year"]],
                    "sin_month":                   mo["sin_month"],
                    "cos_month":                   mo["cos_month"],
                    "d_russia_sanctions":          1,
                }
                drivers_diesel = {
                    "distillate_fuel_kbpd":       mo["distillate_fuel_kbpd"] + dsl_dem_shock,
                    "distillate_stocks_mbbl":     mo["distillate_stocks_mbbl"] + dsl_stocks_shock,
                    "us_refinery_util_pct":       92.0,
                    "brent_price":                b_path[mo["year"]],
                    "sin_month":                  mo["sin_month"],
                    "cos_month":                  mo["cos_month"],
                    "d_russia_sanctions":         1,
                }
                drivers_jet = {
                    "jet_fuel_kbpd":              mo["jet_fuel_kbpd"] + jet_dem_shock,
                    "jet_fuel_stocks_mbbl":       mo["jet_fuel_stocks_mbbl"] + jet_stocks_shock,
                    "us_refinery_util_pct":       92.0,
                    "brent_price":                b_path[mo["year"]],
                    "sin_month":                  mo["sin_month"],
                    "cos_month":                  mo["cos_month"],
                    "d_russia_sanctions":         1,
                }

                # ── Stage 1: predict crack spreads ─────────────────────────
                gas_crack = predict_crack_spread(stage1["gasoline_crack_brent"], drivers_gasoline)
                dsl_crack = predict_crack_spread(stage1["diesel_crack_brent"],    drivers_diesel)
                jet_crack = predict_crack_spread(stage1["jet_crack_brent"],       drivers_jet)

                # ── Stage 2: predict per grade ─────────────────────────────
                market = {
                    "brent":                  b_path[mo["year"]],
                    "gasoline_crack":         gas_crack,
                    "diesel_crack":           dsl_crack,
                    "us_refinery_util_pct":   92.0,
                    "cos_month":              mo["cos_month"],
                    **baseline,
                }

                for grade in FOCUS_GRADES:
                    if grade not in assays:
                        continue
                    diff = predict_grade_differential(stage2, assays[grade], market)
                    all_rows.append({
                        "scenario":       scen_label,
                        "brent_scen":     b_scen_name,
                        "demand_scen":    d_scen_name,
                        "date":           mo["date"],
                        "year":           mo["year"],
                        "month":          mo["month"],
                        "brent":          market["brent"],
                        "gasoline_crack": gas_crack,
                        "diesel_crack":   dsl_crack,
                        "jet_crack":      jet_crack,
                        "grade":          grade,
                        "differential":   diff,
                        "realized_pred":  market["brent"] + diff,
                    })

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  ✓ Lagret: {OUT_CSV.name} ({len(df):,} rader)")

    # ── Sammendrag: per scenario per grade (gjennomsnitt 2026-2027) ────────
    print(f"\n[3] Forecast-sammendrag (gjennomsnitt 2026-2027 per scenario):")
    print(f"{'─' * 90}")
    summary = (df.groupby(["scenario", "grade"])
                 .agg(brent=("brent","mean"),
                      diff=("differential","mean"),
                      realized=("realized_pred","mean"))
                 .reset_index())

    # Vis kun base + 2 interessante scenarier
    show_scens = ["Base × Base", "Base × DieselJet-Up", "Bull × All-Demand-Up", "Bear × Gasoline-Down"]
    show_grades = ["Johan Sverdrup", "Alvheim", "Skarv", "Bonny Light"]

    for scen in show_scens:
        sub = summary[summary["scenario"] == scen]
        if len(sub) == 0:
            continue
        print(f"\n  ▶ Scenario: {scen}")
        print(f"    {'Grade':<22} {'Brent':>7} {'Diff':>7} {'Realized':>9}")
        for g in show_grades:
            r = sub[sub["grade"] == g]
            if len(r) > 0:
                row = r.iloc[0]
                print(f"    {g:<22} {row['brent']:>7.2f} {row['diff']:>+7.2f} {row['realized']:>9.2f}")


if __name__ == "__main__":
    main()
