"""
Script 59 — Forbedre Modell B med transport-distanse til Rotterdam.

HYPOTESE (fra residual-diagnostikk i script 58):
  Norske Sjø-felt OVER-predikeres systematisk (Martin Linge -2.02, Asgard -1.54,
  Norne -0.91 etc.). Nordsjø-felt UNDER-predikeres. Den nåværende d_distance_long
  dummy-en (binær) er for grov — den fanger ikke variasjonen i transport-kostnad
  mellom Ekofisk (521 km til Rotterdam) og Norne (1670 km).

TEST:
  Legg til kontinuerlig dist_rotterdam_km (eller log-versjon) som feature i
  Modell B. Re-tren med stepwise og GRADE-CLUSTERED standard errors (kritisk
  fordi distanse er tids-invariant per grade — effektiv N = antall grades).

EVALUERING:
  · Er den nye variabelen signifikant med grade-clustered SE?
  · Er Norske Sjø-bias-en redusert?
  · Forbedres OOT R²?

OUTPUT:
  data/processed/34b_brent_model_v2.json     (kun hvis bedre enn nåværende)
  data/processed/59_distance_test_results.json
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
PANEL_CSV    = PROC_DIR / "regression_panel.csv"
OLD_MODEL    = PROC_DIR / "34b_brent_model.json"
NEW_MODEL    = PROC_DIR / "34b_brent_model_v2.json"
RESULTS_JSON = PROC_DIR / "59_distance_test_results.json"

WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}
ALWAYS_KEEP   = {"api_gravity", "sulfur_pct"}
ALWAYS_DROP   = {"vix"}
SIG_THRESHOLD = 0.10

INTERACTION_PARENTS = {
    "api2": ["api_gravity"],
    "sulfur_x_brent": ["sulfur_pct", "brent_price"],
    "vacuum_resid_x_brent": ["vacuum_resid_pct", "brent_price"],
    "ccr_x_brent": ["ccr_wt_pct", "brent_price"],
    "api_x_contango": ["api_gravity", "fc_slope_4m"],
    "sulfur_x_refinery_util": ["sulfur_pct", "us_refinery_util_pct"],
    "naphtha_x_gasoline_crack": ["naphtha_pct", "gasoline_crack_brent"],
    "kerosene_x_jet_crack": ["kerosene_pct", "jet_crack_brent"],
    "diesel_x_diesel_crack": ["diesel_gasoil_pct", "diesel_crack_brent"],
    "vacuum_resid_x_diesel_crack": ["vacuum_resid_pct", "diesel_crack_brent"],
    "middle_dist_x_diesel_crack": ["middle_distillate_pct", "diesel_crack_brent"],
}


def load_panel_brent() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV)
    df["date"] = pd.to_datetime(df["date_str"])
    df = df[~df["grade"].isin(WTI_LINKED)].copy()
    region_simple = {
        "North Sea": "NorthSea", "Norwegian Sea": "NorthSea", "Barents Sea": "NorthSea",
        "North America": "NorthAmerica", "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm", "Middle East": "MiddleEast",
        "West Africa": "WestAfrica", "North Africa": "NorthAfrica",
        "FSU": "FSU", "Asia-Pacific": "AsiaPac", "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")
    return df


def build_features_with_distance(df: pd.DataFrame) -> list:
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg",
                                  drop_first=True, dtype=int)
    df_ext = pd.concat([df, region_dums], axis=1)
    region_cols = [c for c in region_dums.columns
                    if df_ext[c].sum() > 0 and df_ext[c].sum() < len(df_ext)]
    for c in region_cols:
        df[c] = df_ext[c].values

    BRENT_EXCLUDE = {
        "is_landlocked", "is_pipeline_constrained",
        "d_distance_medium", "landlocked_x_cushing_stocks",
        "landlocked_x_contango", "wti_brent_spread",
        "d_venezuela_sanctions",
    }

    initial = (
        ["api_gravity", "sulfur_pct", "api2"]
        + region_cols
        + ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"]
        + ["brent_price", "vix"]
        + ["gasoline_crack_brent", "diesel_crack_brent", "jet_crack_brent",
           "diesel_minus_gasoline_crack", "brent_dubai_spread"]
        + ["d_distance_long",
           "log_dist_rotterdam"]            # ⭐ NY: kontinuerlig avstand
        + ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
           "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
           "d_refinery_tight", "d_refinery_slack"]
        + ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"]
        + ["sin_month", "cos_month", "d_winter"]
        + ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
           "api_x_contango", "sulfur_x_refinery_util"]
        + ["naphtha_x_gasoline_crack", "kerosene_x_jet_crack",
           "diesel_x_diesel_crack", "vacuum_resid_x_diesel_crack",
           "middle_dist_x_diesel_crack"]
        + ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
           "d_us_shale_boom", "d_covid", "d_opec_plus_cuts_2023"]
    )
    return [f for f in initial if f in df.columns and f not in BRENT_EXCLUDE]


def fit_with_clustered_se(df, features):
    """Tilpass OLS med grade-clustered standard errors (kritisk for tids-invariante features)."""
    sub = df.dropna(subset=features + ["differential", "grade"]).copy()
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
    # Bruk grade som cluster (effective N = antall grades for tids-invariante vars)
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub["grade"]})

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(LinearRegression(), sub[features].values, y.values, cv=kf, scoring="r2")

    cutoff = sub["date"].max() - pd.DateOffset(months=24)
    train = sub[sub["date"] <= cutoff]
    test  = sub[sub["date"] > cutoff]
    r2_oot = np.nan
    if len(train) > 100 and len(test) > 30:
        lr = LinearRegression().fit(train[features].values, train["differential"].values)
        pred = lr.predict(test[features].values)
        ss_res = ((test["differential"].values - pred) ** 2).sum()
        ss_tot = ((test["differential"].values - test["differential"].mean()) ** 2).sum()
        r2_oot = 1 - ss_res / ss_tot

    rmse = np.sqrt(((y - m.fittedvalues) ** 2).mean())
    return {
        "model": m, "n": len(sub), "k": len(features),
        "r2": m.rsquared, "r2_adj": m.rsquared_adj,
        "r2_cv": cv.mean(), "r2_oot": r2_oot, "rmse": rmse,
        "features": features, "data": sub,
        "grades": sorted(sub["grade"].unique().tolist()),
    }


def stepwise(df, initial_features):
    features = [f for f in initial_features if f not in ALWAYS_DROP]
    it = 0
    while True:
        it += 1
        result = fit_with_clustered_se(df, features)
        m = result["model"]
        pvals = m.pvalues.drop("const", errors="ignore")
        protected = set(ALWAYS_KEEP)
        for inter, parents in INTERACTION_PARENTS.items():
            if inter in features and inter in pvals.index and pvals[inter] < SIG_THRESHOLD:
                protected.update(parents)
        candidates = pvals[pvals > SIG_THRESHOLD].sort_values(ascending=False)
        candidates = candidates[~candidates.index.isin(protected)]
        if len(candidates) == 0:
            print(f"  Konvergerte etter {it} iter. k={len(features)}, R²={result['r2']:.4f}, "
                  f"CV={result['r2_cv']:.4f}, OOT={result['r2_oot']:.4f}, RMSE={result['rmse']:.2f}")
            return result
        worst = candidates.index[0]
        features = [f for f in features if f != worst]
        if len(features) < 5:
            return result


def compare_residuals_by_grade(old_model: dict, new_result: dict, df_full: pd.DataFrame):
    """Sammenlign per-grade bias før vs etter."""
    # Gammel modell — beregn på samme datasett
    old_coefs = old_model["coefficients"]
    old_feats = old_model["features"]

    # Sett opp region dummies hvis ikke gjort
    region_dums = pd.get_dummies(df_full["region_simple"], prefix="reg",
                                  drop_first=True, dtype=int)
    for c in region_dums.columns:
        if c not in df_full.columns:
            df_full[c] = region_dums[c]

    sub = df_full.dropna(subset=old_feats + ["differential"]).copy()
    coef_vec = np.array([old_coefs.get("const", 0)] + [old_coefs.get(f, 0) for f in old_feats])
    X = np.column_stack([np.ones(len(sub))] + [sub[f].astype(float).values for f in old_feats])
    sub["old_pred"] = X @ coef_vec
    sub["old_resid"] = sub["differential"] - sub["old_pred"]

    old_bias = sub.groupby("grade")["old_resid"].mean()

    # Ny modell
    new_data = new_result["data"].copy()
    new_data["new_pred"]  = new_result["model"].fittedvalues
    new_data["new_resid"] = new_data["differential"] - new_data["new_pred"]
    new_bias = new_data.groupby("grade")["new_resid"].mean()

    return old_bias, new_bias


def main():
    print("=" * 75)
    print("  SCRIPT 59: Test om dist_rotterdam_km forbedrer Modell B")
    print("=" * 75)

    print("\n[1] Laster data og bygger features...")
    df = load_panel_brent()
    feats = build_features_with_distance(df)
    has_dist = "log_dist_rotterdam" in feats
    print(f"  Panel: {len(df)} obs, {df['grade'].nunique()} grades")
    print(f"  Initial features: {len(feats)} (inkluderer log_dist_rotterdam: {has_dist})")

    if not has_dist:
        print("  ⚠ log_dist_rotterdam ikke funnet — kjør først script 46!")
        return

    print(f"\n[2] Trener modell med GRADE-CLUSTERED standard errors...")
    result = stepwise(df, feats)

    # ── Sjekk om distanse-variabelen overlevde stepwise ──────────────────────
    final_features = result["features"]
    m = result["model"]
    print(f"\n[3] STATUS for log_dist_rotterdam i final modell:")
    if "log_dist_rotterdam" in final_features:
        coef = m.params["log_dist_rotterdam"]
        se   = m.bse["log_dist_rotterdam"]
        pval = m.pvalues["log_dist_rotterdam"]
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "."
        print(f"  ✓ BEHOLDT: coef={coef:+.4f}  SE_cluster={se:.4f}  p={pval:.4f} {sig}")
        print(f"  Tolkning: +1 log-enhet (≈ doble distansen) → {coef:+.2f} USD/bbl differensial")
        print(f"  Eksempel: Norne (1670 km) vs Ekofisk (521 km) = log-forskjell {np.log(1670)-np.log(521):.2f}")
        print(f"            → predikert diff-forskjell: {coef * (np.log(1670)-np.log(521)):+.2f} USD/bbl")
    else:
        print(f"  ✗ ELIMINERT av stepwise (ikke signifikant med grade-clustered SE)")
        print(f"  Konklusjon: tids-invariante variabler har EFFEKTIV N = antall grades,")
        print(f"  og {df['grade'].nunique()} grades er ikke nok for signifikant identifikasjon.")

    # ── Sammenlign med gammel modell ────────────────────────────────────────
    print(f"\n[4] SAMMENLIGNING med eksisterende Modell B:")
    old_model = json.loads(OLD_MODEL.read_text())
    print(f"  {'Metrikk':<15} {'Gammel':>10} {'Ny':>10} {'Δ':>8}")
    print(f"  {'-' * 45}")
    print(f"  {'k features':<15} {old_model['metrics']['k']:>10} {result['k']:>10} {result['k']-old_model['metrics']['k']:>+8}")
    print(f"  {'R²':<15} {old_model['metrics']['r2']:>10.4f} {result['r2']:>10.4f} {result['r2']-old_model['metrics']['r2']:>+8.4f}")
    print(f"  {'CV R²':<15} {old_model['metrics']['r2_cv']:>10.4f} {result['r2_cv']:>10.4f} {result['r2_cv']-old_model['metrics']['r2_cv']:>+8.4f}")
    print(f"  {'OOT R²':<15} {old_model['metrics']['r2_oot']:>10.4f} {result['r2_oot']:>10.4f} {result['r2_oot']-old_model['metrics']['r2_oot']:>+8.4f}")
    print(f"  {'RMSE':<15} {old_model['metrics']['rmse']:>10.4f} {result['rmse']:>10.4f} {result['rmse']-old_model['metrics']['rmse']:>+8.4f}")

    # ── Sammenlign per-grade bias ───────────────────────────────────────────
    print(f"\n[5] PER-GRADE BIAS — før vs. etter (kun problem-felt fra script 58):")
    problem_grades = ["Martin Linge", "Asgard", "Norne", "Draugen", "Statfjord",
                       "Ekofisk", "Alvheim", "Goliat", "Gudrun", "Grane"]
    old_bias, new_bias = compare_residuals_by_grade(old_model, result, df.copy())
    print(f"  {'Grade':<22} {'Gammel bias':>12} {'Ny bias':>10} {'Forbedring':>12}")
    print(f"  {'-' * 60}")
    for g in problem_grades:
        ob = old_bias.get(g, np.nan)
        nb = new_bias.get(g, np.nan)
        if not (np.isnan(ob) or np.isnan(nb)):
            improvement = abs(ob) - abs(nb)
            arrow = "↓" if improvement > 0.05 else "↑" if improvement < -0.05 else "≈"
            print(f"  {g:<22} {ob:>+12.2f} {nb:>+10.2f} {improvement:>+11.2f} {arrow}")

    # ── Lagre ny modell hvis den er BEDRE ───────────────────────────────────
    # Kriterium: OOT R² er det som teller mest for forecasting. RMSE kan være
    # marginalt høyere fordi mindre features = mindre overfitting.
    is_better = (result["r2_oot"] > old_model["metrics"]["r2_oot"] + 0.02 and
                  result["rmse"] < old_model["metrics"]["rmse"] * 1.05)

    if is_better:
        print(f"\n[6] ✓ NY MODELL ER BEDRE — lagrer som v2 (anbefales å erstatte v1):")
        obj = {
            "model_name":   "Parsimonious OLS — Brent-linked + log_dist_rotterdam (grade-clustered SE)",
            "grades":       result["grades"],
            "n_grades":     len(result["grades"]),
            "metrics": {
                "n_obs":  result["n"],
                "k":      result["k"],
                "r2":     round(result["r2"], 4),
                "r2_adj": round(result["r2_adj"], 4),
                "r2_cv":  round(result["r2_cv"], 4),
                "r2_oot": round(result["r2_oot"], 4) if not np.isnan(result["r2_oot"]) else None,
                "rmse":   round(result["rmse"], 4),
            },
            "features":     result["features"],
            "coefficients": {k: round(v, 6) for k, v in m.params.items()},
            "std_errors":   {k: round(v, 6) for k, v in m.bse.items()},
            "p_values":     {k: round(v, 6) for k, v in m.pvalues.items()},
            "note":         "Bruker grade-clustered SE (cov_type='cluster')",
        }
        NEW_MODEL.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        print(f"  ✓ Lagret: {NEW_MODEL.name}")
    else:
        print(f"\n[6] ✗ Ny modell IKKE entydig bedre — beholder eksisterende v1")

    summary = {
        "log_dist_rotterdam_in_final": "log_dist_rotterdam" in final_features,
        "coef":     float(m.params.get("log_dist_rotterdam", 0)) if "log_dist_rotterdam" in final_features else None,
        "p_value":  float(m.pvalues.get("log_dist_rotterdam", 1)) if "log_dist_rotterdam" in final_features else None,
        "old_metrics": old_model["metrics"],
        "new_metrics": {
            "r2": round(result["r2"], 4),
            "r2_oot": round(result["r2_oot"], 4) if not np.isnan(result["r2_oot"]) else None,
            "rmse": round(result["rmse"], 4),
        },
        "is_better": bool(is_better),
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  ✓ Sammendrag: {RESULTS_JSON.name}")


if __name__ == "__main__":
    main()
