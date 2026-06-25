"""
Script 62 — Finaliser Modell v3: permanent integrasjon av is_fpso.

Etter hypotese-testing i script 58-61 har vi bekreftet at is_fpso er den
sterkeste enkelt-forbedringen (p<0.001, ΔOOT R² +0.018, fikser Norske Sjø-bias).

Dette script:
  1. Legger is_fpso PERMANENT til regression_panel.csv
  2. Re-trener Brent-modellen med is_fpso i initial features
  3. Bruker GRADE-CLUSTERED standard errors (samme som v2)
  4. Lagrer som 34b_brent_model.json (erstatter forrige v2)
  5. Bekrefter at downstream-scripts (42, 43, 49, 55) må oppdateres
     for å sette is_fpso korrekt per felt
"""

from pathlib import Path
import json
import warnings
import shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
PANEL_CSV    = PROC_DIR / "regression_panel.csv"
MODEL_JSON   = PROC_DIR / "34b_brent_model.json"
BACKUP_JSON  = PROC_DIR / "34b_brent_model_v2_backup.json"

WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}
ALWAYS_KEEP   = {"api_gravity", "sulfur_pct"}
ALWAYS_DROP   = {"vix"}
SIG_THRESHOLD = 0.10

# Grades som lastes via FPSO (samme set som brukt i script 60/61)
# Kilde: Sodir + operatør-info
FPSO_GRADES = {
    # NCS FPSOs
    "Alvheim", "Asgard", "Balder", "Goliat", "Norne", "Skarv", "Heidrun",
    "Martin Linge", "Knarr", "Jotun", "Gina Krog", "Njord", "Draugen",
    # West Africa FPSOs
    "Bonny Light", "Forcados", "Cabinda", "Rabi Light", "Qua Iboe",
}
# Pipeline-grades: alle andre i Brent-panelet
# NCS pipeline: Ekofisk, Statfjord, Gullfaks, Troll, Oseberg, Grane,
#               Gudrun, Johan Sverdrup
# Midtøsten: Arab Light/Medium/Extra Light, Basrah Light, Dubai Fateh
# Nord-Afrika: Saharan Blend

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


def add_is_fpso_to_panel():
    """Legg is_fpso permanent til regression_panel.csv."""
    print("\n[1] Legger is_fpso permanent til regression_panel.csv...")
    df = pd.read_csv(PANEL_CSV)
    df["is_fpso"] = df["grade"].isin(FPSO_GRADES).astype(int)

    # Bekreft fordelingen
    fpso_count = df.groupby("grade")["is_fpso"].first().sum()
    total_grades = df["grade"].nunique()
    print(f"  ✓ is_fpso = 1 for {int(fpso_count)} av {total_grades} grades")

    # Liste FPSO-grades for verifisering
    fpso_in_panel = sorted(df[df["is_fpso"] == 1]["grade"].unique())
    print(f"  FPSO-grades i panel: {', '.join(fpso_in_panel)}")

    df.to_csv(PANEL_CSV, index=False)
    print(f"  ✓ Lagret oppdatert panel ({len(df):,} obs)")
    return df


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


def build_initial_features_with_fpso(df: pd.DataFrame) -> list[str]:
    """Brent-modell initial features + is_fpso."""
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
        + ["d_distance_long", "is_fpso"]   # ⭐ NY: is_fpso
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


def fit_clustered(df, features):
    sub = df.dropna(subset=features + ["differential", "grade"]).copy()
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
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
        result = fit_clustered(df, features)
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


def main():
    print("=" * 75)
    print("  SCRIPT 62: Finaliser Modell v3 (med is_fpso permanent)")
    print("=" * 75)

    # Backup nåværende v2-modell
    if MODEL_JSON.exists():
        shutil.copy(MODEL_JSON, BACKUP_JSON)
        print(f"  ✓ Backup av v2: {BACKUP_JSON.name}")

    # Steg 1: Legg is_fpso permanent til panel
    add_is_fpso_to_panel()

    # Steg 2: Re-tren med is_fpso i initial features
    print(f"\n[2] Re-trener Brent-modell med is_fpso (clustered SE)...")
    df = load_panel_brent()
    feats = build_initial_features_with_fpso(df)
    print(f"  Panel: {len(df)} obs, {df['grade'].nunique()} grades, "
          f"{len(feats)} kandidat-features")
    result = stepwise(df, feats)

    # Steg 3: Sjekk at is_fpso ble beholdt
    m = result["model"]
    print(f"\n[3] STATUS for is_fpso i final modell:")
    if "is_fpso" in result["features"]:
        coef = m.params["is_fpso"]
        se   = m.bse["is_fpso"]
        pval = m.pvalues["is_fpso"]
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "."
        print(f"  ✓ BEHOLDT: coef={coef:+.4f}  SE={se:.4f}  p={pval:.4f} {sig}")
        print(f"  Tolkning: FPSO-grades handler {coef:+.2f} USD/bbl mot pipeline-grades")
    else:
        print(f"  ⚠ ELIMINERT — overraskende, undersøk")

    # Steg 4: Lagre som ny hovedmodell
    print(f"\n[4] Lagrer som hovedmodell {MODEL_JSON.name}...")
    obj = {
        "model_name":   "Parsimonious OLS — Brent-linked + is_fpso (v3, clustered SE)",
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
        "note":         (
            "v3: Grade-clustered SE + is_fpso. "
            "FPSO-flag basert på Sodir + operatør-info."
        ),
        "fpso_grades":  sorted(list(FPSO_GRADES)),
    }
    MODEL_JSON.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  ✓ Lagret som ny v3")

    # Steg 5: Forteller om nedstrøms-impact
    print(f"\n{'=' * 75}")
    print(f"  NEDSTRØMS-SCRIPTS som må oppdateres")
    print(f"{'=' * 75}")
    print(f"  Disse scripts bygger feature-dict for prediksjoner og MÅ legge til")
    print(f"  is_fpso for hvert felt de håndterer:")
    print(f"")
    print(f"  · scripts/42_akrbp_realized_price_decomposition.py")
    print(f"    → build_field_features(): legg til \"is_fpso\" basert på AKRBP-felt")
    print(f"      AKRBP FPSO-felt: ALVHEIM, BØYLA, SKOGUL, SKARV")
    print(f"      AKRBP pipeline-felt: VALHALL, HOD, ULA, TAMBAR, TAMBAR ØST,")
    print(f"                            EDVARD GRIEG, IVAR AASEN, JOHAN SVERDRUP")
    print(f"")
    print(f"  · scripts/43_akrbp_forward_prediction.py")
    print(f"    → predict_differential(): samme som over")
    print(f"")
    print(f"  · scripts/49_interactive_ncs_map.py")
    print(f"    → build_field_features(): bruk FPSO_GRADES-listen fra script 62")
    print(f"")
    print(f"  · scripts/55_combined_forward_forecast.py")
    print(f"    → predict_grade_differential(): legg til \"is_fpso\" per grade")


if __name__ == "__main__":
    main()
