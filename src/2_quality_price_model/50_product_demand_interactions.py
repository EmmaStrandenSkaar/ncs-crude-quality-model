"""
Script 50 — Yield × Crack-spread interaksjoner: hvordan produkt-etterspørsel
            påvirker prisen på spesifikke crude-grades.

ØKONOMISK INTUISJON:
  Hvert raffineri trekker en yield-fordeling ut av en crude (LPG, naphtha,
  kerosene, diesel, vacuum residue). Når etterspørselen etter et bestemt
  produkt øker, stiger crack-spreaden for det produktet, OG crudes med høy
  yield for det produktet bør tradere på premium.

  Eksempel: økt jet fuel-demand → jet_crack_brent stiger →
            crudes med høy kerosene_pct bør stige mer enn andre.

INTERAKSJONER VI TESTER:
  naphtha_pct        × gasoline_crack_brent   (gasoline-demand)
  kerosene_pct       × jet_crack_brent        (jet fuel-demand)  ⭐ NY
  diesel_gasoil_pct  × diesel_crack_brent     (diesel-demand)
  vacuum_resid_pct   × diesel_crack_brent     (allerede beregnet)
  middle_distillate  × diesel_crack_brent     (allerede beregnet)

MODELL:
  Re-trener Modell B (Brent-linked, 32 grades) med disse interaksjonene
  i kandidat-listen. Stepwise eliminator avgjør hvilke som er signifikante.

OUTPUT:
  data/processed/regression_panel.csv      (oppdatert med ny interaksjon)
  data/processed/34b_brent_model.json      (re-trent modell)
  data/processed/50_yield_crack_results.json (sammendrag)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PANEL_CSV     = PROCESSED_DIR / "regression_panel.csv"

# Sett fra script 47 — synkronisert
WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}

ALWAYS_KEEP   = {"api_gravity", "sulfur_pct"}
ALWAYS_DROP   = {"vix"}
SIG_THRESHOLD = 0.10

# Interaksjoner og deres parents (samme regel som script 47)
INTERACTION_PARENTS = {
    "api2":                       ["api_gravity"],
    "sulfur_x_brent":             ["sulfur_pct", "brent_price"],
    "vacuum_resid_x_brent":       ["vacuum_resid_pct", "brent_price"],
    "ccr_x_brent":                ["ccr_wt_pct", "brent_price"],
    "api_x_contango":             ["api_gravity", "fc_slope_4m"],
    "sulfur_x_refinery_util":     ["sulfur_pct", "us_refinery_util_pct"],
    # NYE yield × crack-interaksjoner:
    "naphtha_x_gasoline_crack":   ["naphtha_pct", "gasoline_crack_brent"],
    "kerosene_x_jet_crack":       ["kerosene_pct", "jet_crack_brent"],
    "diesel_x_diesel_crack":      ["diesel_gasoil_pct", "diesel_crack_brent"],
    "vacuum_resid_x_diesel_crack":["vacuum_resid_pct", "diesel_crack_brent"],
    "middle_dist_x_diesel_crack": ["middle_distillate_pct", "diesel_crack_brent"],
}


# ────────────────────────────────────────────────────────────────────────────
# STEG 1: LEGG TIL MANGLENDE INTERAKSJONER I PANELET
# ────────────────────────────────────────────────────────────────────────────

def add_yield_crack_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn de manglende yield × crack-interaksjonene."""
    print("\n[1] Legger til yield × crack-interaksjoner i panelet...")

    # kerosene × jet crack (NY — denne mangler)
    if "kerosene_x_jet_crack" not in df.columns:
        df["kerosene_x_jet_crack"] = df["kerosene_pct"] * df["jet_crack_brent"]
        print(f"  ✓ Lagt til kerosene_x_jet_crack")
    else:
        print(f"  · kerosene_x_jet_crack finnes allerede")

    # diesel × diesel crack (NY)
    if "diesel_x_diesel_crack" not in df.columns:
        df["diesel_x_diesel_crack"] = df["diesel_gasoil_pct"] * df["diesel_crack_brent"]
        print(f"  ✓ Lagt til diesel_x_diesel_crack")
    else:
        print(f"  · diesel_x_diesel_crack finnes allerede")

    # Eksisterende interaksjoner — verifiser
    for col in ["naphtha_x_gasoline_crack", "vacuum_resid_x_diesel_crack",
                "middle_dist_x_diesel_crack"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"  · {col:<35} finnes ({nn} non-null)")
        else:
            print(f"  ⚠ {col} mangler!")

    return df


# ────────────────────────────────────────────────────────────────────────────
# STEG 2: TREN MODELL B MED NYE INTERAKSJONER
# ────────────────────────────────────────────────────────────────────────────

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


def build_initial_features_brent_with_yields(df: pd.DataFrame) -> list[str]:
    """Som script 47 sin Brent-versjon, pluss yield × crack-interaksjonene."""
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg",
                                  drop_first=True, dtype=int)
    df_ext = pd.concat([df, region_dums], axis=1)
    region_cols = list(region_dums.columns)
    for c in region_cols:
        df[c] = df_ext[c].values

    region_cols_valid = [c for c in region_cols
                          if df[c].sum() > 0 and df[c].sum() < len(df)]

    BRENT_EXCLUDE = {
        "is_landlocked", "is_pipeline_constrained",
        "d_distance_medium", "landlocked_x_cushing_stocks",
        "landlocked_x_contango", "wti_brent_spread",
        "d_venezuela_sanctions",
    }

    initial = (
        ["api_gravity", "sulfur_pct", "api2"]
        + region_cols_valid
        + ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"]
        + ["brent_price", "vix"]
        + ["gasoline_crack_brent", "diesel_crack_brent", "jet_crack_brent",
           "diesel_minus_gasoline_crack", "brent_dubai_spread"]
        + ["d_distance_long"]
        + ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
           "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
           "d_refinery_tight", "d_refinery_slack"]
        + ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"]
        + ["sin_month", "cos_month", "d_winter"]
        + ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
           "api_x_contango", "sulfur_x_refinery_util"]
        # ⭐ NYE: yield × crack-interaksjoner
        + ["naphtha_x_gasoline_crack",
           "kerosene_x_jet_crack",
           "diesel_x_diesel_crack",
           "vacuum_resid_x_diesel_crack",
           "middle_dist_x_diesel_crack"]
        + ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
           "d_us_shale_boom", "d_covid", "d_opec_plus_cuts_2023"]
    )
    return [f for f in initial if f in df.columns and f not in BRENT_EXCLUDE]


def fit_eval(df: pd.DataFrame, features: list[str]) -> dict | None:
    sub = df.dropna(subset=features + ["differential"]).copy()
    if len(sub) < 100:
        return None
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
    m = sm.OLS(y, X).fit(cov_type="HC1")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(LinearRegression(), sub[features].values, y.values,
                          cv=kf, scoring="r2")

    cutoff = sub["date"].max() - pd.DateOffset(months=24)
    train = sub[sub["date"] <= cutoff]
    test  = sub[sub["date"] > cutoff]
    r2_oot = np.nan
    if len(train) > 100 and len(test) > 30:
        lr = LinearRegression().fit(train[features].values, train["differential"].values)
        pred   = lr.predict(test[features].values)
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


def stepwise_eliminate(df: pd.DataFrame, initial_features: list[str]) -> dict:
    features = [f for f in initial_features if f not in ALWAYS_DROP]
    it = 0
    while True:
        it += 1
        result = fit_eval(df, features)
        if result is None:
            return None
        m     = result["model"]
        pvals = m.pvalues.drop("const", errors="ignore")

        protected = set(ALWAYS_KEEP)
        for inter, parents in INTERACTION_PARENTS.items():
            if inter in features and inter in pvals.index and pvals[inter] < SIG_THRESHOLD:
                protected.update(parents)

        candidates = pvals[pvals > SIG_THRESHOLD].sort_values(ascending=False)
        candidates = candidates[~candidates.index.isin(protected)]

        if len(candidates) == 0:
            print(f"  Konvergerte etter {it} iter. k={len(features)}, "
                  f"R²={result['r2']:.4f}, CV={result['r2_cv']:.4f}, "
                  f"OOT={result['r2_oot']:.4f}, RMSE={result['rmse']:.2f}")
            return result

        worst = candidates.index[0]
        features = [f for f in features if f != worst]
        if len(features) < 5:
            return result


def main() -> None:
    print("=" * 75)
    print("  SCRIPT 50: Yield × Crack-spread interaksjoner")
    print("=" * 75)

    # ── Steg 1: Last panel og legg til manglende interaksjoner ─────────────
    df_full = pd.read_csv(PANEL_CSV)
    df_full = add_yield_crack_interactions(df_full)
    df_full.to_csv(PANEL_CSV, index=False)
    print(f"\n  Panel oppdatert med nye kolonner.")

    # ── Steg 2: Re-tren Modell B med yield × crack-interaksjoner ──────────
    print("\n[2] Re-trener Modell B med yield × crack-interaksjoner...")
    df_brent = load_panel_brent()
    feats    = build_initial_features_brent_with_yields(df_brent)
    print(f"  Panel: {len(df_brent)} obs, {df_brent['grade'].nunique()} grades, "
          f"{len(feats)} kandidat-features")

    result = stepwise_eliminate(df_brent, feats)

    # ── Steg 3: Hvilke yield × crack-interaksjoner er signifikante? ────────
    print("\n[3] Yield × crack-interaksjoner i FINAL modell:")
    print("─" * 75)
    yield_crack_cols = [
        "naphtha_x_gasoline_crack",
        "kerosene_x_jet_crack",
        "diesel_x_diesel_crack",
        "vacuum_resid_x_diesel_crack",
        "middle_dist_x_diesel_crack",
    ]
    m = result["model"]
    yield_results = []
    for col in yield_crack_cols:
        in_model = col in result["features"]
        if in_model:
            coef = m.params[col]
            se   = m.bse[col]
            pval = m.pvalues[col]
            sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "."
            print(f"  ✓ {col:<35}  coef={coef:+8.5f}  SE={se:.5f}  p={pval:.4f} {sig}")
        else:
            print(f"  ✗ {col:<35}  (eliminert av stepwise)")
        yield_results.append({
            "feature":   col,
            "in_model":  in_model,
            "coef":      float(m.params[col]) if in_model else None,
            "se":        float(m.bse[col])    if in_model else None,
            "p_value":   float(m.pvalues[col]) if in_model else None,
        })

    # ── Steg 4: Lagre oppdatert modell ─────────────────────────────────────
    print(f"\n[4] Lagrer oppdatert modell...")
    obj = {
        "model_name":   "Parsimonious OLS — Brent-linked + yield × crack (32 grades)",
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
    }
    model_path = PROCESSED_DIR / "34b_brent_model.json"
    model_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  ✓ Oppdatert: {model_path.name}")

    # Sammendrag-fil for downstream scripts
    summary = {
        "yield_crack_interactions": yield_results,
        "n_significant":            sum(1 for r in yield_results if r["in_model"]),
        "model_rmse":               result["rmse"],
        "model_oot_r2":             result["r2_oot"] if not np.isnan(result["r2_oot"]) else None,
    }
    summary_path = PROCESSED_DIR / "50_yield_crack_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  ✓ Sammendrag: {summary_path.name}")

    # ── Steg 5: Tolkning ───────────────────────────────────────────────────
    print(f"\n[5] TOLKNING")
    print("─" * 75)
    sig_count = sum(1 for r in yield_results if r["in_model"])
    if sig_count > 0:
        print(f"  ✓ {sig_count} av 5 yield × crack-interaksjoner er statistisk signifikante.")
        print(f"  Dette betyr: produkt-etterspørsel (målt via crack-spreads) påvirker")
        print(f"  spesifikke crude-grades basert på deres produkt-yield.")
        print(f"  → Neste steg: script 51 bygger per-grade sensitivitetsanalyse.")
    else:
        print(f"  ⚠ Ingen yield × crack-interaksjoner ble signifikante.")
        print(f"  Mulig forklaring: effektene fanges allerede av andre features")
        print(f"  (f.eks. middle_distillate_pct alene fanger diesel-relaterte effekter).")


if __name__ == "__main__":
    main()
