"""
Script 60 — Test field-karakteristikker for å forklare Norske Sjø-bias.

DIAGNOSTIKK FRA SCRIPT 58 + 59 viste:
  · Norske Sjø-felt (Asgard, Norne, Draugen, Martin Linge) systematisk
    OVER-predikeres med -1 til -2 USD/bbl
  · Distanse alene løser ikke problemet
  · Sannsynlig årsak: kombinasjon av reservoar-alder, små volumer, FPSO-stream

HYPOTESER VI TESTER:
  H1: field_age_years (gamle felt → fallende kvalitet/likviditet)
  H2: log_production_kbpd (lite volum → mindre likvid handel → rabatt)
  H3: is_fpso (FPSO-grades har mer variabel kvalitet enn pipeline)

METODE:
  Legg til hver av disse som NY feature i Brent-modellen, re-tren med
  grade-clustered SE. Sammenlign per-grade bias før/etter.

OUTPUT:
  data/processed/60_field_characteristics_results.json
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
MODEL_JSON   = PROC_DIR / "34b_brent_model.json"
RESULTS_JSON = PROC_DIR / "60_field_characteristics_results.json"

WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}

# ── First-oil-år per grade (kilde: Sodir Factpages) ───────────────────────
FIRST_OIL_YEAR = {
    # NCS — sterkest bias-kilde
    "Statfjord": 1979, "Ekofisk": 1971, "Oseberg": 1988, "Gullfaks": 1986,
    "Troll": 1995, "Heidrun": 1995, "Draugen": 1993, "Norne": 1997,
    "Asgard": 1999, "Balder": 1999, "Grane": 2003, "Skarv": 2013,
    "Alvheim": 2008, "Gudrun": 2014, "Goliat": 2016, "Johan Sverdrup": 2019,
    "Knarr": 2015, "Jotun": 1999, "Gina Krog": 2017, "Martin Linge": 2020,
    "Njord": 1997,
    # Andre Brent-linkede
    "Bonny Light": 1965, "Forcados": 1972, "Cabinda": 1968, "Qua Iboe": 1969,
    "Rabi Light": 1989, "Arab Light": 1948, "Arab Medium": 1960,
    "Arab Extra Light": 1961, "Basrah Light": 1953, "Dubai Fateh": 1969,
    "Saharan Blend": 1956,
}

# ── FPSO vs pipeline (kilde: Sodir + operatør-info) ────────────────────────
FPSO_GRADES = {
    "Alvheim", "Asgard", "Balder", "Goliat", "Norne", "Skarv", "Heidrun",
    "Martin Linge", "Knarr", "Jotun", "Gina Krog", "Njord", "Draugen",
    "Bonny Light", "Forcados", "Cabinda", "Rabi Light", "Qua Iboe",
}
# Pipeline-grades (alle ikke i FPSO_GRADES, men også eksterne MEG/Algerie)
# Norske pipeline-grades: Ekofisk, Statfjord, Gullfaks, Troll, Oseberg, Grane,
#                          Gudrun, Johan Sverdrup


def load_panel_with_features() -> pd.DataFrame:
    """Last panel og legg til field-characteristics-features."""
    df = pd.read_csv(PANEL_CSV)
    df["date"] = pd.to_datetime(df["date_str"])
    df = df[~df["grade"].isin(WTI_LINKED)].copy()

    # Region simple
    region_simple = {
        "North Sea": "NorthSea", "Norwegian Sea": "NorthSea", "Barents Sea": "NorthSea",
        "North America": "NorthAmerica", "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm", "Middle East": "MiddleEast",
        "West Africa": "WestAfrica", "North Africa": "NorthAfrica",
        "FSU": "FSU", "Asia-Pacific": "AsiaPac", "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")

    # NY: field-alder (års-differanse fra first oil)
    df["field_age_years"] = (df["date"].dt.year - df["grade"].map(FIRST_OIL_YEAR)).fillna(20)

    # NY: FPSO-flagg
    df["is_fpso"] = df["grade"].isin(FPSO_GRADES).astype(int)

    # Sikre at log_production finnes
    if "log_production" not in df.columns and "production_kbpd" in df.columns:
        df["log_production"] = np.log1p(df["production_kbpd"])

    # Sett opp region-dummies (samme som ved modell-trening)
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg",
                                  drop_first=True, dtype=int)
    for c in region_dums.columns:
        if c not in df.columns:
            df[c] = region_dums[c].values

    return df


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
    sub["pred"]  = m.fittedvalues
    sub["resid"] = sub["differential"] - sub["pred"]
    return {
        "model": m, "n": len(sub), "k": len(features),
        "r2": m.rsquared, "r2_cv": cv.mean(), "r2_oot": r2_oot, "rmse": rmse,
        "features": features, "data": sub,
    }


def test_single_addition(df: pd.DataFrame, base_features: list, new_feature: str) -> dict:
    """Test om en enkelt ny variabel forbedrer modellen."""
    if new_feature not in df.columns:
        return {"error": f"{new_feature} mangler"}
    features = base_features + [new_feature]
    result = fit_clustered(df, features)
    m = result["model"]
    coef = float(m.params.get(new_feature, 0))
    pval = float(m.pvalues.get(new_feature, 1))
    return {
        "feature":   new_feature,
        "coef":      coef,
        "p_value":   pval,
        "r2_oot":    float(result["r2_oot"]) if not np.isnan(result["r2_oot"]) else None,
        "rmse":      float(result["rmse"]),
        "n":         result["n"],
        "result":    result,
    }


def main():
    print("=" * 75)
    print("  SCRIPT 60: Test field-karakteristikker for Norske Sjø-bias")
    print("=" * 75)

    print("\n[1] Laster panel og legger til nye field-characteristics...")
    df = load_panel_with_features()
    print(f"  Panel: {len(df)} obs, {df['grade'].nunique()} grades")
    print(f"  Nye features lagt til: field_age_years, is_fpso, log_production")

    # Verifiser FPSO og field-age er korrekt fordelt
    print(f"\n  NCS-grades — alder og FPSO-flag:")
    ncs_grades = ["Asgard", "Norne", "Draugen", "Martin Linge", "Heidrun",
                  "Ekofisk", "Statfjord", "Alvheim", "Johan Sverdrup", "Skarv"]
    for g in ncs_grades:
        sub = df[df["grade"] == g]
        if len(sub):
            age = sub["field_age_years"].iloc[-1]
            fpso = "FPSO" if sub["is_fpso"].iloc[0] else "pipeline"
            print(f"    {g:<22} alder {age:.0f} år | {fpso}")

    # ── Base-modell (uten nye features) — referansepunkt ───────────────────
    print("\n[2] Re-tren BASE-modell (Modell B v2 features) som referanse...")
    model = json.loads(MODEL_JSON.read_text())
    base_features = model["features"]
    base_result = fit_clustered(df, base_features)
    print(f"  Base: k={base_result['k']}, R²={base_result['r2']:.4f}, "
          f"OOT={base_result['r2_oot']:.4f}, RMSE={base_result['rmse']:.2f}")

    # ── Test hver kandidat enkeltvis ────────────────────────────────────────
    print(f"\n[3] Tester hver kandidat-variabel ÉN OM GANGEN:")
    print(f"  {'Variabel':<22} {'Coef':>10} {'p-verdi':>9} {'OOT R²':>9} "
          f"{'ΔOOT':>7} {'ΔRMSE':>7}")
    print(f"  {'-' * 72}")

    candidates = ["field_age_years", "log_production", "is_fpso"]
    candidate_results = {}
    for cand in candidates:
        res = test_single_addition(df, base_features, cand)
        if "error" in res:
            print(f"  ⚠ {cand}: {res['error']}")
            continue
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else \
              "*" if res["p_value"] < 0.05 else "." if res["p_value"] < 0.10 else ""
        d_oot = res["r2_oot"] - base_result["r2_oot"] if res["r2_oot"] else 0
        d_rmse = res["rmse"] - base_result["rmse"]
        print(f"  {cand:<22} {res['coef']:>+10.4f} {res['p_value']:>7.3f}{sig:<3} "
              f"{res['r2_oot']:>9.4f} {d_oot:>+7.4f} {d_rmse:>+7.3f}")
        candidate_results[cand] = res

    # ── Test alle 3 kombinert ───────────────────────────────────────────────
    print(f"\n[4] Tester ALLE tre kombinert:")
    combo_features = base_features + ["field_age_years", "log_production", "is_fpso"]
    combo_result = fit_clustered(df, combo_features)
    m = combo_result["model"]
    print(f"  Kombinert: k={combo_result['k']}, OOT={combo_result['r2_oot']:.4f} "
          f"(Δ={combo_result['r2_oot']-base_result['r2_oot']:+.4f})")
    for f in ["field_age_years", "log_production", "is_fpso"]:
        coef = m.params.get(f, 0)
        pval = m.pvalues.get(f, 1)
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else \
               "*" if pval < 0.05 else "." if pval < 0.10 else ""
        print(f"    {f:<22} coef={coef:+8.4f}  p={pval:.3f} {sig}")

    # ── Per-grade bias-sammenligning: base vs. beste kombinasjon ────────────
    print(f"\n[5] PER-GRADE BIAS — base vs. kombinert modell (problem-felt):")
    problem = ["Martin Linge", "Asgard", "Draugen", "Norne", "Statfjord",
               "Heidrun", "Ekofisk", "Alvheim", "Goliat", "Grane"]
    base_bias  = base_result["data"].groupby("grade")["resid"].mean()
    combo_bias = combo_result["data"].groupby("grade")["resid"].mean()
    print(f"  {'Grade':<22} {'Base bias':>10} {'Combo bias':>11} {'Forbedring':>11}")
    print(f"  {'-' * 60}")
    for g in problem:
        if g in base_bias.index and g in combo_bias.index:
            ob, nb = base_bias[g], combo_bias[g]
            impr = abs(ob) - abs(nb)
            arrow = "↓" if impr > 0.05 else "↑" if impr < -0.05 else "≈"
            print(f"  {g:<22} {ob:>+10.2f} {nb:>+11.2f} {impr:>+11.2f} {arrow}")

    # ── Lagre resultater ────────────────────────────────────────────────────
    summary = {
        "base_metrics": {
            "k": base_result["k"], "r2": base_result["r2"],
            "r2_oot": base_result["r2_oot"], "rmse": base_result["rmse"],
        },
        "single_variable_tests": {
            k: {
                "coef":    v["coef"],
                "p_value": v["p_value"],
                "r2_oot":  v["r2_oot"],
                "rmse":    v["rmse"],
                "delta_oot": (v["r2_oot"] - base_result["r2_oot"]) if v["r2_oot"] else None,
            } for k, v in candidate_results.items()
        },
        "combined_test": {
            "k": combo_result["k"],
            "r2_oot": combo_result["r2_oot"],
            "rmse": combo_result["rmse"],
            "delta_oot": combo_result["r2_oot"] - base_result["r2_oot"],
        },
        "best_per_grade_improvement": {
            g: {
                "base_bias":  float(base_bias[g]),
                "combo_bias": float(combo_bias[g]),
                "improvement_usd_bbl": float(abs(base_bias[g]) - abs(combo_bias[g])),
            } for g in problem
            if g in base_bias.index and g in combo_bias.index
        },
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  ✓ Lagret: {RESULTS_JSON.name}")

    # ── Anbefaling ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  ANBEFALING")
    print(f"{'=' * 75}")
    best = max(candidate_results.items(),
                key=lambda x: x[1]["r2_oot"] if x[1]["r2_oot"] else 0)
    print(f"\n  Sterkeste enkelt-variabel: {best[0]}")
    print(f"    OOT R² forbedring: {best[1]['r2_oot'] - base_result['r2_oot']:+.4f}")
    print(f"    Signifikans: p={best[1]['p_value']:.3f}")

    combo_imp = combo_result["r2_oot"] - base_result["r2_oot"]
    print(f"\n  Alle tre kombinert: OOT-forbedring {combo_imp:+.4f}")
    if combo_imp > 0.02:
        print(f"  ✓ Anbefalt: legg til alle tre i hovedmodellen")
    elif combo_imp > 0:
        print(f"  ~ Mindre forbedring — beste enkelt-variabel kan være nok")
    else:
        print(f"  ✗ Ingen forbedring — hold på base-modellen")


if __name__ == "__main__":
    main()
