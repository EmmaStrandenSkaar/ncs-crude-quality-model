"""
Script 61 — Utvidet hypotese-testing for å fullføre modell-forbedringen.

ETTER SCRIPT 60 vet vi at is_fpso er den sterkeste enkelt-forbedringen.
Men før vi konkluderer, tester vi flere hypoteser parallelt:

  H4: SEA-AREA SPLIT — er Norske Sjø spesifikt forskjellig fra Nordsjø?
       (i nåværende modell er begge i samme "NorthSea"-dummy)
  H5: is_equinor — operatør-marketing-kraft hypotese
  H6: production_decline_12m — fallende felt mister premium
  H7: d_post_2022 — regime-skift (Red Sea, US Gulf-eksport, etc.)
  H8: small_field_dummy (<30 kbpd) — likviditets-rabatt for små grades

Alle testes ENKELTVIS og deretter i KOMBINASJON med is_fpso.

OUTPUT:
  data/processed/61_extended_hypothesis_results.json
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
RESULTS_JSON = PROC_DIR / "61_extended_hypothesis_results.json"

WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}

# ── Operatør per grade (fra Sodir + manuell mapping for NCS) ───────────────
OPERATOR_BY_GRADE = {
    "Johan Sverdrup": "Equinor", "Statfjord": "Equinor", "Gullfaks": "Equinor",
    "Oseberg": "Equinor", "Heidrun": "Equinor", "Asgard": "Equinor",
    "Norne": "Equinor", "Troll": "Equinor", "Grane": "Equinor",
    "Gudrun": "Equinor", "Gina Krog": "Equinor", "Martin Linge": "Equinor",
    "Knarr": "Equinor", "Njord": "Equinor",
    "Alvheim": "Aker BP", "Skarv": "Aker BP",
    "Ekofisk": "ConocoPhillips",
    "Goliat": "Var Energi", "Balder": "Var Energi", "Jotun": "Var Energi",
    "Draugen": "OKEA",
    # Internasjonale operatører (forenkling — bruk "Other")
    "Bonny Light": "Shell", "Forcados": "Shell",
    "Cabinda": "Chevron", "Qua Iboe": "ExxonMobil",
    "Rabi Light": "Shell", "Saharan Blend": "Sonatrach",
    "Arab Light": "Aramco", "Arab Medium": "Aramco",
    "Arab Extra Light": "Aramco", "Basrah Light": "SOMO",
    "Dubai Fateh": "ADNOC",
}

# FPSO-set fra script 60
FPSO_GRADES = {
    "Alvheim", "Asgard", "Balder", "Goliat", "Norne", "Skarv", "Heidrun",
    "Martin Linge", "Knarr", "Jotun", "Gina Krog", "Njord", "Draugen",
    "Bonny Light", "Forcados", "Cabinda", "Rabi Light", "Qua Iboe",
}


def load_panel_with_features() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV)
    df["date"] = pd.to_datetime(df["date_str"])
    df = df[~df["grade"].isin(WTI_LINKED)].copy()

    # region_simple (samme som tidligere)
    region_simple = {
        "North Sea": "NorthSea", "Norwegian Sea": "NorthSea", "Barents Sea": "NorthSea",
        "North America": "NorthAmerica", "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm", "Middle East": "MiddleEast",
        "West Africa": "WestAfrica", "North Africa": "NorthAfrica",
        "FSU": "FSU", "Asia-Pacific": "AsiaPac", "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg",
                                  drop_first=True, dtype=int)
    for c in region_dums.columns:
        if c not in df.columns:
            df[c] = region_dums[c].values

    # ── NYE features for hypotese-testing ─────────────────────────────────
    # H4: Sea-area split (skiller Norwegian Sea + Barents fra North Sea proper)
    df["is_norwegian_sea"] = (df["region"] == "Norwegian Sea").astype(int)
    df["is_barents_sea"]   = (df["region"] == "Barents Sea").astype(int)

    # H5: is_equinor
    df["is_equinor"] = (df["grade"].map(OPERATOR_BY_GRADE) == "Equinor").astype(int)

    # H6: production_decline_12m (12-mnd glidende endring i produksjon)
    if "production_kbpd" in df.columns:
        df = df.sort_values(["grade", "date"]).copy()
        df["production_decline_12m"] = (
            df.groupby("grade")["production_kbpd"]
              .transform(lambda x: x.pct_change(periods=12).fillna(0))
        )
        # Clip for å unngå outliers (e.g. nye felt med stor relativ vekst)
        df["production_decline_12m"] = df["production_decline_12m"].clip(-0.5, 0.5)

    # H7: d_post_2022 (regime-skift)
    df["d_post_2022"] = (df["date"] >= "2022-12-01").astype(int)

    # H8: small_field_dummy (gjennomsnittlig prod < 30 kbpd)
    avg_prod_per_grade = df.groupby("grade")["production_kbpd"].mean()
    small_grades = set(avg_prod_per_grade[avg_prod_per_grade < 30].index)
    df["is_small_field"] = df["grade"].isin(small_grades).astype(int)

    # Også behold is_fpso fra script 60
    df["is_fpso"] = df["grade"].isin(FPSO_GRADES).astype(int)

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


def test_addition(df, base_features, new_feats):
    if isinstance(new_feats, str):
        new_feats = [new_feats]
    for f in new_feats:
        if f not in df.columns:
            return {"error": f"{f} mangler"}
    features = base_features + new_feats
    return fit_clustered(df, features)


def fmt_sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else " "


def main():
    print("=" * 80)
    print("  SCRIPT 61: Utvidet hypotese-testing (etter is_fpso fra script 60)")
    print("=" * 80)

    df = load_panel_with_features()
    print(f"\n  Panel: {len(df)} obs, {df['grade'].nunique()} grades")

    # Verifiser nye features
    print(f"\n  Nye features:")
    feats_check = {
        "is_norwegian_sea":      df.groupby("grade")["is_norwegian_sea"].first().sum(),
        "is_barents_sea":        df.groupby("grade")["is_barents_sea"].first().sum(),
        "is_equinor":            df.groupby("grade")["is_equinor"].first().sum(),
        "is_small_field":        df.groupby("grade")["is_small_field"].first().sum(),
        "is_fpso":               df.groupby("grade")["is_fpso"].first().sum(),
    }
    for f, n in feats_check.items():
        print(f"    {f:<22} treffer {int(n):>2} grades")

    # ── Base-modell ─────────────────────────────────────────────────────────
    model = json.loads(MODEL_JSON.read_text())
    base_features = model["features"]
    base = fit_clustered(df, base_features)
    print(f"\n[1] BASE-modell (current 34b_brent_model.json): "
          f"k={base['k']}, OOT={base['r2_oot']:.4f}, RMSE={base['rmse']:.2f}")

    # ── H1-H8: Enkelt-test ──────────────────────────────────────────────────
    print(f"\n[2] ENKELT-TEST av hver hypotese:")
    print(f"  {'Hypotese':<26} {'Coef':>9} {'p-verdi':>9} {'ΔOOT':>8} {'ΔRMSE':>8}")
    print(f"  {'-' * 70}")

    hypotheses = [
        ("is_fpso (script 60)",            "is_fpso"),
        ("is_norwegian_sea",               "is_norwegian_sea"),
        ("is_barents_sea",                 "is_barents_sea"),
        ("is_equinor",                     "is_equinor"),
        ("production_decline_12m",         "production_decline_12m"),
        ("d_post_2022 (regime-skift)",     "d_post_2022"),
        ("is_small_field",                 "is_small_field"),
    ]
    single_results = {}
    for label, feat in hypotheses:
        r = test_addition(df, base_features, feat)
        if "error" in r:
            print(f"  ⚠ {label:<26} {r['error']}")
            continue
        m = r["model"]
        coef = m.params.get(feat, 0)
        pval = m.pvalues.get(feat, 1)
        d_oot = r["r2_oot"] - base["r2_oot"]
        d_rmse = r["rmse"] - base["rmse"]
        sig = fmt_sig(pval)
        print(f"  {label:<26} {coef:>+9.4f} {pval:>7.3f}{sig:<2} {d_oot:>+8.4f} {d_rmse:>+8.3f}")
        single_results[feat] = {
            "label":   label,
            "coef":    float(coef),
            "p_value": float(pval),
            "d_oot":   float(d_oot),
            "d_rmse":  float(d_rmse),
            "r2_oot":  float(r["r2_oot"]) if not np.isnan(r["r2_oot"]) else None,
        }

    # ── Beste kombinasjon ──────────────────────────────────────────────────
    print(f"\n[3] BESTE KOMBINASJON (alle signifikante enkelt-vinnere):")
    sig_features = [f for f, v in single_results.items()
                     if v["p_value"] < 0.10 and v["d_oot"] > 0.005]
    print(f"  Velger features med p < 0.10 OG ΔOOT > 0.005:")
    for f in sig_features:
        print(f"    · {f} (p={single_results[f]['p_value']:.3f}, ΔOOT={single_results[f]['d_oot']:+.4f})")

    if sig_features:
        combo = test_addition(df, base_features, sig_features)
        m_combo = combo["model"]
        print(f"\n  Kombinert modell: k={combo['k']}, "
              f"OOT={combo['r2_oot']:.4f} (Δ={combo['r2_oot']-base['r2_oot']:+.4f}), "
              f"RMSE={combo['rmse']:.2f}")
        print(f"  Koeffisienter i kombinert modell (clustered SE):")
        for f in sig_features:
            coef = m_combo.params.get(f, 0)
            pval = m_combo.pvalues.get(f, 1)
            sig = fmt_sig(pval)
            print(f"    {f:<26} coef={coef:+.4f}  p={pval:.3f} {sig}")

    # ── Per-grade bias-sammenligning ─────────────────────────────────────────
    print(f"\n[4] PER-GRADE BIAS — base vs. beste kombinasjon:")
    problem = ["Martin Linge", "Asgard", "Draugen", "Norne", "Statfjord",
               "Heidrun", "Ekofisk", "Alvheim", "Goliat", "Grane", "Skarv",
               "Johan Sverdrup", "Bonny Light"]
    base_bias = base["data"].groupby("grade")["resid"].mean()
    if sig_features:
        combo_bias = combo["data"].groupby("grade")["resid"].mean()
        print(f"  {'Grade':<22} {'Base bias':>10} {'Combo bias':>11} {'Forbedring':>11}")
        print(f"  {'-' * 60}")
        for g in problem:
            if g in base_bias.index and g in combo_bias.index:
                ob, nb = base_bias[g], combo_bias[g]
                impr = abs(ob) - abs(nb)
                arrow = "↓" if impr > 0.05 else "↑" if impr < -0.05 else "≈"
                print(f"  {g:<22} {ob:>+10.2f} {nb:>+11.2f} {impr:>+11.2f} {arrow}")

    # ── Lagre ───────────────────────────────────────────────────────────────
    summary = {
        "base":              {"k": base["k"], "r2_oot": base["r2_oot"], "rmse": base["rmse"]},
        "single_results":    single_results,
        "best_features":     sig_features,
        "combined_metrics":  {
            "k": combo["k"],
            "r2_oot": combo["r2_oot"],
            "rmse": combo["rmse"],
            "delta_oot": combo["r2_oot"] - base["r2_oot"],
        } if sig_features else None,
        "combined_coefs": {
            f: {
                "coef": float(m_combo.params.get(f, 0)),
                "p_value": float(m_combo.pvalues.get(f, 1)),
            } for f in sig_features
        } if sig_features else None,
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  ✓ Lagret: {RESULTS_JSON.name}")

    # ── Anbefaling ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  ANBEFALING")
    print(f"{'=' * 80}")
    print(f"\n  Sterkeste signifikante features (p<0.10 + ΔOOT>0.005):")
    if sig_features:
        for f in sig_features:
            sr = single_results[f]
            print(f"    · {f:<26} coef {sr['coef']:+.3f}, p={sr['p_value']:.3f}, ΔOOT {sr['d_oot']:+.4f}")
    else:
        print(f"    Ingen utover is_fpso fra script 60")


if __name__ == "__main__":
    main()
