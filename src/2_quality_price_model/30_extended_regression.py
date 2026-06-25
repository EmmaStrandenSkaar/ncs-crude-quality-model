"""
Utvidet regresjonsmodell med assay-data + crack spreads + sanksjons-dummies.

Bygger inkrementelt:
  M1: Baseline (API, svovel)
  M2: + API² sweet-spot
  M3: + region
  M4: + assay (vacuum residue, middle distillate, CCR, metaller)
  M5: + markedskontroller (Brent, WTI-Brent, VIX)
  M6: + crack spreads (gasoline, diesel, jet)
  M7: + interaksjoner kvalitet × marked
  M8: + hendelsesdummies (sanksjoner, COVID, OPEC)
  M9: Full + region-spesifikke effekter

Cross-validation, feature importance, og prediksjonstabell.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "regression_panel.csv")
    df["date"] = pd.to_datetime(df["date_str"])

    region_simple = {
        "North Sea": "NorthSea", "Norwegian Sea": "NorthSea", "Barents Sea": "NorthSea",
        "North America": "NorthAmerica", "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm", "Middle East": "MiddleEast",
        "West Africa": "WestAfrica", "North Africa": "NorthAfrica",
        "FSU": "FSU", "Asia-Pacific": "AsiaPac", "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")
    return df


def fit_model(df: pd.DataFrame, features: list[str], name: str) -> dict:
    sub = df.dropna(subset=features + ["differential"]).copy()
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
    m = sm.OLS(y, X).fit(cov_type="HC1")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LinearRegression(), sub[features].values, y.values, cv=kf, scoring="r2"
    )

    return {
        "name": name,
        "model": m,
        "n_obs": len(sub),
        "n_grades": sub["grade"].nunique(),
        "r2": m.rsquared,
        "r2_adj": m.rsquared_adj,
        "r2_cv": cv_scores.mean(),
        "r2_cv_std": cv_scores.std(),
        "rmse": np.sqrt(mean_squared_error(y, m.fittedvalues)),
        "mae": mean_absolute_error(y, m.fittedvalues),
        "aic": m.aic,
        "features": features,
        "data": sub,
    }


def sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."


def main() -> None:
    df = load_panel()
    print(f"Panel: {len(df):,} obs, {df['grade'].nunique()} grades, {df.shape[1]} kolonner\n")

    # Region-dummies
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg", drop_first=True, dtype=int)
    df = pd.concat([df, region_dums], axis=1)
    region_cols = list(region_dums.columns)

    # Feature-grupper
    f_baseline = ["api_gravity", "sulfur_pct"]
    f_quad = f_baseline + ["api2"]
    f_assay = ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni",
               "nitrogen_ppm"]
    f_market = ["brent_price", "wti_brent_spread", "vix"]
    f_cracks = ["gasoline_crack_brent", "diesel_crack_brent", "jet_crack_brent",
                "diesel_minus_gasoline_crack", "brent_dubai_spread"]
    f_interact = ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
                  "middle_dist_x_diesel_crack", "vacuum_resid_x_diesel_crack",
                  "naphtha_x_gasoline_crack"]
    f_events = ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
                "d_venezuela_sanctions", "d_us_shale_boom", "d_covid",
                "d_opec_plus_cuts_2023", "russia_sanctions_x_russian"]

    # Modell-spesifikasjoner
    specs = [
        ("M1: Baseline (API, S)", f_baseline),
        ("M2: + API²", f_quad),
        ("M3: + region", f_quad + region_cols),
        ("M4: + assay (yields, CCR, metaller)", f_quad + region_cols + f_assay),
        ("M5: + marked", f_quad + region_cols + f_assay + f_market),
        ("M6: + crack spreads", f_quad + region_cols + f_assay + f_market + f_cracks),
        ("M7: + interaksjoner", f_quad + region_cols + f_assay + f_market + f_cracks + f_interact),
        ("M8: + hendelser", f_quad + region_cols + f_assay + f_market + f_cracks + f_interact + f_events),
    ]

    print(f"{'='*100}")
    print("MODELLSAMMENLIGNING — inkrementell tilføyelse av features")
    print(f"{'='*100}")

    results = []
    for name, feats in specs:
        r = fit_model(df, feats, name)
        results.append(r)
        print(f"  {name:45s}  N={r['n_obs']:4d}  k={len(r['features']):2d}  "
              f"R²={r['r2']:.4f}  CV={r['r2_cv']:.4f}  RMSE={r['rmse']:.2f}")

    # === Beste modell ===
    best = max(results, key=lambda r: r["r2_cv"])
    m = best["model"]
    data = best["data"].copy()

    print(f"\n{'='*100}")
    print(f"BESTE MODELL (etter CV R²): {best['name']}")
    print(f"  R² = {best['r2']:.4f}, R²adj = {best['r2_adj']:.4f}")
    print(f"  R²_CV (5-fold) = {best['r2_cv']:.4f} ± {best['r2_cv_std']:.4f}")
    print(f"  RMSE = {best['rmse']:.2f} USD/fat, MAE = {best['mae']:.2f} USD/fat")
    print(f"  N = {best['n_obs']:,}, grades = {best['n_grades']}, features = {len(best['features'])}")
    print(f"{'='*100}")

    # Koeffisient-tabell gruppert
    print("\nKOEFFISIENTER:")
    groups = {
        "Kjernekvalitet": ["api_gravity", "api2", "sulfur_pct"],
        "Region": [c for c in best["features"] if c.startswith("reg_")],
        "Assay (refining-økonomi)": f_assay,
        "Marked": f_market,
        "Crack spreads": f_cracks,
        "Interaksjoner": f_interact,
        "Hendelser": f_events,
    }

    for group_name, group_feats in groups.items():
        present = [f for f in group_feats if f in m.params.index]
        if not present:
            continue
        print(f"\n  --- {group_name} ---")
        for feat in present:
            coef = m.params[feat]
            se = m.bse[feat]
            p = m.pvalues[feat]
            print(f"    {feat:35s}: {coef:+9.4f}  (SE {se:.4f}, p {p:.4f}) {sig(p)}")

    # === Sweet spot ===
    if "api_gravity" in m.params and "api2" in m.params:
        opt = -m.params["api_gravity"] / (2 * m.params["api2"])
        print(f"\n  Optimal API-grad (sweet spot): {opt:.1f}°")

    # === Residualer per grade ===
    X_best = sm.add_constant(data[best["features"]].astype(float))
    data["predicted"] = m.predict(X_best)
    data["residual"] = data["differential"] - data["predicted"]

    grade_summary = data.groupby("grade").agg(
        n=("differential", "size"),
        api=("api_gravity", "first"),
        sulfur=("sulfur_pct", "first"),
        vac_resid=("vacuum_resid_pct", "first"),
        ccr=("ccr_wt_pct", "first"),
        region=("region_simple", "first"),
        mean_actual=("differential", "mean"),
        mean_pred=("predicted", "mean"),
        resid=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
    ).sort_values("resid", ascending=False)

    print(f"\n{'='*100}")
    print("RESIDUALER PER GRADE (positiv = handles dyrere enn modellen forventer)")
    print(f"{'='*100}")
    print(grade_summary.to_string(float_format=lambda x: f"{x:+.2f}"))

    # === Lagre ===
    out_resid = OUT_DIR / "30_extended_residuals.csv"
    data[["grade", "date_str", "differential", "predicted", "residual",
          "api_gravity", "sulfur_pct", "vacuum_resid_pct", "ccr_wt_pct",
          "brent_price", "region_simple"]].to_csv(out_resid, index=False)
    print(f"\nResidualer lagret: {out_resid}")

    # Eksport av modellkoeffisienter
    model_export = {
        "model_name": best["name"],
        "r2": round(best["r2"], 4),
        "r2_adj": round(best["r2_adj"], 4),
        "r2_cv": round(best["r2_cv"], 4),
        "r2_cv_std": round(best["r2_cv_std"], 4),
        "rmse": round(best["rmse"], 2),
        "mae": round(best["mae"], 2),
        "n_obs": int(best["n_obs"]),
        "n_grades": int(data["grade"].nunique()),
        "n_features": len(best["features"]),
        "coefficients": {k: round(float(v), 6) for k, v in m.params.items()},
        "p_values": {k: round(float(v), 6) for k, v in m.pvalues.items()},
        "features": best["features"],
    }
    out_json = OUT_DIR / "30_model_coefficients.json"
    with open(out_json, "w") as f:
        json.dump(model_export, f, indent=2)
    print(f"Modellkoeffisienter lagret: {out_json}")

    # === PLOTT ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # 1. Predikert vs faktisk
    ax = axes[0, 0]
    region_colors = {
        "NorthSea": "steelblue", "NorthAmerica": "coral",
        "MiddleEast": "gold", "WestAfrica": "green",
        "LatAm": "purple", "NorthAfrica": "orange",
        "FSU": "brown", "AsiaPac": "pink",
    }
    for region in sorted(data["region_simple"].unique()):
        sub = data[data["region_simple"] == region]
        ax.scatter(sub["predicted"], sub["differential"], alpha=0.25, s=10,
                   color=region_colors.get(region, "gray"), label=region)
    lim = max(abs(data["predicted"]).max(), abs(data["differential"]).max()) + 2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predikert (USD/fat)")
    ax.set_ylabel("Faktisk (USD/fat)")
    ax.set_title(f"Predikert vs faktisk\nR²={best['r2']:.3f}, CV={best['r2_cv']:.3f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # 2. R² inkrementelt
    ax = axes[0, 1]
    model_short = [r["name"][:20] for r in results]
    r2_is = [r["r2"] for r in results]
    r2_cv = [r["r2_cv"] for r in results]
    x_pos = np.arange(len(results))
    ax.bar(x_pos - 0.18, r2_is, 0.35, label="R² in-sample", color="steelblue")
    ax.bar(x_pos + 0.18, r2_cv, 0.35, label="R² CV", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"M{i+1}" for i in range(len(results))], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Inkrementell forklaringskraft")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(r2_is, r2_cv)):
        ax.text(i - 0.18, a + 0.005, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.18, b + 0.005, f"{b:.2f}", ha="center", fontsize=7)

    # 3. Feature importance (standardiserte koeffisienter, topp 20)
    ax = axes[0, 2]
    feat_cols = [f for f in best["features"] if not f.startswith("reg_") and not f.startswith("d_")]
    coefs = pd.Series({f: m.params.get(f, 0) for f in feat_cols})
    stds = data[feat_cols].std()
    std_coefs = (coefs * stds).sort_values()
    top = pd.concat([std_coefs.head(10), std_coefs.tail(10)])
    pvals = [m.pvalues.get(f, 1) for f in top.index]
    colors_b = ["steelblue" if p < 0.05 else "lightgray" for p in pvals]
    ax.barh(top.index, top.values, color=colors_b)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardisert koeffisient")
    ax.set_title("Top 20 features (etter |effekt|)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

    # 4. Vacuum residue effekt (refining-økonomi)
    ax = axes[1, 0]
    if "vacuum_resid_pct" in m.params and "vacuum_resid_x_brent" in m.params:
        vr_range = np.linspace(5, 55, 100)
        for bp in [40, 60, 80, 100]:
            effect = m.params["vacuum_resid_pct"] * vr_range + m.params["vacuum_resid_x_brent"] * vr_range * bp
            ax.plot(vr_range, effect, label=f"Brent=${bp}")
        ax.set_xlabel("Vakuumresidue (% av fat)")
        ax.set_ylabel("Effekt på differensial (USD/fat)")
        ax.set_title("Vakuumresidue-rabatt øker med Brent-pris")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

    # 5. Svovel-effekt (med interaksjon)
    ax = axes[1, 1]
    s_range = np.linspace(0, 4, 100)
    for bp in [40, 60, 80, 100]:
        s_coef = m.params.get("sulfur_pct", 0)
        sb_coef = m.params.get("sulfur_x_brent", 0)
        effect = s_coef * s_range + sb_coef * s_range * bp
        ax.plot(s_range, effect, label=f"Brent=${bp}")
    ax.set_xlabel("Svovel (%)")
    ax.set_ylabel("Effekt på differensial (USD/fat)")
    ax.set_title("Svovel-rabatt × Brent-pris")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    # 6. Residualer per grade
    ax = axes[1, 2]
    gs = grade_summary.sort_values("resid")
    nor_grades_list = ["Johan Sverdrup", "Troll", "Ekofisk", "Oseberg", "Alvheim",
                       "Gullfaks", "Statfjord", "Heidrun", "Grane", "Asgard"]
    colors_r = ["steelblue" if g in nor_grades_list else "lightgray" for g in gs.index]
    ax.barh(gs.index, gs["resid"], color=colors_r, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean residual (USD/fat)")
    ax.set_title("Residualer per grade\n(blå = norske felt)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Utvidet regresjonsmodell: kvalitet + assay + marked + crack spreads + hendelser\n"
                 f"N={best['n_obs']:,}, grades={best['n_grades']}, R²={best['r2']:.3f}, CV={best['r2_cv']:.3f}, "
                 f"RMSE={best['rmse']:.2f} USD/fat",
                 fontsize=13)
    fig.tight_layout()
    out_png = OUT_DIR / "30_extended_regression.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nHovedplott lagret: {out_png}")

    # === Tidsserier ===
    fig2, axes2 = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
    highlight = ["Johan Sverdrup", "Troll", "Ekofisk", "Alvheim",
                 "Maya", "Arab Light", "WTI", "Bonny Light"]
    highlight = [g for g in highlight if g in data["grade"].unique()]
    for i, grade in enumerate(highlight[:8]):
        ax = axes2[i // 2, i % 2]
        sub = data[data["grade"] == grade].sort_values("date_str")
        sub["date_dt"] = pd.to_datetime(sub["date_str"])
        ax.plot(sub["date_dt"], sub["differential"], label="Faktisk", color="steelblue", linewidth=1.2)
        ax.plot(sub["date_dt"], sub["predicted"], label="Predikert", color="coral", linestyle="--", linewidth=1.2)
        ax.fill_between(sub["date_dt"], sub["differential"], sub["predicted"], alpha=0.1, color="gray")
        ax.axhline(0, color="gray", linewidth=0.5)
        rmse = np.sqrt((sub["residual"]**2).mean())
        ax.set_title(f"{grade} (API={sub['api_gravity'].iloc[0]:.0f}°, S={sub['sulfur_pct'].iloc[0]:.1f}%, "
                     f"VR={sub['vacuum_resid_pct'].iloc[0]:.0f}%, RMSE={rmse:.1f})")
        ax.set_ylabel("Diff. (USD/fat)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Utvidet modell — tidsserier med vakuumresidue og assay-info", fontsize=13)
    fig2.tight_layout()
    out_png2 = OUT_DIR / "30_extended_timeseries.png"
    fig2.savefig(out_png2, dpi=150)
    print(f"Tidsserier lagret: {out_png2}")


if __name__ == "__main__":
    main()
