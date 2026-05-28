"""
Full regresjonsmodell — alt på bordet:
  Kvalitet (API, sulfur, vacuum residue, CCR, metaller)
  + Region (dummies)
  + Marked (Brent, spreads, VIX)
  + Crack spreads (gasoline, diesel, jet, 3-2-1)
  + Logistikk (distance-band, landlocked, pipeline-constrained)
  + EIA fundamentals (inventories, refinery utilization, exports)
  + Forward curve (contango/backwardation slope)
  + Sesong (sin/cos for måned)
  + Hendelser (sanksjoner, COVID, OPEC, shale boom)
  + Interaksjoner (kvalitet × marked, logistikk × marked)
  + Year fixed effects (residual tidsvariasjon)

Bygger inkrementelt slik at vi ser hva hver gruppe tilfører.
Robust SE, 5-fold CV, og out-of-time validering på siste 3 år.
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

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR


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


def fit_and_eval(df: pd.DataFrame, features: list[str], name: str) -> dict:
    sub = df.dropna(subset=features + ["differential"]).copy()
    if len(sub) < 100:
        return None
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
    m = sm.OLS(y, X).fit(cov_type="HC1")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LinearRegression(), sub[features].values, y.values, cv=kf, scoring="r2"
    )

    # Out-of-time: holdback siste 24 mnd
    sub["date"] = pd.to_datetime(sub["date_str"])
    cutoff = sub["date"].max() - pd.DateOffset(months=24)
    train = sub[sub["date"] <= cutoff]
    test = sub[sub["date"] > cutoff]
    r2_oot = np.nan
    rmse_oot = np.nan
    if len(train) > 100 and len(test) > 50:
        lr = LinearRegression()
        lr.fit(train[features].values, train["differential"].values)
        pred_oot = lr.predict(test[features].values)
        ss_res = ((test["differential"].values - pred_oot) ** 2).sum()
        ss_tot = ((test["differential"].values - test["differential"].mean()) ** 2).sum()
        r2_oot = 1 - ss_res / ss_tot
        rmse_oot = np.sqrt(((test["differential"].values - pred_oot) ** 2).mean())

    return {
        "name": name,
        "model": m,
        "n_obs": len(sub),
        "n_grades": sub["grade"].nunique(),
        "r2": m.rsquared,
        "r2_adj": m.rsquared_adj,
        "r2_cv": cv_scores.mean(),
        "r2_cv_std": cv_scores.std(),
        "r2_oot": r2_oot,
        "rmse_oot": rmse_oot,
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

    # Year fixed effects
    df["year_str"] = df["year"].astype(int).astype(str)
    year_dums = pd.get_dummies(df["year_str"], prefix="yr", drop_first=True, dtype=int)
    df = pd.concat([df, year_dums], axis=1)
    year_cols = list(year_dums.columns)

    # === Feature-grupper ===
    f_baseline = ["api_gravity", "sulfur_pct", "api2"]
    f_assay = ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"]
    f_market = ["brent_price", "wti_brent_spread", "vix"]
    f_cracks = ["gasoline_crack_brent", "diesel_crack_brent",
                "diesel_minus_gasoline_crack", "brent_dubai_spread"]
    f_logistics = ["d_distance_medium", "d_distance_long",
                   "is_landlocked", "is_pipeline_constrained"]
    f_fundamentals = ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
                      "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
                      "d_refinery_tight", "d_refinery_slack"]
    f_forward = ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"]
    f_seasonal = ["sin_month", "cos_month", "d_winter"]
    f_interactions = ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
                      "landlocked_x_cushing_stocks", "api_x_contango",
                      "sulfur_x_refinery_util", "landlocked_x_contango"]
    f_events = ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
                "d_venezuela_sanctions", "d_us_shale_boom", "d_covid",
                "d_opec_plus_cuts_2023"]

    # Filtrer kolonner som faktisk finnes
    def avail(cols):
        return [c for c in cols if c in df.columns]

    f_assay = avail(f_assay)
    f_market = avail(f_market)
    f_cracks = avail(f_cracks)
    f_logistics = avail(f_logistics)
    f_fundamentals = avail(f_fundamentals)
    f_forward = avail(f_forward)
    f_seasonal = avail(f_seasonal)
    f_interactions = avail(f_interactions)
    f_events = avail(f_events)

    # === Modell-spesifikasjoner (inkrementelt) ===
    specs = [
        ("M1: Baseline (API+S+API²)", f_baseline),
        ("M2: + region", f_baseline + region_cols),
        ("M3: + assay", f_baseline + region_cols + f_assay),
        ("M4: + marked", f_baseline + region_cols + f_assay + f_market),
        ("M5: + crack spreads", f_baseline + region_cols + f_assay + f_market + f_cracks),
        ("M6: + logistikk", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics),
        ("M7: + EIA fundamentals", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics + f_fundamentals),
        ("M8: + forward curve", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics + f_fundamentals + f_forward),
        ("M9: + sesong", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics + f_fundamentals + f_forward + f_seasonal),
        ("M10: + interaksjoner", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics + f_fundamentals + f_forward + f_seasonal + f_interactions),
        ("M11: + hendelser", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics + f_fundamentals + f_forward + f_seasonal + f_interactions + f_events),
        ("M12: + year FE", f_baseline + region_cols + f_assay + f_market + f_cracks + f_logistics + f_fundamentals + f_forward + f_seasonal + f_interactions + f_events + year_cols),
    ]

    print(f"{'='*120}")
    print(f"{'MODELL':<45} {'N':>5} {'k':>4} {'R²':>8} {'CV R²':>8} {'OOT R²':>8} {'RMSE':>7} {'OOT RMSE':>9}")
    print(f"{'='*120}")

    results = []
    for name, feats in specs:
        r = fit_and_eval(df, feats, name)
        if r is None:
            print(f"  {name:45s} SKIPPED (for få obs)")
            continue
        results.append(r)
        print(f"  {name:43s}  {r['n_obs']:5d}  {len(r['features']):3d}  "
              f"{r['r2']:.4f}  {r['r2_cv']:.4f}  "
              f"{r['r2_oot']:7.4f}  {r['rmse']:.2f}   {r['rmse_oot']:7.2f}")

    # === Beste modell (etter CV R²) ===
    best = max(results, key=lambda r: r["r2_cv"])
    m = best["model"]
    data = best["data"].copy()

    print(f"\n{'='*120}")
    print(f"BESTE MODELL (etter CV R²): {best['name']}")
    print(f"  In-sample R²    = {best['r2']:.4f}")
    print(f"  Adjusted R²     = {best['r2_adj']:.4f}")
    print(f"  5-fold CV R²    = {best['r2_cv']:.4f} ± {best['r2_cv_std']:.4f}")
    print(f"  Out-of-time R²  = {best['r2_oot']:.4f} (siste 24 mnd holdback)")
    print(f"  RMSE            = {best['rmse']:.2f} USD/fat")
    print(f"  OOT RMSE        = {best['rmse_oot']:.2f} USD/fat")
    print(f"  MAE             = {best['mae']:.2f} USD/fat")
    print(f"  N obs           = {best['n_obs']:,}")
    print(f"  Grades          = {best['n_grades']}")
    print(f"  Features        = {len(best['features'])}")
    print(f"{'='*120}")

    # === Koeffisient-tabell ===
    print("\nKOEFFISIENTER (gruppert):")
    groups = {
        "Kjernekvalitet": ["api_gravity", "api2", "sulfur_pct"],
        "Region": region_cols,
        "Assay": f_assay,
        "Marked": f_market,
        "Crack spreads": f_cracks,
        "Logistikk": f_logistics,
        "EIA fundamentals": f_fundamentals,
        "Forward curve": f_forward,
        "Sesong": f_seasonal,
        "Interaksjoner": f_interactions,
        "Hendelser": f_events,
        "Year FE": year_cols if best == results[-1] else [],
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
            print(f"    {feat:40s}: {coef:+9.4f}  (SE {se:7.4f}, p {p:.4f}) {sig(p)}")

    if "api_gravity" in m.params and "api2" in m.params:
        opt = -m.params["api_gravity"] / (2 * m.params["api2"])
        print(f"\n  Optimal API-grad (sweet spot): {opt:.1f}°")

    # === Per-grade residualer ===
    X_best = sm.add_constant(data[best["features"]].astype(float))
    data["predicted"] = m.predict(X_best)
    data["residual"] = data["differential"] - data["predicted"]

    grade_summary = data.groupby("grade").agg(
        n=("differential", "size"),
        api=("api_gravity", "first"),
        sulfur=("sulfur_pct", "first"),
        vac_resid=("vacuum_resid_pct", "first"),
        distance=("distance_band", "first"),
        landlocked=("is_landlocked", "first"),
        mean_actual=("differential", "mean"),
        mean_pred=("predicted", "mean"),
        resid=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
    ).sort_values("resid", ascending=False)

    print(f"\n{'='*120}")
    print("RESIDUALER PER GRADE (positiv = handles dyrere enn modellen forventer)")
    print(f"{'='*120}")
    print(grade_summary.to_string(float_format=lambda x: f"{x:+.2f}"))

    # === Lagre ===
    out_resid = OUT_DIR / "33_full_residuals.csv"
    save_cols = ["grade", "date_str", "differential", "predicted", "residual",
                 "api_gravity", "sulfur_pct", "vacuum_resid_pct", "ccr_wt_pct",
                 "brent_price", "region_simple", "distance_band", "is_landlocked"]
    save_cols = [c for c in save_cols if c in data.columns]
    data[save_cols].to_csv(out_resid, index=False)
    print(f"\nResidualer lagret: {out_resid}")

    grade_summary.reset_index().to_csv(OUT_DIR / "33_grade_summary.csv", index=False)

    model_export = {
        "model_name": best["name"],
        "metrics": {
            "r2": round(best["r2"], 4),
            "r2_adj": round(best["r2_adj"], 4),
            "r2_cv": round(best["r2_cv"], 4),
            "r2_cv_std": round(best["r2_cv_std"], 4),
            "r2_oot": round(best["r2_oot"], 4) if best["r2_oot"] is not None else None,
            "rmse": round(best["rmse"], 2),
            "rmse_oot": round(best["rmse_oot"], 2) if best["rmse_oot"] is not None else None,
            "mae": round(best["mae"], 2),
            "n_obs": int(best["n_obs"]),
            "n_grades": int(data["grade"].nunique()),
            "n_features": len(best["features"]),
        },
        "coefficients": {k: round(float(v), 6) for k, v in m.params.items()},
        "p_values": {k: round(float(v), 6) for k, v in m.pvalues.items()},
        "features": best["features"],
    }
    out_json = OUT_DIR / "33_full_model.json"
    with open(out_json, "w") as f:
        json.dump(model_export, f, indent=2)
    print(f"Modellkoeffisienter lagret: {out_json}")

    # === PLOTT ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # 1. Predikert vs faktisk farget på region
    ax = axes[0, 0]
    region_colors = {
        "NorthSea": "steelblue", "NorthAmerica": "coral",
        "MiddleEast": "gold", "WestAfrica": "green",
        "LatAm": "purple", "NorthAfrica": "orange",
        "FSU": "brown", "AsiaPac": "pink",
    }
    for region in sorted(data["region_simple"].unique()):
        sub = data[data["region_simple"] == region]
        ax.scatter(sub["predicted"], sub["differential"], alpha=0.25, s=8,
                   color=region_colors.get(region, "gray"), label=region)
    lim = max(abs(data["predicted"]).max(), abs(data["differential"]).max()) + 2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predikert (USD/fat)")
    ax.set_ylabel("Faktisk (USD/fat)")
    ax.set_title(f"Predikert vs faktisk — R²={best['r2']:.3f}, CV={best['r2_cv']:.3f}, OOT={best['r2_oot']:.3f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # 2. Inkrementell R² (in-sample + CV + OOT)
    ax = axes[0, 1]
    x_pos = np.arange(len(results))
    width = 0.25
    ax.bar(x_pos - width, [r["r2"] for r in results], width, label="In-sample", color="steelblue")
    ax.bar(x_pos, [r["r2_cv"] for r in results], width, label="CV", color="coral")
    ax.bar(x_pos + width, [r["r2_oot"] for r in results], width, label="Out-of-time", color="seagreen")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"M{i+1}" for i in range(len(results))], fontsize=9, rotation=0)
    ax.set_ylabel("R²")
    ax.set_title("Inkrementell forklaringskraft (in-sample, CV, out-of-time)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # 3. Top features (standardiserte koef)
    ax = axes[0, 2]
    skip_prefix = ("reg_", "yr_", "d_")
    feat_cols = [f for f in best["features"]
                 if not any(f.startswith(p) for p in skip_prefix)]
    coefs = pd.Series({f: m.params.get(f, 0) for f in feat_cols})
    stds = data[feat_cols].std()
    std_coefs = (coefs * stds).sort_values()
    top = pd.concat([std_coefs.head(10), std_coefs.tail(10)])
    pvals = [m.pvalues.get(f, 1) for f in top.index]
    colors_b = ["steelblue" if p < 0.05 else "lightgray" for p in pvals]
    ax.barh(top.index, top.values, color=colors_b)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardisert koeffisient")
    ax.set_title("Topp 20 features (etter |effekt|)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

    # 4. RMSE per grade (sortert)
    ax = axes[1, 0]
    gs = grade_summary.sort_values("rmse", ascending=False)
    nor_grades = ["Johan Sverdrup", "Troll", "Ekofisk", "Oseberg", "Alvheim",
                  "Gullfaks", "Statfjord", "Heidrun", "Grane", "Asgard"]
    colors_r = ["steelblue" if g in nor_grades else "lightgray" for g in gs.index]
    ax.barh(gs.index, gs["rmse"], color=colors_r, alpha=0.8)
    ax.set_xlabel("RMSE (USD/fat) — lavere = bedre prediksjon")
    ax.set_title("Prediksjonsfeil per grade (blå = norske)")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="x", alpha=0.3)

    # 5. Vacuum residue × Brent-effekt
    ax = axes[1, 1]
    vr_range = np.linspace(5, 55, 100)
    for bp in [40, 60, 80, 100]:
        effect = m.params.get("vacuum_resid_pct", 0) * vr_range
        if "vacuum_resid_x_brent" in m.params:
            effect += m.params["vacuum_resid_x_brent"] * vr_range * bp
        if "ccr_x_brent" in m.params and "ccr_wt_pct" in m.params:
            # Vacuum residue og CCR korrelerer ~0.95, vis kombinert effekt
            ccr_proxy = vr_range * 0.25  # rough scaling
            effect += m.params["ccr_x_brent"] * ccr_proxy * bp
        ax.plot(vr_range, effect, label=f"Brent=${bp}")
    ax.set_xlabel("Vakuumresidue (% av fat)")
    ax.set_ylabel("Effekt på differensial (USD/fat)")
    ax.set_title("Vakuumresidue-rabatt ved ulike Brent-priser")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    # 6. Residualer per grade
    ax = axes[1, 2]
    gs_resid = grade_summary.sort_values("resid")
    colors_r2 = ["steelblue" if g in nor_grades else "lightgray" for g in gs_resid.index]
    ax.barh(gs_resid.index, gs_resid["resid"], color=colors_r2, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean residual (USD/fat)")
    ax.set_title("Residualer per grade\n(blå = norske)")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"FULL modell: kvalitet+assay+marked+cracks+logistikk+EIA+FC+sesong+hendelser+yearFE\n"
                 f"N={best['n_obs']:,}, grades={best['n_grades']}, features={len(best['features'])}, "
                 f"R²={best['r2']:.3f}, CV={best['r2_cv']:.3f}, OOT={best['r2_oot']:.3f}, "
                 f"RMSE={best['rmse']:.2f} USD/fat",
                 fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "33_full_regression.png"
    fig.savefig(out_png, dpi=140)
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
        api = sub["api_gravity"].iloc[0]
        sulfur = sub["sulfur_pct"].iloc[0]
        vr = sub["vacuum_resid_pct"].iloc[0]
        ax.set_title(f"{grade} (API={api:.0f}°, S={sulfur:.1f}%, VR={vr:.0f}%, RMSE={rmse:.1f})")
        ax.set_ylabel("Diff. (USD/fat)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Full modell — tidsserier predikert vs faktisk", fontsize=13)
    fig2.tight_layout()
    out_png2 = OUT_DIR / "33_full_timeseries.png"
    fig2.savefig(out_png2, dpi=140)
    print(f"Tidsserier lagret: {out_png2}")


if __name__ == "__main__":
    main()
