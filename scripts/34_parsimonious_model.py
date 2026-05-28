"""
Parsimonious regresjonsmodell — fjern alt som ikke gir signifikant bidrag.

Strategi (backwards stepwise elimination):
  1. Start fra full feature-set (uten year FE, som ødelegger OOT)
  2. Eksplisitt fjern VIX (equity-vol er ikke oil-vol — bruker brent_volatility_3m)
  3. Iterativt fjern variabelen med høyest p-verdi hvis p > 0.10
  4. Beholdregel: en base-variabel beholdes hvis dens interaksjon er signifikant
  5. Beholdregel: API + sulfur er teoretiske ankere (alltid med)
  6. Stopp når alle p ≤ 0.10

Sammenligner deretter parsimonious vs full modell:
  - In-sample R²
  - 5-fold CV R²
  - Out-of-time R² (siste 24 mnd)
  - Antall features (parsimonious bør være ~halvparten)
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

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR

# Variabler vi ALDRI fjerner (teoretiske ankere)
ALWAYS_KEEP = {"api_gravity", "sulfur_pct"}

# Variabler vi eksplisitt ikke vil ha (brukerens input)
ALWAYS_DROP = {"vix"}

# Interaksjon → base-variabler (hvis interaksjonen er sig, behold basene)
INTERACTION_PARENTS = {
    "api2": ["api_gravity"],
    "sulfur_x_brent": ["sulfur_pct", "brent_price"],
    "vacuum_resid_x_brent": ["vacuum_resid_pct", "brent_price"],
    "ccr_x_brent": ["ccr_wt_pct", "brent_price"],
    "landlocked_x_cushing_stocks": ["is_landlocked", "cushing_stocks_kbbl_dev_5y_pct"],
    "api_x_contango": ["api_gravity", "fc_slope_4m"],
    "sulfur_x_refinery_util": ["sulfur_pct", "us_refinery_util_pct"],
    "landlocked_x_contango": ["is_landlocked", "fc_slope_4m"],
}

SIG_THRESHOLD = 0.10  # p-verdi for å beholde en variabel


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


def fit_eval(df: pd.DataFrame, features: list[str]) -> dict:
    sub = df.dropna(subset=features + ["differential"]).copy()
    if len(sub) < 100:
        return None
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
    m = sm.OLS(y, X).fit(cov_type="HC1")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(LinearRegression(), sub[features].values, y.values, cv=kf, scoring="r2")

    sub["date"] = pd.to_datetime(sub["date_str"])
    cutoff = sub["date"].max() - pd.DateOffset(months=24)
    train = sub[sub["date"] <= cutoff]
    test = sub[sub["date"] > cutoff]
    r2_oot, rmse_oot = np.nan, np.nan
    if len(train) > 100 and len(test) > 50:
        lr = LinearRegression()
        lr.fit(train[features].values, train["differential"].values)
        pred = lr.predict(test[features].values)
        ss_res = ((test["differential"].values - pred) ** 2).sum()
        ss_tot = ((test["differential"].values - test["differential"].mean()) ** 2).sum()
        r2_oot = 1 - ss_res / ss_tot
        rmse_oot = np.sqrt(((test["differential"].values - pred) ** 2).mean())

    rmse = np.sqrt(((y - m.fittedvalues) ** 2).mean())
    return {
        "model": m, "n": len(sub), "k": len(features),
        "r2": m.rsquared, "r2_adj": m.rsquared_adj,
        "r2_cv": cv.mean(), "r2_cv_std": cv.std(),
        "r2_oot": r2_oot, "rmse": rmse, "rmse_oot": rmse_oot,
        "features": features, "data": sub,
    }


def stepwise_eliminate(df: pd.DataFrame, initial_features: list[str], verbose: bool = True) -> dict:
    """Backwards stepwise: fjern variabelen med høyest p-verdi over terskel,
    så lenge vi ikke bryter beholdregler."""
    features = [f for f in initial_features if f not in ALWAYS_DROP]
    iteration = 0

    while True:
        iteration += 1
        result = fit_eval(df, features)
        if result is None:
            return None
        m = result["model"]
        pvals = m.pvalues.drop("const", errors="ignore")

        # Hvilke variabler er over terskel?
        candidates = pvals[pvals > SIG_THRESHOLD].sort_values(ascending=False)

        # Filtrer bort: ALWAYS_KEEP og base-variabler for signifikante interaksjoner
        protected = set(ALWAYS_KEEP)
        for inter, parents in INTERACTION_PARENTS.items():
            if inter in features and inter in pvals.index and pvals[inter] < SIG_THRESHOLD:
                protected.update(parents)

        candidates = candidates[~candidates.index.isin(protected)]

        if len(candidates) == 0:
            if verbose:
                print(f"\n  Konvergerte etter {iteration} iterasjoner.")
                print(f"  Sluttet med {len(features)} features (alle p ≤ {SIG_THRESHOLD} eller beskyttet).")
            return result

        # Fjern verste
        worst = candidates.index[0]
        worst_p = candidates.iloc[0]
        if verbose and iteration <= 3 or iteration % 5 == 0:
            print(f"  Iter {iteration:2d}: dropper '{worst}' (p={worst_p:.3f}), "
                  f"k={len(features)}→{len(features)-1}, "
                  f"R²={result['r2']:.4f}, CV={result['r2_cv']:.4f}, OOT={result['r2_oot']:.4f}")
        features = [f for f in features if f != worst]

        if len(features) < 5:
            if verbose:
                print("  For få features igjen — stopper.")
            return result


def sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."


def main() -> None:
    df = load_panel()
    print(f"Panel: {len(df):,} obs, {df['grade'].nunique()} grades\n")

    # Region-dummies
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg", drop_first=True, dtype=int)
    df = pd.concat([df, region_dums], axis=1)
    region_cols = list(region_dums.columns)

    # Full feature-set (samme som M11 i 33-scriptet — uten year FE)
    initial_features = (
        ["api_gravity", "sulfur_pct", "api2"]
        + region_cols
        + ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"]
        + ["brent_price", "wti_brent_spread", "vix"]   # vix blir droppet av ALWAYS_DROP
        + ["gasoline_crack_brent", "diesel_crack_brent",
           "diesel_minus_gasoline_crack", "brent_dubai_spread"]
        + ["d_distance_medium", "d_distance_long",
           "is_landlocked", "is_pipeline_constrained"]
        + ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
           "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
           "d_refinery_tight", "d_refinery_slack"]
        + ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"]
        + ["sin_month", "cos_month", "d_winter"]
        + ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
           "landlocked_x_cushing_stocks", "api_x_contango",
           "sulfur_x_refinery_util", "landlocked_x_contango"]
        + ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
           "d_venezuela_sanctions", "d_us_shale_boom", "d_covid",
           "d_opec_plus_cuts_2023"]
    )
    initial_features = [f for f in initial_features if f in df.columns]
    print(f"Starter med {len(initial_features)} kandidat-features.")
    print(f"  - Beskyttet (ALWAYS_KEEP): {sorted(ALWAYS_KEEP)}")
    print(f"  - Eksplisitt droppet (ALWAYS_DROP): {sorted(ALWAYS_DROP)}")
    print(f"  - Signifikans-terskel: p ≤ {SIG_THRESHOLD}\n")

    # === Full modell (referanse) ===
    print("=" * 100)
    print("FULL MODELL (alle features unntatt VIX)")
    print("=" * 100)
    full = fit_eval(df, [f for f in initial_features if f not in ALWAYS_DROP])
    print(f"  N={full['n']:,}, k={full['k']}, R²={full['r2']:.4f}, "
          f"adj R²={full['r2_adj']:.4f}, CV={full['r2_cv']:.4f}, "
          f"OOT={full['r2_oot']:.4f}, RMSE={full['rmse']:.2f}")

    # === Stepwise eliminering ===
    print("\n" + "=" * 100)
    print("BACKWARDS STEPWISE ELIMINATION")
    print("=" * 100)
    parsi = stepwise_eliminate(df, initial_features, verbose=True)

    # === Sammenligning ===
    print("\n" + "=" * 100)
    print("SAMMENLIGNING: FULL vs PARSIMONIOUS")
    print("=" * 100)
    print(f"{'Modell':<20} {'k':>5} {'R²':>8} {'adj R²':>8} {'CV R²':>8} {'OOT R²':>8} {'RMSE':>7}")
    print("-" * 100)
    print(f"{'Full (uten VIX)':<20} {full['k']:>5} {full['r2']:>8.4f} {full['r2_adj']:>8.4f} "
          f"{full['r2_cv']:>8.4f} {full['r2_oot']:>8.4f} {full['rmse']:>7.2f}")
    print(f"{'Parsimonious':<20} {parsi['k']:>5} {parsi['r2']:>8.4f} {parsi['r2_adj']:>8.4f} "
          f"{parsi['r2_cv']:>8.4f} {parsi['r2_oot']:>8.4f} {parsi['rmse']:>7.2f}")
    print(f"{'Forskjell':<20} {parsi['k']-full['k']:>+5} {parsi['r2']-full['r2']:>+8.4f} "
          f"{parsi['r2_adj']-full['r2_adj']:>+8.4f} {parsi['r2_cv']-full['r2_cv']:>+8.4f} "
          f"{parsi['r2_oot']-full['r2_oot']:>+8.4f} {parsi['rmse']-full['rmse']:>+7.2f}")

    # === Endelig parsimonious modell ===
    m = parsi["model"]
    data = parsi["data"]

    print("\n" + "=" * 100)
    print(f"PARSIMONIOUS MODELL — alle koeffisienter (k={parsi['k']})")
    print("=" * 100)

    coef_table = pd.DataFrame({
        "coef": m.params,
        "std_err": m.bse,
        "t": m.tvalues,
        "p_value": m.pvalues,
    })
    coef_table["signif"] = coef_table["p_value"].apply(sig)
    print(coef_table.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Grupper i tabell-form
    print("\n" + "=" * 100)
    print("KOEFFISIENTER GRUPPERT:")
    print("=" * 100)
    groups = {
        "Kjernekvalitet": ["api_gravity", "api2", "sulfur_pct"],
        "Region": region_cols,
        "Assay (refining-økonomi)": ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"],
        "Marked": ["brent_price", "wti_brent_spread"],
        "Crack spreads": ["gasoline_crack_brent", "diesel_crack_brent",
                          "diesel_minus_gasoline_crack", "brent_dubai_spread"],
        "Logistikk": ["d_distance_medium", "d_distance_long",
                      "is_landlocked", "is_pipeline_constrained"],
        "EIA fundamentals": ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
                             "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
                             "d_refinery_tight", "d_refinery_slack"],
        "Forward curve": ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"],
        "Sesong": ["sin_month", "cos_month", "d_winter"],
        "Interaksjoner": ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
                          "landlocked_x_cushing_stocks", "api_x_contango",
                          "sulfur_x_refinery_util", "landlocked_x_contango"],
        "Hendelser": ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
                      "d_venezuela_sanctions", "d_us_shale_boom", "d_covid",
                      "d_opec_plus_cuts_2023"],
    }

    for group_name, group_feats in groups.items():
        in_model = [f for f in group_feats if f in m.params.index]
        dropped = [f for f in group_feats if f in initial_features and f not in m.params.index]
        if not in_model and not dropped:
            continue
        print(f"\n  --- {group_name} ---")
        for f in in_model:
            print(f"    [BEHOLDT]  {f:38s}: {m.params[f]:+9.4f}  (p {m.pvalues[f]:.4f}) {sig(m.pvalues[f])}")
        for f in dropped:
            print(f"    [DROPPET]  {f}")

    # Sweet spot
    if "api_gravity" in m.params and "api2" in m.params:
        opt = -m.params["api_gravity"] / (2 * m.params["api2"])
        print(f"\n  Optimal API-grad (sweet spot): {opt:.1f}°")

    # === Per-grade analyse ===
    X = sm.add_constant(data[parsi["features"]].astype(float))
    data = data.copy()
    data["predicted"] = m.predict(X)
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

    print("\n" + "=" * 100)
    print("RESIDUALER PER GRADE")
    print("=" * 100)
    print(grade_summary.to_string(float_format=lambda x: f"{x:+.2f}"))

    # === Lagre ===
    save_cols = [c for c in ["grade", "date_str", "differential", "predicted", "residual",
                              "api_gravity", "sulfur_pct", "vacuum_resid_pct",
                              "brent_price", "region_simple", "distance_band", "is_landlocked"]
                 if c in data.columns]
    data[save_cols].to_csv(OUT_DIR / "34_parsimonious_residuals.csv", index=False)
    grade_summary.reset_index().to_csv(OUT_DIR / "34_parsimonious_grade_summary.csv", index=False)

    export = {
        "model_name": "Parsimonious (backwards stepwise)",
        "metrics": {
            "r2": round(parsi["r2"], 4), "r2_adj": round(parsi["r2_adj"], 4),
            "r2_cv": round(parsi["r2_cv"], 4), "r2_cv_std": round(parsi["r2_cv_std"], 4),
            "r2_oot": round(parsi["r2_oot"], 4),
            "rmse": round(parsi["rmse"], 2), "rmse_oot": round(parsi["rmse_oot"], 2),
            "n_obs": int(parsi["n"]), "n_features": parsi["k"],
            "n_grades": int(data["grade"].nunique()),
        },
        "coefficients": {k: round(float(v), 6) for k, v in m.params.items()},
        "std_errors": {k: round(float(v), 6) for k, v in m.bse.items()},
        "p_values": {k: round(float(v), 6) for k, v in m.pvalues.items()},
        "features": parsi["features"],
        "dropped_features": [f for f in initial_features if f not in parsi["features"]],
    }
    with open(OUT_DIR / "34_parsimonious_model.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nLagret: 34_parsimonious_model.json, 34_parsimonious_residuals.csv")

    # === PLOTT ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # 1. Predikert vs faktisk
    ax = axes[0, 0]
    region_colors = {"NorthSea": "steelblue", "NorthAmerica": "coral",
                     "MiddleEast": "gold", "WestAfrica": "green",
                     "LatAm": "purple", "NorthAfrica": "orange",
                     "FSU": "brown", "AsiaPac": "pink"}
    for region in sorted(data["region_simple"].unique()):
        sub = data[data["region_simple"] == region]
        ax.scatter(sub["predicted"], sub["differential"], alpha=0.25, s=10,
                   color=region_colors.get(region, "gray"), label=region)
    lim = max(abs(data["predicted"]).max(), abs(data["differential"]).max()) + 2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predikert (USD/fat)")
    ax.set_ylabel("Faktisk (USD/fat)")
    ax.set_title(f"Parsimonious — R²={parsi['r2']:.3f}, CV={parsi['r2_cv']:.3f}, OOT={parsi['r2_oot']:.3f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # 2. Full vs Parsimonious sammenligning
    ax = axes[0, 1]
    cats = ["In-sample R²", "Adj R²", "CV R²", "OOT R²"]
    full_vals = [full["r2"], full["r2_adj"], full["r2_cv"], full["r2_oot"]]
    pars_vals = [parsi["r2"], parsi["r2_adj"], parsi["r2_cv"], parsi["r2_oot"]]
    x_pos = np.arange(len(cats))
    width = 0.35
    ax.bar(x_pos - width/2, full_vals, width, label=f"Full ({full['k']} feat)", color="coral", alpha=0.8)
    ax.bar(x_pos + width/2, pars_vals, width, label=f"Parsimonious ({parsi['k']} feat)", color="steelblue", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats)
    ax.set_ylabel("R²")
    ax.set_title("Full vs Parsimonious — har vi mistet noe?")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (f, p) in enumerate(zip(full_vals, pars_vals)):
        ax.text(i - width/2, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
        ax.text(i + width/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)

    # 3. Topp features (standardiserte koef, ekskl regions/dummies)
    ax = axes[0, 2]
    feat_cols = [f for f in parsi["features"]
                 if not f.startswith("reg_") and not f.startswith("d_")]
    if feat_cols:
        coefs = pd.Series({f: m.params.get(f, 0) for f in feat_cols})
        stds = data[feat_cols].std()
        std_coefs = (coefs * stds).sort_values()
        n_show = min(20, len(std_coefs))
        top = pd.concat([std_coefs.head(n_show // 2), std_coefs.tail(n_show // 2)])
        pvals = [m.pvalues.get(f, 1) for f in top.index]
        colors_b = ["steelblue" if p < 0.05 else "skyblue" if p < 0.10 else "lightgray" for p in pvals]
        ax.barh(top.index, top.values, color=colors_b)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Standardisert koeffisient (mørk blå = p<0.05)")
        ax.set_title("Topp features etter |standardisert effekt|")
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

    # 4. RMSE per grade
    ax = axes[1, 0]
    gs = grade_summary.sort_values("rmse", ascending=False)
    nor_grades = ["Johan Sverdrup", "Troll", "Ekofisk", "Oseberg", "Alvheim",
                  "Gullfaks", "Statfjord", "Heidrun", "Grane", "Asgard"]
    colors_r = ["steelblue" if g in nor_grades else "lightgray" for g in gs.index]
    ax.barh(gs.index, gs["rmse"], color=colors_r, alpha=0.8)
    ax.set_xlabel("RMSE (USD/fat)")
    ax.set_title("Prediksjonsfeil per grade (blå = norske)")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="x", alpha=0.3)

    # 5. Residualer per grade
    ax = axes[1, 1]
    gs_resid = grade_summary.sort_values("resid")
    colors_r2 = ["steelblue" if g in nor_grades else "lightgray" for g in gs_resid.index]
    ax.barh(gs_resid.index, gs_resid["resid"], color=colors_r2, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean residual (USD/fat)")
    ax.set_title("Residualer per grade")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="x", alpha=0.3)

    # 6. Effekt-størrelse av feature-grupper (varians forklart)
    ax = axes[1, 2]
    group_summary = {}
    for gname, gfeats in groups.items():
        in_model = [f for f in gfeats if f in m.params.index]
        if in_model:
            # Forklart varians fra denne gruppen alene
            pred_group = (data[in_model] * m.params[in_model]).sum(axis=1)
            var_explained = np.var(pred_group)
            group_summary[gname] = var_explained
    if group_summary:
        gs_sorted = pd.Series(group_summary).sort_values()
        ax.barh(gs_sorted.index, gs_sorted.values, color="steelblue", alpha=0.8)
        ax.set_xlabel("Varians forklart fra denne gruppen alene")
        ax.set_title("Bidrag per feature-gruppe (parsimonious)")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Parsimonious modell — alle features har p ≤ {SIG_THRESHOLD}\n"
                 f"N={parsi['n']:,}, grades={data['grade'].nunique()}, "
                 f"features={parsi['k']}, R²={parsi['r2']:.3f}, "
                 f"CV={parsi['r2_cv']:.3f}, OOT={parsi['r2_oot']:.3f}, RMSE={parsi['rmse']:.2f}",
                 fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "34_parsimonious_model.png"
    fig.savefig(out_png, dpi=140)
    print(f"Plott lagret: {out_png}")


if __name__ == "__main__":
    main()
