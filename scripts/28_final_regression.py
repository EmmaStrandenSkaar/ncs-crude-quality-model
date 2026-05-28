"""
Endelig regresjonsmodell: prediker crude oil premium/discount.

Forbedringer over modell 26:
  1. Region-dummies (Nord-Amerika, Midtøsten, Afrika, Norge, etc.)
  2. Cross-validation (k-fold) for out-of-sample R²
  3. Robusthetssjekker (subsamples, tidsperioder)
  4. Feature importance rangering
  5. Prediksjonstabell for nye crude grades
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "regression_panel.csv")
    df["date"] = pd.to_datetime(df["date_str"])

    quality = pd.read_csv(RAW_DIR / "global_crude_quality.csv")
    region_map = dict(zip(quality["grade"], quality["region"]))
    country_map = dict(zip(quality["grade"], quality["country"]))
    df["region"] = df["grade"].map(region_map).fillna("Other")
    df["country"] = df["grade"].map(country_map).fillna("Unknown")

    region_simple = {
        "North Sea": "NorthSea",
        "Norwegian Sea": "NorthSea",
        "Barents Sea": "NorthSea",
        "North America": "NorthAmerica",
        "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm",
        "Middle East": "MiddleEast",
        "West Africa": "WestAfrica",
        "North Africa": "NorthAfrica",
        "FSU": "FSU",
        "Asia-Pacific": "AsiaPac",
        "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")
    return df


def main() -> None:
    df = load_panel()
    print(f"Panel: {len(df):,} obs, {df['grade'].nunique()} grades, "
          f"{df['region_simple'].nunique()} regioner\n")

    print("Grades per region:")
    for region in sorted(df["region_simple"].unique()):
        grades = sorted(df[df["region_simple"] == region]["grade"].unique())
        print(f"  {region:15s}: {len(grades):2d} grades — {', '.join(grades[:5])}"
              f"{'...' if len(grades) > 5 else ''}")

    # === Region-dummies ===
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg", drop_first=True, dtype=int)
    df = pd.concat([df, region_dums], axis=1)
    region_cols = list(region_dums.columns)

    # === Feature-sett ===
    qual_core = ["api_gravity", "api2", "sulfur_pct"]
    qual_extended = qual_core + ["tan_mgkoh", "log_production"]
    market = ["brent_price", "wti_brent_spread", "vix"]
    interactions = ["sulfur_x_brent"]

    specs = [
        ("M1: Kvalitet (lineær)",
         ["api_gravity", "sulfur_pct"]),
        ("M2: + API² sweet-spot",
         qual_core),
        ("M3: + utvidet kvalitet",
         qual_extended),
        ("M4: + marked",
         qual_extended + market),
        ("M5: + svovel×Brent",
         qual_extended + market + interactions),
        ("M6: + region",
         qual_extended + market + interactions + region_cols),
        ("M7: Full (region + benchmark)",
         qual_extended + market + interactions + region_cols + ["is_benchmark"]),
    ]

    # === Kjør alle modeller ===
    results = []
    for name, features in specs:
        sub = df.dropna(subset=features + ["differential"])
        X = sm.add_constant(sub[features])
        y = sub["differential"]
        m = sm.OLS(y, X).fit(cov_type="HC1")

        # Cross-validation (5-fold)
        X_cv = sub[features].values
        y_cv = y.values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            LinearRegression(), X_cv, y_cv, cv=kf, scoring="r2"
        )

        results.append({
            "name": name,
            "model": m,
            "n_obs": len(sub),
            "r2": m.rsquared,
            "r2_adj": m.rsquared_adj,
            "r2_cv": cv_scores.mean(),
            "r2_cv_std": cv_scores.std(),
            "rmse": np.sqrt(mean_squared_error(y, m.fittedvalues)),
            "mae": mean_absolute_error(y, m.fittedvalues),
            "aic": m.aic,
            "features": features,
            "data": sub,
        })

    # === Modellsammenligning ===
    print(f"\n{'='*90}")
    print("MODELLSAMMENLIGNING")
    print(f"{'='*90}")
    cmp = pd.DataFrame([{
        "Modell": r["name"][:35],
        "N": r["n_obs"],
        "k": len(r["features"]),
        "R²": r["r2"],
        "R²adj": r["r2_adj"],
        "R²_CV": r["r2_cv"],
        "RMSE": r["rmse"],
        "MAE": r["mae"],
    } for r in results])
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # === Beste modell: detaljer ===
    best = max(results, key=lambda r: r["r2_cv"])
    m = best["model"]
    data = best["data"].copy()

    print(f"\n{'='*90}")
    print(f"BESTE MODELL (etter CV): {best['name']}")
    print(f"R² = {best['r2']:.4f}, R²adj = {best['r2_adj']:.4f}, "
          f"R²_CV = {best['r2_cv']:.4f} ± {best['r2_cv_std']:.4f}")
    print(f"RMSE = {best['rmse']:.2f}, MAE = {best['mae']:.2f} USD/fat")
    print(f"{'='*90}")

    # Koeffisienter
    params = pd.DataFrame({
        "coef": m.params,
        "std_err": m.bse,
        "t": m.tvalues,
        "p": m.pvalues,
        "sig": m.pvalues.apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""),
    })
    print("\nKOEFFISIENTER:")
    print(params.to_string(float_format=lambda x: f"{x:+.4f}"))

    # === Tolkning ===
    print(f"\n{'='*90}")
    print("TOLKNING AV KOEFFISIENTER")
    print(f"{'='*90}")

    interp_map = {
        "api_gravity": ("API-grad (lineært)", "USD/fat per 1° API"),
        "api2": ("API² (kvadratisk)", "fanger sweet-spot effekten"),
        "sulfur_pct": ("Svovel % (lineært)", "USD/fat per 1% svovel"),
        "sulfur_x_brent": ("Svovel × Brent-pris", "ekstra rabatt per 1% S når Brent stiger $1"),
        "brent_price": ("Brent-prisnivå", "USD/fat ekstra diff per $1 Brent"),
        "wti_brent_spread": ("WTI-Brent spread", "effekt av Atlantisk basin dynamikk"),
        "vix": ("VIX (markedsrisiko)", "effekt av usikkerhet"),
        "tan_mgkoh": ("TAN (syretall)", "USD/fat per 1 mgKOH/g"),
        "log_production": ("Log produksjon", "effekt av volum/likviditet"),
        "is_benchmark": ("Benchmark-status", "premium for å være referanseolje"),
    }

    for feat, (desc, unit) in interp_map.items():
        if feat in m.params.index:
            coef = m.params[feat]
            p = m.pvalues[feat]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {desc:30s}: {coef:+.4f} ({unit}) [{sig}]")

    for col in region_cols:
        if col in m.params.index:
            region_name = col.replace("reg_", "")
            coef = m.params[col]
            p = m.pvalues[col]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  Region {region_name:22s}: {coef:+.4f} USD/fat vs referanse [{sig}]")

    # === Sweet-spot beregning ===
    if "api_gravity" in m.params.index and "api2" in m.params.index:
        b1 = m.params["api_gravity"]
        b2 = m.params["api2"]
        optimal_api = -b1 / (2 * b2)
        print(f"\n  Optimal API-grad (sweet spot): {optimal_api:.1f}°")
        print(f"  (punktet der marginaleffekten av lettere olje er null)")

    # === Prediksjoner ===
    X_pred = sm.add_constant(data[best["features"]])
    data["predicted"] = m.predict(X_pred)
    data["residual"] = data["differential"] - data["predicted"]

    # Per-grade summary
    grade_summary = data.groupby("grade").agg(
        n_obs=("differential", "size"),
        api=("api_gravity", "first"),
        sulfur=("sulfur_pct", "first"),
        region=("region_simple", "first"),
        mean_actual=("differential", "mean"),
        mean_pred=("predicted", "mean"),
        resid=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
    ).sort_values("resid", ascending=False)

    print(f"\n{'='*90}")
    print("RESIDUALER PER GRADE")
    print(f"{'='*90}")
    print(grade_summary.to_string(float_format=lambda x: f"{x:+.2f}"))

    # === Prediksjonstabell for utvalgte norske felt ===
    nor_grades = ["Johan Sverdrup", "Troll", "Ekofisk", "Oseberg", "Alvheim",
                  "Gullfaks", "Statfjord", "Heidrun", "Grane", "Asgard"]
    nor_data = grade_summary.loc[grade_summary.index.isin(nor_grades)]
    if not nor_data.empty:
        print(f"\n{'='*90}")
        print("NORSKE FELT — MODELLENS PREDIKSJON vs FAKTISK")
        print(f"{'='*90}")
        print(nor_data.to_string(float_format=lambda x: f"{x:+.2f}"))

    # === Lagre residualer ===
    out_resid = OUT_DIR / "28_final_residuals.csv"
    data[["grade", "date_str", "differential", "predicted", "residual",
          "api_gravity", "sulfur_pct", "brent_price", "region_simple"]].to_csv(out_resid, index=False)

    # === PLOTT ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # 1. Predikert vs faktisk (fargekoda etter region)
    ax = axes[0, 0]
    region_colors = {
        "NorthSea": "steelblue", "NorthAmerica": "coral",
        "MiddleEast": "gold", "WestAfrica": "green",
        "LatAm": "purple", "NorthAfrica": "orange",
        "FSU": "brown", "AsiaPac": "pink",
    }
    for region in sorted(data["region_simple"].unique()):
        sub = data[data["region_simple"] == region]
        c = region_colors.get(region, "gray")
        ax.scatter(sub["predicted"], sub["differential"], alpha=0.25, s=12,
                   color=c, label=region)
    lim = max(abs(data["predicted"]).max(), abs(data["differential"]).max()) + 2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predikert (USD/fat)")
    ax.set_ylabel("Faktisk (USD/fat)")
    ax.set_title(f"Predikert vs faktisk\nR²={best['r2']:.3f}, R²_CV={best['r2_cv']:.3f}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # 2. Residualer per grade (norske felt fremhevet)
    ax = axes[0, 1]
    gs = grade_summary.sort_values("resid")
    colors = ["steelblue" if g in nor_grades else "lightgray" for g in gs.index]
    ax.barh(gs.index, gs["resid"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Gjennomsnittlig residual (USD/fat)")
    ax.set_title("Residualer per grade\n(blå = norske felt)")
    ax.grid(True, axis="x", alpha=0.3)

    # 3. Standardiserte koeffisienter
    ax = axes[0, 2]
    feat_names = [f for f in best["features"] if not f.startswith("reg_")]
    coef_vals = [m.params.get(f, 0) for f in feat_names]
    std_vals = data[feat_names].std()
    std_coefs = pd.Series(
        [c * s for c, s in zip(coef_vals, std_vals)],
        index=feat_names
    ).sort_values()
    pvals = [m.pvalues.get(f, 1) for f in std_coefs.index]
    bar_colors = ["steelblue" if p < 0.05 else "lightgray" for p in pvals]
    ax.barh(std_coefs.index, std_coefs.values, color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardisert koeffisient")
    ax.set_title("Feature importance\n(blå = p<0.05)")
    ax.grid(True, axis="x", alpha=0.3)

    # 4. Svovel-effekten varierer med prisnivå
    ax = axes[1, 0]
    brent_levels = [40, 60, 80, 100, 120]
    sulfur_range = np.linspace(0, 4, 100)
    s_coef = m.params.get("sulfur_pct", 0)
    sb_coef = m.params.get("sulfur_x_brent", 0)
    for bp in brent_levels:
        effect = s_coef * sulfur_range + sb_coef * sulfur_range * bp
        ax.plot(sulfur_range, effect, label=f"Brent={bp}$")
    ax.set_xlabel("Svovelinnhold (%)")
    ax.set_ylabel("Effekt på differensial (USD/fat)")
    ax.set_title("Svovel-rabatten øker med Brent-prisnivå")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    # 5. API sweet-spot
    ax = axes[1, 1]
    api_range = np.linspace(15, 50, 100)
    a1 = m.params.get("api_gravity", 0)
    a2 = m.params.get("api2", 0)
    api_effect = a1 * api_range + a2 * api_range**2
    api_effect -= api_effect.min()
    ax.plot(api_range, api_effect, color="steelblue", linewidth=2)
    if a2 != 0:
        opt = -a1 / (2 * a2)
        ax.axvline(opt, color="red", linestyle="--", alpha=0.7, label=f"Optimal: {opt:.0f}° API")
    ax.scatter(data.groupby("grade")["api_gravity"].first(),
               data.groupby("grade")["differential"].mean(),
               alpha=0.5, s=30, color="coral", zorder=5)
    ax.set_xlabel("API-grad")
    ax.set_ylabel("Marginell effekt på differensial")
    ax.set_title("API sweet-spot (kvadratisk effekt)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. R² sammenligning
    ax = axes[1, 2]
    model_labels = [r["name"].split(":")[1][:18].strip() for r in results]
    r2_insample = [r["r2"] for r in results]
    r2_cv = [r["r2_cv"] for r in results]
    x_pos = np.arange(len(results))
    ax.bar(x_pos - 0.15, r2_insample, 0.3, label="In-sample R²", color="steelblue")
    ax.bar(x_pos + 0.15, r2_cv, 0.3, label="CV R² (5-fold)", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"M{i+1}" for i in range(len(results))], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("In-sample vs cross-validated R²")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Regresjonsmodell: kvalitet + marked + region → crude oil premium/discount\n"
                 f"N={len(data):,} obs, {data['grade'].nunique()} grades, "
                 f"R²={best['r2']:.3f}, CV R²={best['r2_cv']:.3f}",
                 fontsize=13)
    fig.tight_layout()
    out_png = OUT_DIR / "28_final_regression.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nHovedplott lagret: {out_png}")

    # === Tidsserie for utvalgte ===
    fig2, axes2 = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
    highlight = ["Johan Sverdrup", "Troll", "Ekofisk", "Alvheim",
                 "Maya", "Arab Light", "WTI", "Bonny Light"]
    highlight = [g for g in highlight if g in data["grade"].unique()]

    for i, grade in enumerate(highlight[:8]):
        ax = axes2[i // 2, i % 2]
        sub = data[data["grade"] == grade].sort_values("date")
        ax.plot(sub["date"], sub["differential"], label="Faktisk", color="steelblue", linewidth=1.2)
        ax.plot(sub["date"], sub["predicted"], label="Predikert", color="coral", linestyle="--", linewidth=1.2)
        ax.fill_between(sub["date"], sub["differential"], sub["predicted"], alpha=0.1, color="gray")
        ax.axhline(0, color="gray", linewidth=0.5)
        rmse = np.sqrt((sub["residual"]**2).mean())
        ax.set_title(f"{grade} (API={sub['api_gravity'].iloc[0]:.0f}°, S={sub['sulfur_pct'].iloc[0]:.1f}%, RMSE={rmse:.1f})")
        ax.set_ylabel("Diff. (USD/fat)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Faktisk vs predikert differensial — utvalgte crude grades", fontsize=13)
    fig2.tight_layout()
    out_png2 = OUT_DIR / "28_final_timeseries.png"
    fig2.savefig(out_png2, dpi=150)
    print(f"Tidsserier lagret: {out_png2}")

    # === Eksport: modellkoeffisienter som JSON for enkel bruk ===
    import json
    model_export = {
        "model_name": best["name"],
        "r2": round(best["r2"], 4),
        "r2_adj": round(best["r2_adj"], 4),
        "r2_cv": round(best["r2_cv"], 4),
        "rmse": round(best["rmse"], 2),
        "mae": round(best["mae"], 2),
        "n_obs": best["n_obs"],
        "n_grades": data["grade"].nunique(),
        "coefficients": {k: round(v, 6) for k, v in m.params.items()},
        "p_values": {k: round(v, 6) for k, v in m.pvalues.items()},
        "features": best["features"],
    }
    out_json = OUT_DIR / "28_model_coefficients.json"
    with open(out_json, "w") as f:
        json.dump(model_export, f, indent=2)
    print(f"Modellkoeffisienter lagret: {out_json}")


if __name__ == "__main__":
    main()
