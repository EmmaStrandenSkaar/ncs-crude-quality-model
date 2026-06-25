"""
Panelregresjoner: kvalitet + markedskontroller → differensial mot Brent.

Vi kjører åtte modellspesifikasjoner for å vise hva som driver premium/discount:

  1. Pooled OLS: kun kvalitet (API + svovel)
  2. Pooled OLS: kvalitet + kvad. API (sweet-spot)
  3. Pooled OLS: full kvalitet (+ TAN, viskositet, benchmark)
  4. Pooled OLS: kvalitet + markedskontroller
  5. Pooled OLS: kvalitet + interaksjoner (API×Brent, svovel×Brent)
  6. Pooled OLS: full modell + sesong
  7. Fixed Effects (within-grade): tidsvariasjon innad i hvert felt
  8. Random Effects (GLS): mellom + innad

Output:
  - Modellsammenligning
  - Koeffisienter for beste modell
  - Predikert vs faktisk scatterplot
  - Feature importance
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "regression_panel.csv")
    df["date"] = pd.to_datetime(df["date_str"])
    return df


def fit_ols(df: pd.DataFrame, features: list[str], name: str) -> dict:
    """Fit WLS (vektet etter antall observasjoner per grade for balansering)."""
    sub = df.dropna(subset=features + ["differential"])
    X = sm.add_constant(sub[features])
    y = sub["differential"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return {
        "name": name,
        "model": model,
        "n_obs": len(sub),
        "n_grades": sub["grade"].nunique(),
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
        "k": len(features),
        "aic": model.aic,
        "bic": model.bic,
        "features": features,
        "data": sub,
    }


def fit_panel_fe(df: pd.DataFrame, features: list[str], name: str) -> dict:
    """Panel Fixed Effects (within-estimator)."""
    sub = df.dropna(subset=features + ["differential"]).copy()
    sub = sub.set_index(["grade", "date"])
    y = sub["differential"]
    X = sub[features]
    model = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
    return {
        "name": name,
        "model": model,
        "n_obs": model.nobs,
        "n_grades": sub.index.get_level_values(0).nunique(),
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj if hasattr(model, "rsquared_adj") else None,
        "k": len(features),
        "aic": None,
        "bic": None,
        "features": features,
        "data": sub,
    }


def fit_panel_re(df: pd.DataFrame, features: list[str], name: str) -> dict:
    """Panel Random Effects (GLS)."""
    sub = df.dropna(subset=features + ["differential"]).copy()
    sub = sub.set_index(["grade", "date"])
    y = sub["differential"]
    X = sm.add_constant(sub[features])
    model = RandomEffects(y, X).fit(cov_type="clustered", cluster_entity=True)
    return {
        "name": name,
        "model": model,
        "n_obs": model.nobs,
        "n_grades": sub.index.get_level_values(0).nunique(),
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj if hasattr(model, "rsquared_adj") else None,
        "k": len(features),
        "aic": None,
        "bic": None,
        "features": features,
        "data": sub,
    }


def main() -> None:
    df = load_panel()
    print(f"Panel: {len(df):,} obs, {df['grade'].nunique()} grades\n")

    # === Definér feature-sett ===
    qual_basic = ["api_gravity", "sulfur_pct"]
    qual_quad = ["api_gravity", "api2", "sulfur_pct"]
    qual_full = ["api_gravity", "api2", "sulfur_pct", "tan_mgkoh",
                 "is_benchmark", "log_production"]
    market = ["brent_price", "wti_brent_spread", "vix"]
    interact = ["api_x_brent", "sulfur_x_brent"]
    season = ["quarter"]

    # Modeller for tidsvariasjon (FE/RE kan ikke bruke tidsinvariante features)
    time_varying = ["brent_price", "wti_brent_spread", "vix"]

    results = []

    # === 1-6: Pooled OLS ===
    specs = [
        ("1: Kvalitet (lineær)", qual_basic),
        ("2: + API² (sweet-spot)", qual_quad),
        ("3: Full kvalitet", qual_full),
        ("4: + markedskontroller", qual_full + market),
        ("5: + interaksjoner", qual_full + market + interact),
        ("6: + sesong (kvartal)", qual_full + market + interact + season),
    ]

    for name, features in specs:
        r = fit_ols(df, features, name)
        results.append(r)

    # === 7: Fixed Effects ===
    try:
        r = fit_panel_fe(df, time_varying, "7: Fixed Effects (within)")
        results.append(r)
    except Exception as e:
        print(f"FE-modell feilet: {e}")

    # === 8: Random Effects ===
    try:
        r = fit_panel_re(df, qual_full + market, "8: Random Effects (GLS)")
        results.append(r)
    except Exception as e:
        print(f"RE-modell feilet: {e}")

    # === Sammenligning ===
    print("=" * 80)
    print("MODELLSAMMENLIGNING")
    print("=" * 80)
    cmp = pd.DataFrame([{
        "Modell": r["name"],
        "N": r["n_obs"],
        "Grades": r["n_grades"],
        "k": r["k"],
        "R²": r["r2"],
        "R²adj": r["r2_adj"],
        "AIC": r["aic"],
    } for r in results])
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x else ""))
    print()

    # === Detaljert output for alle OLS-modeller ===
    for r in results:
        if not hasattr(r["model"], "summary"):
            continue
        print("=" * 80)
        print(f"  {r['name']}")
        print(f"  N = {r['n_obs']:,}, R² = {r['r2']:.4f}")
        print("=" * 80)

        m = r["model"]
        if hasattr(m, "params") and hasattr(m, "pvalues"):
            params_df = pd.DataFrame({
                "coef": m.params,
                "std_err": m.bse if hasattr(m, "bse") else None,
                "t": m.tvalues if hasattr(m, "tvalues") else None,
                "p": m.pvalues,
            })
            sig = params_df["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
            params_df["sig"] = sig
            print(params_df.to_string(float_format=lambda x: f"{x:+.4f}"))
        print()

    # === Beste pooled OLS-modell: residualer ===
    ols_results = [r for r in results if r["name"].startswith(("1:", "2:", "3:", "4:", "5:", "6:"))]
    best = max(ols_results, key=lambda r: r["r2_adj"] if r["r2_adj"] else 0)
    print(f"\n{'='*80}")
    print(f"BESTE POOLED OLS: {best['name']}")
    print(f"R² = {best['r2']:.4f}, R²adj = {best['r2_adj']:.4f}")
    print(f"{'='*80}")

    m = best["model"]
    data = best["data"].copy()
    X_best = sm.add_constant(data[best["features"]])
    data["predicted"] = m.predict(X_best)
    data["residual"] = data["differential"] - data["predicted"]

    # Per-grade residualer
    grade_resid = data.groupby("grade").agg(
        n_obs=("differential", "size"),
        api=("api_gravity", "first"),
        sulfur=("sulfur_pct", "first"),
        mean_actual=("differential", "mean"),
        mean_predicted=("predicted", "mean"),
        mean_residual=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
    ).sort_values("mean_residual", ascending=False)

    print("\nRESIDUALER PER GRADE (sortert etter over/under-prediksjon):")
    print(grade_resid.to_string(float_format=lambda x: f"{x:+.2f}"))

    # Lagre residualer
    out_resid = OUT_DIR / "26_panel_residuals.csv"
    data[["grade", "date_str", "differential", "predicted", "residual",
          "api_gravity", "sulfur_pct", "brent_price"]].to_csv(out_resid, index=False)

    # === PLOTT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Predikert vs faktisk
    ax = axes[0, 0]
    for grade in data["grade"].unique():
        sub = data[data["grade"] == grade]
        ax.scatter(sub["predicted"], sub["differential"], alpha=0.3, s=15, label=grade)
    lim = max(abs(data["predicted"].min()), abs(data["predicted"].max()),
              abs(data["differential"].min()), abs(data["differential"].max())) + 2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predikert differensial (USD/fat)")
    ax.set_ylabel("Faktisk differensial (USD/fat)")
    ax.set_title(f"Predikert vs faktisk\n{best['name']}, R²={best['r2']:.3f}")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # 2. Residualer per grade (bar chart)
    ax = axes[0, 1]
    colors = ["seagreen" if r >= 0 else "indianred" for r in grade_resid["mean_residual"]]
    ax.barh(grade_resid.index, grade_resid["mean_residual"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Gjennomsnittlig residual (USD/fat)")
    ax.set_title("Residualer per grade\n(grønn = dyrere enn modellen, rød = billigere)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # 3. Koeffisienter (feature importance)
    ax = axes[1, 0]
    coef_df = pd.DataFrame({
        "feature": best["features"],
        "coef": [m.params.get(f, 0) for f in best["features"]],
        "pval": [m.pvalues.get(f, 1) for f in best["features"]],
    })
    # Standardisér for sammenligning
    std_x = data[best["features"]].std()
    coef_df["std_coef"] = coef_df["coef"] * std_x.values
    coef_df = coef_df.sort_values("std_coef", ascending=True)
    colors_c = ["steelblue" if p < 0.05 else "lightgray" for p in coef_df["pval"]]
    ax.barh(coef_df["feature"], coef_df["std_coef"], color=colors_c)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardisert koeffisient (effekt av 1 std-endring)")
    ax.set_title("Feature importance (blå = signifikant p<0.05)")
    ax.grid(True, axis="x", alpha=0.3)

    # 4. Modellsammenligning (R²)
    ax = axes[1, 1]
    model_names = [r["name"].split(":")[0] + ":" + r["name"].split(":")[1][:20] for r in results]
    r2_vals = [r["r2"] for r in results]
    r2_adj_vals = [r["r2_adj"] if r["r2_adj"] else r["r2"] for r in results]
    x_pos = np.arange(len(results))
    ax.bar(x_pos - 0.15, r2_vals, 0.3, label="R²", color="steelblue", alpha=0.8)
    ax.bar(x_pos + 0.15, r2_adj_vals, 0.3, label="R² adj", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"M{i+1}" for i in range(len(results))], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Modellsammenligning")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Panelregresjon: oljekvalitet → pris-differensial mot Brent\n"
                 f"N = {len(data):,} obs, {data['grade'].nunique()} grades",
                 fontsize=13)
    fig.tight_layout()

    out_png = OUT_DIR / "26_panel_regression.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nPlott lagret: {out_png}")

    # === Ekstra: tidsserieplot av residualer for utvalgte felt ===
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    highlight_grades = ["Johan Sverdrup", "Troll", "Ekofisk", "WTI", "Dubai Fateh", "Alvheim"]
    highlight_grades = [g for g in highlight_grades if g in data["grade"].unique()]

    for i, grade in enumerate(highlight_grades[:6]):
        ax = axes2[i // 2, i % 2]
        sub = data[data["grade"] == grade].sort_values("date")
        ax.plot(sub["date"], sub["differential"], label="Faktisk", color="steelblue", linewidth=1.5)
        ax.plot(sub["date"], sub["predicted"], label="Predikert", color="coral", linestyle="--", linewidth=1.5)
        ax.fill_between(sub["date"], sub["differential"], sub["predicted"],
                       alpha=0.15, color="gray")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{grade} (API={sub['api_gravity'].iloc[0]:.0f}, S={sub['sulfur_pct'].iloc[0]:.2f}%)")
        ax.set_ylabel("Differensial (USD/fat)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Tidsserier: faktisk vs predikert differensial (utvalgte grades)", fontsize=13)
    fig2.tight_layout()
    out_png2 = OUT_DIR / "26_panel_timeseries.png"
    fig2.savefig(out_png2, dpi=150)
    print(f"Tidsserie-plott lagret: {out_png2}")


if __name__ == "__main__":
    main()
