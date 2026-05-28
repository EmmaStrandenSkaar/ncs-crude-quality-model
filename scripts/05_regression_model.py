"""
Steg 6: Regresjonsmodell — kvalitet + struktur → pris-differensial mot Brent.

Vi bygger seks modeller, fra enkleste rene-kvalitet til full markeds-modell,
slik at vi ser hva hver variabel tilfører:

  A:  diff = β₀ + β₁·API + β₂·svovel               (rent kjemisk, lineært)
  B:  + API²                                        (sweet-spot for kondensat)
  C:  + kondensat-flagg (API≥45)                   (alternativ til API²)
  D:  + TAN, viskositet, pour pt                   (utvidet kjemi)
  E:  B + BFOET-flagg                              (benchmark-tilhørighet)
  F:  E + log(produksjon)                          (likviditet/markedsstørrelse)

Vi bruker WLS med vekter = antall månedlige observasjoner per felt.

Output:
  - Sammenligningstabell (R², justert R², antall variabler)
  - Detaljert sammendrag av hver modell
  - Residualer for beste modell (forskjell faktisk vs predikert)
  - Plott: predikert vs faktisk + residual-bar-chart
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_CSV = PROJECT_ROOT / "data" / "raw" / "crude_quality.csv"
DIFF_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

CONDENSATE_API_THRESHOLD = 45.0  # API ≥ 45 = svært lett, kondensat-aktig


def aggregate(diff: pd.DataFrame) -> pd.DataFrame:
    """Snitt-differensial per felt + spredning + antall obs (vekter)."""
    return diff.groupby("field").agg(
        mean_diff=("differential_usd", "mean"),
        std_diff=("differential_usd", "std"),
        n_obs=("differential_usd", "size"),
    ).reset_index()


def fit_wls(df: pd.DataFrame, feature_cols: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Vekt=n_obs. add_constant legger til interceptet (β₀)."""
    X = sm.add_constant(df[feature_cols])
    y = df["mean_diff"]
    w = df["n_obs"]
    return sm.WLS(y, X, weights=w).fit()


def model_summary_row(name: str, m, n: int) -> dict:
    return {
        "model": name,
        "n_fields": n,
        "r2": m.rsquared,
        "r2_adj": m.rsquared_adj,
        "k_features": len(m.params) - 1,  # minus interceptet
    }


def main() -> None:
    quality = pd.read_csv(QUALITY_CSV)
    diff = pd.read_csv(DIFF_CSV)

    quality["field"] = quality["field"].str.upper().str.strip()
    diff["field"] = diff["field"].str.upper().str.strip()

    df = quality.merge(aggregate(diff), on="field", how="inner")

    # Avledede features.
    df["api2"] = df["api_gravity"] ** 2
    df["is_condensate"] = (df["api_gravity"] >= CONDENSATE_API_THRESHOLD).astype(int)
    # log av produksjon: fanger at forskjellen 1k→10k betyr mer enn 100k→1M
    # for likviditeten. np.log1p (= ln(1+x)) er trygt selv ved x=0.
    df["log_production"] = np.log1p(df["production_kbpd"])

    print(f"Datasett: {len(df)} felt, totalt {df['n_obs'].sum():,} månedsobservasjoner.")
    print(f"BFOET-felt: {df.loc[df['is_bfoet'] == 1, 'field'].tolist()}")
    print(f"Kondensat-felt (API≥{CONDENSATE_API_THRESHOLD}): "
          f"{df.loc[df['is_condensate'] == 1, 'field'].tolist()}\n")

    # === Tilpass alle modellene ===
    fits: dict[str, tuple] = {}

    fits["A_lin"] = (
        "A: lineær (API + svovel)",
        fit_wls(df, ["api_gravity", "sulfur_pct"]),
        df,
    )
    fits["B_quad"] = (
        "B: + API² (sweet spot)",
        fit_wls(df, ["api_gravity", "api2", "sulfur_pct"]),
        df,
    )
    fits["C_cond"] = (
        "C: + kondensat-flagg (API≥45)",
        fit_wls(df, ["api_gravity", "sulfur_pct", "is_condensate"]),
        df,
    )
    cols_D = ["api_gravity", "sulfur_pct", "tan_mgkoh", "viscosity_cst_20c", "pour_point_c"]
    df_D = df.dropna(subset=cols_D)
    fits["D_full"] = (
        "D: full kvalitet (+ TAN, visk, pour pt)",
        fit_wls(df_D, cols_D),
        df_D,
    )
    fits["E_bfoet"] = (
        "E: B + BFOET-flagg",
        fit_wls(df, ["api_gravity", "api2", "sulfur_pct", "is_bfoet"]),
        df,
    )
    fits["F_full"] = (
        "F: E + log(produksjon)",
        fit_wls(df, ["api_gravity", "api2", "sulfur_pct", "is_bfoet", "log_production"]),
        df,
    )

    # === Sammenligningstabell ===
    print("=" * 70)
    print("MODELLSAMMENLIGNING")
    print("=" * 70)
    rows = [model_summary_row(name, m, len(d)) for _, (name, m, d) in fits.items()]
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # === Detaljert utskrift av hver modell ===
    for key, (name, m, d) in fits.items():
        print("=" * 70)
        print(f"  {name}")
        print(f"  N = {len(d)} felt, R² = {m.rsquared:.3f}, justert R² = {m.rsquared_adj:.3f}")
        print("=" * 70)
        # Vis koeffisient-tabellen kompakt.
        params = pd.DataFrame({
            "coef": m.params,
            "std_err": m.bse,
            "t": m.tvalues,
            "p": m.pvalues,
        })
        print(params.to_string(float_format=lambda x: f"{x:+.4f}"))
        print()

    # === Beste-modell residualer ===
    # Velg modellen med høyest justert R² automatisk.
    best_key = max(fits, key=lambda k: fits[k][1].rsquared_adj)
    name_best, model_best, df_best = fits[best_key]
    df_best = df_best.copy()
    df_best["predicted"] = model_best.fittedvalues
    df_best["residual"] = df_best["mean_diff"] - df_best["predicted"]
    df_best = df_best.sort_values("residual", ascending=False).reset_index(drop=True)

    print("=" * 70)
    print(f"RESIDUALER (beste modell etter justert R²: {name_best})")
    print("Positiv = handles dyrere enn modellen forventer")
    print("Negativ = handles billigere enn modellen forventer")
    print("=" * 70)
    cols_show = ["field", "api_gravity", "sulfur_pct", "is_bfoet",
                 "production_kbpd", "mean_diff", "predicted", "residual", "n_obs"]
    cols_show = [c for c in cols_show if c in df_best.columns]
    print(df_best[cols_show].to_string(
        index=False, float_format=lambda x: f"{x:+.2f}"))

    out_resid = OUT_DIR / f"model_{best_key}_residuals.csv"
    df_best[cols_show].to_csv(out_resid, index=False)
    print(f"\nResidualer lagret: {out_resid}")

    # === Plott: predikert vs. faktisk + residualer ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Venstre: predikert vs faktisk.
    ax = axes[0]
    ax.scatter(df_best["predicted"], df_best["mean_diff"],
               s=50 + df_best["n_obs"], alpha=0.65, color="steelblue",
               edgecolor="navy")
    for _, row in df_best.iterrows():
        ax.annotate(row["field"].title(),
                    (row["predicted"], row["mean_diff"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    lim_lo = min(df_best["predicted"].min(), df_best["mean_diff"].min()) - 1
    lim_hi = max(df_best["predicted"].max(), df_best["mean_diff"].max()) + 1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="gray", alpha=0.6,
            label="Perfekt prediksjon (y=x)")
    ax.set_xlabel("Predikert differensial (USD/fat)")
    ax.set_ylabel("Faktisk snitt-differensial (USD/fat)")
    ax.set_title(f"{name_best}\nR² = {model_best.rsquared:.3f}, just. R² = {model_best.rsquared_adj:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)

    # Høyre: residualer per felt, sortert.
    ax = axes[1]
    colors = ["seagreen" if r >= 0 else "indianred" for r in df_best["residual"]]
    ax.barh(df_best["field"].str.title(), df_best["residual"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Residual (USD/fat) — over/under det modellen forventer")
    ax.set_title("Residualer: hvilke felt avviker fra modellen?")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.suptitle("Regresjonsmodell: oljekvalitet → pris-differensial mot Brent",
                 fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "05_regression_model.png"
    fig.savefig(out_png, dpi=130)
    print(f"Plott lagret: {out_png}")


if __name__ == "__main__":
    main()
