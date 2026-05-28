"""
Steg 8: Diagnostikk + lagre modell + predict-funksjon.

Tre deler:
  1. DIAGNOSTIKK av modell E (beste modell):
     - Residualer vs predikert (sjekker om feilene er sympatisk fordelt)
     - Q-Q-plott (sjekker normalfordeling av residualene)
  2. JACKKNIFE: ta ut ett felt om gangen, refit, sjekk om koeffisientene
     endres dramatisk. Hvis Volve (5 obs) ene kontrollerer hele modellen,
     bør vi vite det.
  3. PREDICT-funksjon + lagring av modell:
     - Eksporter koeffisienter til JSON så modellen kan brukes i andre script.
     - Vis et eksempel: "Hva ville modellen si om et nytt hypotetisk felt?"
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_CSV = PROJECT_ROOT / "data" / "raw" / "crude_quality.csv"
DIFF_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "model"

FEATURES_E = ["api_gravity", "api2", "sulfur_pct", "is_bfoet"]


def build_dataset() -> pd.DataFrame:
    quality = pd.read_csv(QUALITY_CSV)
    diff = pd.read_csv(DIFF_CSV)
    quality["field"] = quality["field"].str.upper().str.strip()
    diff["field"] = diff["field"].str.upper().str.strip()
    agg = diff.groupby("field").agg(
        mean_diff=("differential_usd", "mean"),
        std_diff=("differential_usd", "std"),
        n_obs=("differential_usd", "size"),
    ).reset_index()
    df = quality.merge(agg, on="field", how="inner")
    df["api2"] = df["api_gravity"] ** 2
    return df


def fit_model_E(df: pd.DataFrame):
    X = sm.add_constant(df[FEATURES_E])
    return sm.WLS(df["mean_diff"], X, weights=df["n_obs"]).fit()


def diagnostics_plot(model, df, out_path: Path) -> None:
    """To diagnostiske plott side om side."""
    fitted = model.fittedvalues
    resid = df["mean_diff"] - fitted

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1) Residual vs fitted — bør se 'tilfeldig' ut, ingen mønster.
    ax = axes[0]
    ax.scatter(fitted, resid, s=50 + df["n_obs"], alpha=0.65,
               color="steelblue", edgecolor="navy")
    for _, row in df.iterrows():
        ax.annotate(row["field"].title(),
                    (row.get("predicted", model.fittedvalues[row.name]),
                     row["mean_diff"] - model.fittedvalues[row.name]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Predikert differensial (USD/fat)")
    ax.set_ylabel("Residual (faktisk − predikert)")
    ax.set_title("Residualer vs. predikert\n(bør ligne en sky uten mønster)")
    ax.grid(True, alpha=0.3)

    # 2) Q-Q-plott — bør ligge nær diagonalen for normalfordelte residualer.
    ax = axes[1]
    # Bruk vektede residualer for ærlig sammenligning.
    weighted_resid = resid * np.sqrt(df["n_obs"])
    weighted_resid = (weighted_resid - weighted_resid.mean()) / weighted_resid.std()
    stats.probplot(weighted_resid, dist="norm", plot=ax)
    ax.set_title("Q-Q-plott av standardiserte residualer\n"
                 "(bør ligge på diagonalen hvis normalfordelt)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Diagnostikk for modell E (B + BFOET)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Diagnostikk-plott lagret: {out_path}")


def jackknife(df: pd.DataFrame) -> pd.DataFrame:
    """Refit uten hvert enkelt felt, sammenlign koeffisientene."""
    full = fit_model_E(df)
    full_params = full.params

    rows = []
    for idx, dropped_field in enumerate(df["field"].tolist()):
        df_sub = df.drop(idx).reset_index(drop=True)
        m = fit_model_E(df_sub)
        rows.append({
            "dropped_field": dropped_field,
            "n_obs_dropped": int(df.loc[idx, "n_obs"]),
            **{f"coef_{k}": float(v) for k, v in m.params.items()},
            "r2": float(m.rsquared),
            "r2_adj": float(m.rsquared_adj),
        })
    jk = pd.DataFrame(rows)

    # Hvor mye varierer hver koeffisient når vi tar ut ett felt?
    coef_cols = [c for c in jk.columns if c.startswith("coef_")]
    print("\n=== Jackknife: spread i koeffisienter når vi tar ut ett felt ===")
    print(f"{'Variabel':<24}{'full-fit':>10}{'min':>10}{'max':>10}{'spread':>10}")
    for col in coef_cols:
        var = col.replace("coef_", "")
        full_v = full_params[var]
        jk_min = jk[col].min()
        jk_max = jk[col].max()
        spread = jk_max - jk_min
        print(f"{var:<24}{full_v:>+10.3f}{jk_min:>+10.3f}{jk_max:>+10.3f}{spread:>10.3f}")

    # Hvilket felt påvirker mest? Maks endring i en hvilken som helst koeffisient.
    print("\n=== Topp 5 mest innflytelsesrike felt (størst koeffisient-skifte) ===")
    influence = jk.copy()
    influence["max_abs_shift"] = influence[coef_cols].apply(
        lambda row: max(abs(row[c] - full_params[c.replace('coef_', '')]) for c in coef_cols),
        axis=1,
    )
    top = influence.sort_values("max_abs_shift", ascending=False).head(5)
    print(top[["dropped_field", "n_obs_dropped", "max_abs_shift", "r2_adj"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    return jk


def save_model(model, df, path: Path) -> None:
    """Lagre koeffisienter + metadata som JSON, så modellen kan brukes senere."""
    payload = {
        "model_name": "E_bfoet",
        "description": "diff_USD = β0 + β1·API + β2·API² + β3·svovel + β4·BFOET-flagg",
        "features": ["intercept"] + FEATURES_E,
        "coefficients": {k: float(v) for k, v in model.params.items()},
        "std_errors": {k: float(v) for k, v in model.bse.items()},
        "r_squared": float(model.rsquared),
        "r_squared_adjusted": float(model.rsquared_adj),
        "n_fields": int(len(df)),
        "n_monthly_obs": int(df["n_obs"].sum()),
        "training_period": {
            "first_year": 2012,
            "last_year": 2025,
        },
        "training_fields": sorted(df["field"].tolist()),
        "notes": "Vekt = antall månedlige observasjoner. Brent benchmark = "
                 "Petroleumsprisrådets daglige Brent-pris.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModell lagret: {path}")


def predict_differential(api: float, sulfur: float, is_bfoet: int,
                         coefs: dict[str, float]) -> float:
    """Beregn predikert differensial gitt kvalitet + BFOET-medlemskap.

    Kan importeres fra dette scriptet og brukes i analysene dine fremover:
        from scripts.07_diagnostics_and_predict import predict_differential
    """
    return (
        coefs["const"]
        + coefs["api_gravity"] * api
        + coefs["api2"] * api ** 2
        + coefs["sulfur_pct"] * sulfur
        + coefs["is_bfoet"] * is_bfoet
    )


def example_predictions(model) -> None:
    """Vis tre realistiske 'hva-hvis'-eksempler."""
    coefs = {k: float(v) for k, v in model.params.items()}
    print("\n=== Predict-eksempler ===")
    examples = [
        ("Hypotetisk lett+søt (API=38, svovel=0.10), ikke BFOET",  38.0, 0.10, 0),
        ("Tilsvarende, men BFOET-medlem",                            38.0, 0.10, 1),
        ("Tung sur (API=22, svovel=0.90)",                           22.0, 0.90, 0),
        ("Kondensat (API=50, svovel=0.05)",                          50.0, 0.05, 0),
        ("Sverdrup-aktig (API=28, svovel=0.80)",                     28.0, 0.80, 0),
    ]
    for label, api, svv, bfoet in examples:
        pred = predict_differential(api, svv, bfoet, coefs)
        print(f"  {label}")
        print(f"    -> predikert differensial: {pred:+.2f} USD/fat\n")


def main() -> None:
    df = build_dataset()
    model = fit_model_E(df)

    print("=== Modell E (beste modell) ===")
    print(f"R² = {model.rsquared:.3f}, justert R² = {model.rsquared_adj:.3f}")
    print(f"N felt = {len(df)}, sum vekter (månedsobs.) = {df['n_obs'].sum():,}")

    # 1) Diagnostikk
    df_with_pred = df.copy()
    df_with_pred["predicted"] = model.fittedvalues
    diagnostics_plot(model, df_with_pred, OUT_DIR / "07_diagnostics.png")

    # 2) Jackknife
    jk = jackknife(df)
    jk.to_csv(OUT_DIR / "07_jackknife.csv", index=False)
    print(f"Jackknife-tabell lagret: {OUT_DIR / '07_jackknife.csv'}")

    # 3) Lagre modell + eksempel-prediksjoner
    save_model(model, df, MODEL_DIR / "model_E_bfoet.json")
    example_predictions(model)


if __name__ == "__main__":
    main()
