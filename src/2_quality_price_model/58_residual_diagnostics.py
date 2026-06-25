"""
Script 58 — Residual-diagnostikk for Brent-linked modellen.

FORMÅL:
  Identifiser SYSTEMATISKE feil i Modell B for å vite hvilke variabler
  som mest sannsynlig vil forbedre modellen. Den rigorøse økonometriske
  tilnærmingen: legg ikke til variabler tilfeldig, men gjør det basert
  på hvor modellen faktisk feiler.

ANALYSER:
  1. Residual-statistikk per GRADE — hvilke felt har vedvarende bias?
  2. Residual-statistikk per ÅR — temporal patterns
  3. Residual-statistikk per MÅNED — sesongmønster modellen ikke fanger
  4. Residual-statistikk per BRENT-REGIME (lav/medium/høy)
  5. Residual vs. NY potensiell variabel (test om noen forklarer residualene)
  6. Autokorrelasjon — Durbin-Watson per grade

DIAGNOSE-OUTPUT:
  data/processed/58_residual_diagnostics.csv
  data/processed/58_residual_diagnostics.png

ANBEFALINGER:
  Basert på funnene, presenterer hypotheser for hvilke variabler
  som mest sannsynlig vil forbedre modellen.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.stats.stattools import durbin_watson
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
MODEL_JSON   = PROC_DIR / "34b_brent_model.json"
PANEL_CSV    = PROC_DIR / "regression_panel.csv"
OUT_CSV      = PROC_DIR / "58_residual_diagnostics.csv"
OUT_PNG      = PROC_DIR / "58_residual_diagnostics.png"

WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}


def compute_residuals(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Beregn predikert differensial og residualer for hver observasjon.
    Filtrer til kun obs der alle model-features har gyldige verdier (samme
    som ved trening, der dropna ble brukt).
    """
    coefs = model["coefficients"]
    features = model["features"]

    # Sett opp region dummies (samme som ved trening)
    region_simple = {
        "North Sea": "NorthSea", "Norwegian Sea": "NorthSea", "Barents Sea": "NorthSea",
        "North America": "NorthAmerica", "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm", "Middle East": "MiddleEast",
        "West Africa": "WestAfrica", "North Africa": "NorthAfrica",
        "FSU": "FSU", "Asia-Pacific": "AsiaPac", "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg", drop_first=True, dtype=int)
    df = pd.concat([df, region_dums], axis=1)

    # Sørg for at alle features finnes som kolonner (legg til 0 hvis ikke)
    for f in features:
        if f not in df.columns:
            df[f] = 0

    # Dropp obs med NaN i ANY feature (samme som under modell-trening)
    df = df.dropna(subset=features + ["differential"]).copy()

    # Beregn prediksjon
    coef_vec = np.array([coefs.get("const", 0)] + [coefs.get(f, 0) for f in features])
    X = np.column_stack([np.ones(len(df))] + [df[f].astype(float).values for f in features])
    df["predicted"] = X @ coef_vec
    df["residual"]  = df["differential"] - df["predicted"]
    return df


def analyze_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Residualer per grade."""
    g = df.groupby("grade").agg(
        n=("residual", "count"),
        mean_resid=("residual", "mean"),
        std_resid=("residual", "std"),
        abs_mean=("residual", lambda x: x.abs().mean()),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
        max_abs=("residual", lambda x: x.abs().max()),
    ).round(2)
    g = g.sort_values("mean_resid", key=abs, ascending=False)
    return g


def analyze_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """Residualer per år og måned."""
    annual = df.groupby("year").agg(
        n=("residual", "count"),
        mean_resid=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
    ).round(2)
    seasonal = df.groupby("month").agg(
        n=("residual", "count"),
        mean_resid=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
    ).round(2)
    return annual, seasonal


def analyze_by_brent_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Residualer per Brent-regime: lav (<60), medium (60-80), høy (>80)."""
    df = df.copy()
    df["brent_regime"] = pd.cut(df["brent_price"],
                                 bins=[0, 50, 70, 90, 200],
                                 labels=["<50", "50-70", "70-90", ">90"])
    return df.groupby("brent_regime", observed=True).agg(
        n=("residual", "count"),
        mean_resid=("residual", "mean"),
        rmse=("residual", lambda x: np.sqrt((x**2).mean())),
        mean_brent=("brent_price", "mean"),
    ).round(2)


def test_missing_variable(df: pd.DataFrame, candidate_col: str) -> dict:
    """Test om en kandidat-variabel korrelerer med residualene."""
    if candidate_col not in df.columns:
        return {"available": False}
    sub = df.dropna(subset=["residual", candidate_col])
    if len(sub) < 100:
        return {"available": False}
    corr = sub["residual"].corr(sub[candidate_col])
    # Enkel R²: hvor mye av residual-variansen kan forklares?
    from scipy.stats import pearsonr
    r, p = pearsonr(sub["residual"], sub[candidate_col])
    return {
        "available": True,
        "n":         len(sub),
        "corr":      r,
        "r_squared": r ** 2,
        "p_value":   p,
    }


def main():
    print("=" * 75)
    print("  SCRIPT 58: Residual-diagnostikk for Modell B")
    print("=" * 75)

    print("\n[1] Laster modell og panel-data...")
    model = json.loads(MODEL_JSON.read_text())
    panel = pd.read_csv(PANEL_CSV)
    panel["date"] = pd.to_datetime(panel["date_str"])
    panel["year"]  = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month

    # Filtrer til Brent-linked grades (modellens treningsunivers)
    brent_panel = panel[~panel["grade"].isin(WTI_LINKED)].copy()
    print(f"  Modell: {model['model_name']}")
    print(f"  OOT R²: {model['metrics']['r2_oot']}, RMSE: {model['metrics']['rmse']}")
    print(f"  Brent-panel: {len(brent_panel)} obs, {brent_panel['grade'].nunique()} grades")

    print("\n[2] Beregner residualer for alle observasjoner...")
    brent_panel = compute_residuals(brent_panel, model)
    brent_panel = brent_panel.dropna(subset=["residual"])
    print(f"  Total residualer: {len(brent_panel):,}")
    print(f"  Snitt residual: {brent_panel['residual'].mean():+.3f} (forventet ≈ 0)")
    print(f"  RMSE:           {np.sqrt((brent_panel['residual']**2).mean()):.3f}")

    # ── ANALYSE 1: Per grade ─────────────────────────────────────────────────
    print(f"\n[3] RESIDUALER PER GRADE — bias-rangering")
    print(f"{'─' * 75}")
    grade_diag = analyze_by_grade(brent_panel)
    print(f"  {'Grade':<22} {'N':>5} {'Mean':>7} {'Std':>6} {'RMSE':>6} {'MaxAbs':>7}")
    print(f"  {'-' * 60}")
    for g, row in grade_diag.head(8).iterrows():
        print(f"  {g:<22} {int(row['n']):>5} {row['mean_resid']:>+7.2f} "
              f"{row['std_resid']:>6.2f} {row['rmse']:>6.2f} {row['max_abs']:>7.2f}")
    print(f"  ... (mellom)")
    for g, row in grade_diag.tail(5).iterrows():
        print(f"  {g:<22} {int(row['n']):>5} {row['mean_resid']:>+7.2f} "
              f"{row['std_resid']:>6.2f} {row['rmse']:>6.2f} {row['max_abs']:>7.2f}")

    # ── ANALYSE 2: Temporal ─────────────────────────────────────────────────
    print(f"\n[4] RESIDUALER PER ÅR")
    print(f"{'─' * 75}")
    annual, seasonal = analyze_by_period(brent_panel)
    print(f"  {'År':<6} {'N':>5} {'Mean resid':>11} {'RMSE':>6}")
    print(f"  {'-' * 40}")
    for yr, row in annual.iterrows():
        flag = " ⚠" if abs(row["mean_resid"]) > 0.5 else ""
        print(f"  {yr:<6} {int(row['n']):>5} {row['mean_resid']:>+11.3f} {row['rmse']:>6.2f}{flag}")

    print(f"\n[5] RESIDUALER PER MÅNED (sesongmønster modellen ikke fanger)")
    print(f"{'─' * 75}")
    print(f"  {'Mnd':<5} {'N':>5} {'Mean':>7} {'RMSE':>6}")
    print(f"  {'-' * 32}")
    for mo, row in seasonal.iterrows():
        bar = "█" * int(abs(row["mean_resid"]) * 5)
        side = "+" if row["mean_resid"] > 0 else "-"
        print(f"  {int(mo):<5} {int(row['n']):>5} {row['mean_resid']:>+7.3f} {row['rmse']:>6.2f}  {side}{bar}")

    # ── ANALYSE 3: Brent-regime ─────────────────────────────────────────────
    print(f"\n[6] RESIDUALER PER BRENT-REGIME")
    print(f"{'─' * 75}")
    brent_diag = analyze_by_brent_regime(brent_panel)
    print(f"  {'Regime':<10} {'N':>5} {'Mean':>7} {'RMSE':>6} {'snitt Brent':>13}")
    print(f"  {'-' * 50}")
    for reg, row in brent_diag.iterrows():
        flag = " ⚠" if abs(row["mean_resid"]) > 0.3 else ""
        print(f"  {str(reg):<10} {int(row['n']):>5} {row['mean_resid']:>+7.3f} "
              f"{row['rmse']:>6.2f} {row['mean_brent']:>13.1f}{flag}")

    # ── ANALYSE 4: Test om eksisterende kandidat-variabler forklarer residualer ─
    print(f"\n[7] FORKLAREKRAFT AV EKSISTERENDE 'UBRUKTE' VARIABLER")
    print(f"   (variabler i panelet men IKKE i modellen — kan de forklare residualene?)")
    print(f"{'─' * 75}")
    candidates = [
        "us_crude_exports_kbpd",
        "vix",
        "us_refinery_util_pct",
        "fc_slope_4m",
        "wti_brent_spread",
        "jet_crack_brent",
        "diesel_gasoil_pct",
        "log_v_ni",
    ]
    candidates = [c for c in candidates if c not in model["features"]]
    print(f"  {'Kandidat':<35} {'Corr m/ residual':>17} {'R²':>7} {'p':>7}")
    print(f"  {'-' * 70}")
    for cand in candidates:
        res = test_missing_variable(brent_panel, cand)
        if res["available"]:
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else \
                  "*" if res["p_value"] < 0.05 else ""
            print(f"  {cand:<35} {res['corr']:>+17.3f} {res['r_squared']:>7.3f} "
                  f"{res['p_value']:>5.3f} {sig}")

    # ── Lagre detaljert tabell ──────────────────────────────────────────────
    print(f"\n[8] Lagrer detaljert residual-tabell...")
    grade_diag.to_csv(OUT_CSV)
    print(f"  ✓ {OUT_CSV.name}")

    # ── Plott: 4-panel diagnostikk ──────────────────────────────────────────
    print(f"\n[9] Genererer diagnostisk figur...")
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])  # Per grade
    ax2 = fig.add_subplot(gs[1, 0])  # Per år
    ax3 = fig.add_subplot(gs[1, 1])  # Per måned
    ax4 = fig.add_subplot(gs[2, 0])  # Per Brent-regime
    ax5 = fig.add_subplot(gs[2, 1])  # Residual-distribusjon

    # Panel 1: Per grade
    gd = grade_diag.sort_values("mean_resid")
    colors = ["#C0392B" if v < -0.3 else "#27AE60" if v > 0.3 else "#888"
              for v in gd["mean_resid"]]
    ax1.barh(range(len(gd)), gd["mean_resid"], color=colors, alpha=0.85)
    ax1.set_yticks(range(len(gd)))
    ax1.set_yticklabels(gd.index, fontsize=8)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.axvline(-0.5, color="red", lw=0.5, ls=":", alpha=0.5)
    ax1.axvline(0.5, color="red", lw=0.5, ls=":", alpha=0.5)
    ax1.set_xlabel("Mean residual (USD/bbl) — neg = modell overpredikerer", fontsize=9)
    ax1.set_title("Residual per grade (rødt = systematisk bias > ±0.5 USD/bbl)",
                  fontsize=10, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Panel 2: Per år
    yrs = annual.index
    ax2.bar(yrs, annual["mean_resid"], color="#2980B9", alpha=0.7, label="Mean residual")
    ax2_t = ax2.twinx()
    ax2_t.plot(yrs, annual["rmse"], color="#C0392B", lw=2, marker="o", label="RMSE")
    ax2.set_xlabel("År", fontsize=9)
    ax2.set_ylabel("Mean residual", fontsize=9, color="#2980B9")
    ax2_t.set_ylabel("RMSE", fontsize=9, color="#C0392B")
    ax2.set_title("Per år — temporal stabilitet", fontsize=10, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Per måned
    months = seasonal.index
    ax3.bar(months, seasonal["mean_resid"],
             color=["#3498DB" if v > 0 else "#E74C3C" for v in seasonal["mean_resid"]],
             alpha=0.8)
    ax3.set_xticks(months)
    ax3.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax3.set_xlabel("Måned", fontsize=9)
    ax3.set_ylabel("Mean residual (USD/bbl)", fontsize=9)
    ax3.set_title("Per måned — sesongstøy modellen ikke fanger",
                  fontsize=10, fontweight="bold")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4: Per Brent-regime
    bd = brent_diag
    ax4.bar(range(len(bd)), bd["mean_resid"],
             color=["#566573", "#2980B9", "#16A085", "#A93226"], alpha=0.8)
    ax4.set_xticks(range(len(bd)))
    ax4.set_xticklabels([str(r) for r in bd.index])
    ax4.set_xlabel("Brent-regime (USD/bbl)", fontsize=9)
    ax4.set_ylabel("Mean residual (USD/bbl)", fontsize=9)
    ax4.set_title("Per Brent-regime — regime-avhengig bias",
                  fontsize=10, fontweight="bold")
    ax4.axhline(0, color="black", lw=0.5)
    for i, (idx, row) in enumerate(bd.iterrows()):
        ax4.text(i, row["mean_resid"] + (0.05 if row["mean_resid"] > 0 else -0.08),
                  f"n={int(row['n'])}", ha="center", fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # Panel 5: Residual-distribusjon
    ax5.hist(brent_panel["residual"], bins=60, color="#2980B9", alpha=0.7, edgecolor="white")
    ax5.axvline(0, color="black", lw=1)
    ax5.axvline(brent_panel["residual"].mean(), color="red", lw=1.5, ls="--",
                 label=f"Mean: {brent_panel['residual'].mean():+.2f}")
    ax5.set_xlabel("Residual (USD/bbl)", fontsize=9)
    ax5.set_ylabel("Antall obs", fontsize=9)
    ax5.set_title("Residual-distribusjon (skal være ca. normal rundt 0)",
                  fontsize=10, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("Diagnostikk: Hvor og når feiler Modell B?",
                  fontsize=13, fontweight="bold", y=0.995)

    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {OUT_PNG.name}")

    # ── Hypoteser for forbedring ────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  HYPOTESER — hva mangler basert på diagnostikken?")
    print(f"{'=' * 75}")

    # Sjekk om noen grades har stor positiv/negativ bias
    big_bias = grade_diag[abs(grade_diag["mean_resid"]) > 0.5]
    if len(big_bias) > 0:
        print(f"\n  ▸ {len(big_bias)} grades har systematisk bias > ±0.5 USD/bbl:")
        for g, row in big_bias.iterrows():
            direction = "OVER-predikerer" if row["mean_resid"] < 0 else "UNDER-predikerer"
            print(f"    · {g}: modellen {direction} med {abs(row['mean_resid']):.2f} USD/bbl "
                  f"({int(row['n'])} obs)")

    # Sjekk om noen år har stor bias
    big_year = annual[abs(annual["mean_resid"]) > 0.5]
    if len(big_year) > 0:
        print(f"\n  ▸ {len(big_year)} år har systematisk bias > ±0.5:")
        for yr, row in big_year.iterrows():
            print(f"    · {yr}: mean residual {row['mean_resid']:+.2f}")

    # Sjekk om noen Brent-regimer har bias
    big_regime = brent_diag[abs(brent_diag["mean_resid"]) > 0.3]
    if len(big_regime) > 0:
        print(f"\n  ▸ Brent-regime-avhengig bias:")
        for reg, row in big_regime.iterrows():
            print(f"    · {reg} USD/bbl: residual {row['mean_resid']:+.2f}")


if __name__ == "__main__":
    main()
