"""
Script 56 — Visualisering: forward grade-pris-prognoser per scenario

Designed for IB/ER-presentasjon. Viser hvordan kombinasjonen av Brent-scenarier
+ produkt-demand-scenarier påvirker forventet realisert pris for utvalgte
NCS-grades.

LAYOUT (3×2 grid):
  Topp:    Stage 1-output — predikerte crack spreads forward 2026-2027
  Midten:  Stage 2-output — predikerte grade-differensialer per scenario
  Bunn:    Slutt-resultat — realisert pris per grade × scenario

OUTPUT:
  data/processed/56_forward_forecast.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
FORECAST_CSV = PROC_DIR / "55_forward_forecast.csv"
OUT_PNG      = PROC_DIR / "56_forward_forecast.png"


SCENARIO_COLORS = {
    "Base × Base":              "#2980B9",
    "Base × DieselJet-Up":      "#27AE60",
    "Base × Gasoline-Down":     "#F39C12",
    "Bull × All-Demand-Up":     "#A93226",
    "Bear × Gasoline-Down":     "#566573",
}

FOCUS_GRADES_PLOT = [
    "Alvheim", "Skarv", "Johan Sverdrup", "Bonny Light",
    "Ekofisk", "Heidrun",
]


def main():
    df = pd.read_csv(FORECAST_CSV)
    df["date"] = pd.to_datetime(df["date"])

    fig = plt.figure(figsize=(16, 14), facecolor="white")
    fig.suptitle(
        "Forward Crude-Pris Prognose 2026–2027\n"
        "Stage 1 (Demand → Crack-spreads) × Stage 2 (Crack → grade-pris)",
        fontsize=14, fontweight="bold", color="#1A1A2E", y=0.99
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.28)

    # ── Panel 1: Crack spread forecast under demand-scenarier (Base Brent) ─
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    base_brent_only = df[df["brent_scen"] == "Base"]
    crack_summary = (base_brent_only.groupby(["date", "demand_scen"])
                        .agg(diesel_crack=("diesel_crack","first"),
                              jet_crack=("jet_crack","first"),
                              gasoline_crack=("gasoline_crack","first"))
                        .reset_index())

    for d_scen in ["Base", "DieselJet-Up", "Gasoline-Down", "All-Demand-Up"]:
        sub = crack_summary[crack_summary["demand_scen"] == d_scen].sort_values("date")
        if len(sub):
            ax1.plot(sub["date"], sub["diesel_crack"], lw=2.0, marker="o", ms=3,
                      label=d_scen, alpha=0.85)
            ax2.plot(sub["date"], sub["jet_crack"], lw=2.0, marker="o", ms=3,
                      label=d_scen, alpha=0.85)

    for ax, title in [(ax1, "Diesel crack vs. Brent"),
                       (ax2, "Jet fuel crack vs. Brent")]:
        ax.set_title(f"Stage 1: {title}", fontsize=11, fontweight="bold")
        ax.set_ylabel("USD/bbl", fontsize=9)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(fontsize=8, loc="best", framealpha=0.95)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    # ── Panel 2: Grade-differensial per scenario (Base Brent, 6 grader) ─────
    ax3 = fig.add_subplot(gs[1, :])

    base_b = df[df["brent_scen"] == "Base"]
    grade_summary = (base_b.groupby(["grade", "demand_scen"])
                            .agg(diff=("differential","mean"))
                            .reset_index())

    grades_sorted = FOCUS_GRADES_PLOT
    x = np.arange(len(grades_sorted))
    width = 0.20

    for i, d_scen in enumerate(["Base", "DieselJet-Up", "Gasoline-Down", "All-Demand-Up"]):
        vals = []
        for g in grades_sorted:
            r = grade_summary[(grade_summary["grade"] == g) & (grade_summary["demand_scen"] == d_scen)]
            vals.append(r.iloc[0]["diff"] if len(r) else 0)
        ax3.bar(x + i*width - 1.5*width, vals, width, label=d_scen, alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels(grades_sorted, fontsize=9, rotation=15)
    ax3.set_ylabel("Differensial vs. Brent (USD/bbl)", fontsize=9)
    ax3.set_title("Stage 2: Predikert grade-differensial 2026–2027 (Brent baseline-scenario)",
                  fontsize=11, fontweight="bold")
    ax3.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax3.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.95)
    ax3.grid(axis="y", alpha=0.3, linestyle=":")
    for s in ["top", "right"]:
        ax3.spines[s].set_visible(False)

    # ── Panel 3: Realisert pris under combo Brent × Demand scenarier ───────
    ax4 = fig.add_subplot(gs[2, :])

    combo_summary = (df.groupby(["grade", "scenario"])
                        .agg(realized=("realized_pred","mean"))
                        .reset_index())

    show_scenarios = ["Bear × Gasoline-Down", "Base × Base",
                      "Base × DieselJet-Up", "Bull × All-Demand-Up"]

    x = np.arange(len(grades_sorted))
    width = 0.21
    for i, scen in enumerate(show_scenarios):
        vals = []
        for g in grades_sorted:
            r = combo_summary[(combo_summary["grade"] == g) & (combo_summary["scenario"] == scen)]
            vals.append(r.iloc[0]["realized"] if len(r) else 0)
        color = SCENARIO_COLORS.get(scen, "#888")
        bars = ax4.bar(x + i*width - 1.5*width, vals, width, label=scen,
                        color=color, alpha=0.85)
        # Annoter
        for bar, v in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      f"{v:.1f}", ha="center", fontsize=7.5,
                      fontweight="bold", color=color)

    ax4.set_xticks(x)
    ax4.set_xticklabels(grades_sorted, fontsize=10, rotation=15, fontweight="bold")
    ax4.set_ylabel("Realisert oljepris (USD/bbl)", fontsize=9)
    ax4.set_title("Realisert pris forecast 2026–2027 — Brent × Demand-scenarier kombinert",
                   fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9, ncol=4, loc="lower left", framealpha=0.95)
    ax4.grid(axis="y", alpha=0.3, linestyle=":")
    ax4.set_ylim(50, max(combo_summary["realized"]) * 1.1)
    for s in ["top", "right"]:
        ax4.spines[s].set_visible(False)

    # Fotnote
    fig.text(0.02, 0.005,
             "Kilder: STEO (US produkt-demand & stocks med 2-års forward forecast), Equinor crude assays, vår Brent-linked OLS-modell.\n"
             "Metodologi: Stage 1 = OLS (demand + stocks + refinery util → crack spread); Stage 2 = vår eksisterende Modell B (crack + assay → grade-diff).\n"
             "OBS: Demand-elastisitetene i Stage 1 er hovedsakelig drevet av stocks-respons. Forecasts antar konstant raffinerings-utilisering 92%.",
             fontsize=7.5, color="#777", style="italic")

    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Figur lagret: {OUT_PNG}")

    # ── Skriv ut nøkkeltall ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KEY FORECASTS — Aker BP-relevante grades (snitt 2026-2027)")
    print(f"{'='*70}")
    print(f"\n  Bull × All-Demand-Up vs. Bear × Gasoline-Down spread per grade:")
    for g in ["Alvheim", "Skarv", "Johan Sverdrup", "Ekofisk", "Grane", "Heidrun"]:
        bull = combo_summary[(combo_summary["grade"]==g) & (combo_summary["scenario"]=="Bull × All-Demand-Up")]
        bear = combo_summary[(combo_summary["grade"]==g) & (combo_summary["scenario"]=="Bear × Gasoline-Down")]
        if len(bull) and len(bear):
            b, br = bull.iloc[0]["realized"], bear.iloc[0]["realized"]
            print(f"    {g:<20} Bull ${b:.2f}  vs. Bear ${br:.2f}  →  spread ${b-br:.2f}")


if __name__ == "__main__":
    main()
