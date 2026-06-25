"""
Script 44 — Sammenligning: Predikert AKRBP-pris vs. rapportert vs. Brent spot/futures

Grafen viser per kvartal (2020–2026):
  1. Modellpredikert realisert oljepris (AKRBP)
  2. AKRBP offisielt rapportert realisert pris
  3. Brent spot (kvartalsgjennomsnitt)
  4. Brent front-month futures / M1 (kvartalsgjennomsnitt)
  5. Brent 2-month futures / M2 (kvartalsgjennomsnitt)

Merk på Brent M1/M2:
  - M1 ≈ Brent spot (månedlig gjennomsnitt av front-month futures og spot er nesten identiske)
  - M2 = Brent spot + WTI-terminstruktur (M2-M1 slope) som proxy for Brent-terminstruktur

Korrelasjoner vises mot rapportert pris for å undersøke hvem som følger tettest.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
OUT_PNG = DATA_PROC / "44_price_comparison_brent_vs_model.png"

# ---------------------------------------------------------------------------
# AKRBP offisielt rapporterte priser (USD/boe), kilde: Aker BP kvartalsrapporter
# ---------------------------------------------------------------------------
AKRBP_REPORTED = {
    "2019-Q1": 66.3, "2019-Q2": 71.3, "2019-Q3": 61.6, "2019-Q4": 64.1,
    "2020-Q1": 46.9, "2020-Q2": 26.1, "2020-Q3": 42.9, "2020-Q4": 46.0,
    "2021-Q1": 60.3, "2021-Q2": 70.9, "2021-Q3": 73.1, "2021-Q4": 80.0,
    "2022-Q1": 100.9, "2022-Q2": 117.5, "2022-Q3": 101.1, "2022-Q4": 86.6,
    "2023-Q1": 78.4,  "2023-Q2": 76.8,  "2023-Q3": 87.6,  "2023-Q4": 83.6,
    "2024-Q1": 82.9,  "2024-Q2": 83.1,  "2024-Q3": 80.3,  "2024-Q4": 74.1,
    "2025-Q1": 75.0,  "2025-Q2": 66.9,  "2025-Q3": 70.3,  "2025-Q4": 63.1,
}


def quarter_to_yearq(row) -> str:
    return f"{int(row['year'])}-Q{int(row['quarter'])}"


def main():
    # -----------------------------------------------------------------------
    # 1. Last inn kvartalsvis modellpredikert pris (script 42 output)
    # -----------------------------------------------------------------------
    df_pred = pd.read_csv(DATA_PROC / "42_akrbp_quarterly_realized.csv")
    df_pred["qstr"] = df_pred["year"].astype(str) + "-Q" + df_pred["quarter"].astype(str)
    df_pred = df_pred[["qstr", "year", "quarter", "brent_q", "realized_pred"]].copy()
    df_pred["reported"] = df_pred["qstr"].map(AKRBP_REPORTED)

    # -----------------------------------------------------------------------
    # 2. Brent spot månedlig → kvartalsgjennomsnitt
    # -----------------------------------------------------------------------
    mc = pd.read_csv(DATA_PROC / "market_controls_monthly.csv")
    mc["year_month"] = mc["year_month"].astype(str)
    mc["year"] = mc["year_month"].str[:4].astype(int)
    mc["month"] = mc["year_month"].str[5:7].astype(int)
    mc["quarter"] = ((mc["month"] - 1) // 3 + 1).astype(int)
    brent_q = (mc.groupby(["year", "quarter"])["brent_price"]
               .mean().reset_index()
               .rename(columns={"brent_price": "brent_spot_q"}))
    brent_q["qstr"] = brent_q["year"].astype(str) + "-Q" + brent_q["quarter"].astype(str)

    # -----------------------------------------------------------------------
    # 3. WTI term structure → proxy for Brent M1/M2
    # -----------------------------------------------------------------------
    fc = pd.read_csv(DATA_PROC / "forward_curve_monthly.csv")
    fc["year_month"] = fc["year_month"].astype(str)
    fc["year"] = fc["year_month"].str[:4].astype(int)
    fc["month"] = fc["year_month"].str[5:7].astype(int)
    fc["quarter"] = ((fc["month"] - 1) // 3 + 1).astype(int)
    # Kvartalsgjennomsnitt av term-structure slope (M2 - M1)
    fc_q = (fc.groupby(["year", "quarter"])[["wti_m1", "wti_m2"]]
             .mean().reset_index())
    fc_q["wti_slope_m2m1"] = fc_q["wti_m2"] - fc_q["wti_m1"]
    fc_q["qstr"] = fc_q["year"].astype(str) + "-Q" + fc_q["quarter"].astype(str)

    # -----------------------------------------------------------------------
    # 4. Slå sammen
    # -----------------------------------------------------------------------
    df = df_pred.merge(brent_q[["qstr", "brent_spot_q"]], on="qstr", how="left")
    df = df.merge(fc_q[["qstr", "wti_slope_m2m1"]], on="qstr", how="left")

    # Brent M1 ≈ spot (front-month futures og spot er nesten identiske på månedlig basis)
    df["brent_m1_q"] = df["brent_spot_q"]   # = spot (se note i docstring)

    # Brent M2 ≈ spot + WTI term structure (M2-M1) som proxy
    df["brent_m2_q"] = df["brent_spot_q"] + df["wti_slope_m2m1"]

    # Bruk brent_q fra script 42 som primær Brent spot (konsistent med modellen)
    # brent_q er allerede kvartalsgjennomsnitt brukt i modellen
    df["brent_spot_model"] = df["brent_q"]  # fra script 42

    # Brent M2 re-basert på script 42 brent (for konsistens)
    df["brent_m2_q"] = df["brent_q"] + df["wti_slope_m2m1"]

    # -----------------------------------------------------------------------
    # 5. Filtrer til plottperiode: 2019-Q1 → siste kvartal med data
    # -----------------------------------------------------------------------
    df_plot = df[df["qstr"] >= "2019-Q1"].copy()

    # -----------------------------------------------------------------------
    # 6. Korrelasjoner mot rapportert pris
    # -----------------------------------------------------------------------
    df_corr = df_plot[df_plot["reported"].notna()].copy()
    corr_model  = df_corr["realized_pred"].corr(df_corr["reported"])
    corr_brent  = df_corr["brent_spot_model"].corr(df_corr["reported"])
    corr_m1     = df_corr["brent_m1_q"].corr(df_corr["reported"])
    corr_m2     = df_corr["brent_m2_q"].corr(df_corr["reported"])

    mae_model   = (df_corr["realized_pred"] - df_corr["reported"]).abs().mean()
    mae_brent   = (df_corr["brent_spot_model"] - df_corr["reported"]).abs().mean()
    mae_m1      = (df_corr["brent_m1_q"] - df_corr["reported"]).abs().mean()
    mae_m2      = (df_corr["brent_m2_q"] - df_corr["reported"]).abs().mean()

    print("=== Korrelasjoner mot AKRBP rapportert pris ===")
    print(f"  Modellpredikert:  corr={corr_model:.3f}, MAE={mae_model:.2f} USD/boe")
    print(f"  Brent spot:       corr={corr_brent:.3f}, MAE={mae_brent:.2f} USD/boe")
    print(f"  Brent M1:         corr={corr_m1:.3f}, MAE={mae_m1:.2f} USD/boe")
    print(f"  Brent M2:         corr={corr_m2:.3f}, MAE={mae_m2:.2f} USD/boe")

    # -----------------------------------------------------------------------
    # 7. Bygg graf
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11),
                                    gridspec_kw={"height_ratios": [3, 1.1]},
                                    facecolor="white")
    fig.suptitle("AKRBP realisert oljepris — Modell vs. rapportert vs. Brent spot/futures",
                 fontsize=14, fontweight="bold", y=0.99)

    x = list(range(len(df_plot)))
    xlabels = df_plot["qstr"].tolist()
    rep_mask = df_plot["reported"].notna()

    # --- Fargepalett ---
    c_model   = "#154360"   # mørk marineblå — modell
    c_report  = "#1D6A39"   # skoggrønn — rapportert (referanse)
    c_spot    = "#C0392B"   # rød — brent spot
    c_m1      = "#D35400"   # mørk oransje — M1 (= spot, ikke plottet separat)
    c_m2      = "#7D3C98"   # lilla — M2

    # Bakgrunnsfarger for perioder
    # Covid: 2020-Q1 → 2020-Q4
    # Ukraine/krig: 2022-Q1 → 2022-Q4
    for qstart, qend, label, col in [
        ("2020-Q1", "2020-Q4", "Covid-19", "#FADBD8"),
        ("2022-Q1", "2022-Q4", "Ukraina-krig", "#FDEBD0"),
    ]:
        if qstart in xlabels and qend in xlabels:
            xs = xlabels.index(qstart) - 0.5
            xe = xlabels.index(qend) + 0.5
            ax1.axvspan(xs, xe, color=col, alpha=0.5, zorder=0)
            ax1.text((xs + xe) / 2, 5, label, ha="center", va="bottom",
                     fontsize=8, color="#7B7D7D", style="italic")

    # Brent M2 (tynn stiplet linje, minst prominent)
    ax1.plot(x, df_plot["brent_m2_q"], color=c_m2, lw=1.4, ls=":",
             alpha=0.85, zorder=2,
             label=f"Brent M2 fut.  corr={corr_m2:.3f}  MAE={mae_m2:.1f} USD")

    # Brent spot (stiplet, litt tykkere)
    ax1.plot(x, df_plot["brent_q"], color=c_spot, lw=2.0, ls="--",
             alpha=0.9, zorder=3,
             label=f"Brent spot     corr={corr_brent:.3f}  MAE={mae_brent:.1f} USD")

    # Modellpredikert (solid, tykk)
    ax1.plot(x, df_plot["realized_pred"], color=c_model, lw=2.4,
             alpha=0.92, zorder=4,
             label=f"Modell pred.   corr={corr_model:.3f}  MAE={mae_model:.1f} USD")

    # AKRBP rapportert (solid med markører — øverst, referanselinje)
    rep_x = [i for i, m in enumerate(rep_mask) if m]
    rep_y = df_plot["reported"][rep_mask].values
    ax1.plot(rep_x, rep_y, color=c_report, lw=2.8, marker="o", ms=4.5,
             alpha=1.0, zorder=5,
             label=f"AKRBP rapportert  (referanse)")

    # Fyll mellom modell og rapportert
    y_model_full = df_plot["realized_pred"].values
    y_rep_full   = df_plot["reported"].values
    ax1.fill_between(x, y_model_full, y_rep_full,
                     where=rep_mask.values, alpha=0.08, color=c_model, zorder=1)

    # Grid og akser
    ax1.set_ylabel("USD / boe", fontsize=11, labelpad=8)
    ax1.set_xlim(-0.5, len(x) - 0.5)
    ax1.set_ylim(0, df_plot[["brent_q", "realized_pred", "reported"]].max().max() * 1.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax1.grid(axis="y", alpha=0.25, linestyle=":", color="gray")
    ax1.grid(axis="x", alpha=0.10, linestyle=":", color="gray")
    ax1.set_facecolor("#FAFAFA")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # Legend
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.92,
               title="Prisserie  (korrelasjon og MAE mot rapportert pris)",
               title_fontsize=9, edgecolor="#CCCCCC")

    # Stats-boks (høyre hjørne)
    bias_model = (df_plot["realized_pred"][rep_mask] - df_plot["reported"][rep_mask]).mean()
    bias_brent = (df_plot["brent_q"][rep_mask]       - df_plot["reported"][rep_mask]).mean()
    stats_txt = (
        f"Nøkkelstatistikk (mot rapportert pris)\n"
        f"─────────────────────────────────────\n"
        f"{'Serie':<18} {'Corr':>6}  {'MAE':>6}  {'Bias':>7}\n"
        f"{'Modell pred.':<18} {corr_model:>6.3f}  {mae_model:>5.1f}  {bias_model:>+6.1f}\n"
        f"{'Brent spot':<18} {corr_brent:>6.3f}  {mae_brent:>5.1f}  {bias_brent:>+6.1f}\n"
        f"{'Brent M2 fut.':<18} {corr_m2:>6.3f}  {mae_m2:>5.1f}  {(df_plot['brent_m2_q'][rep_mask]-df_plot['reported'][rep_mask]).mean():>+6.1f}"
    )
    ax1.text(0.99, 0.97, stats_txt, transform=ax1.transAxes, fontsize=8,
             va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#AAAAAA", alpha=0.92))

    # Metodenote
    note = (
        "⚑  Brent M1 ≈ spot (front-month futures og spot er nært identiske på månedlig basis) — vises ikke separat.  "
        "Brent M2 = Brent spot + WTI terminstruktur-slope (M2−M1) som proxy for ICE Brent M2."
    )
    ax1.text(0.01, -0.01, note, transform=ax1.transAxes, fontsize=7,
             color="#999999", va="top", style="italic")

    # -----------------------------------------------------------------------
    # Nedre panel: Prediksjonsfeil (søylediagram)
    # -----------------------------------------------------------------------
    df_plot["err_model"] = df_plot["realized_pred"] - df_plot["reported"]
    df_plot["err_brent"] = df_plot["brent_q"]       - df_plot["reported"]
    df_plot["err_m2"]    = df_plot["brent_m2_q"]    - df_plot["reported"]

    bar_w = 0.27
    rep_idx = [i for i, m in enumerate(rep_mask) if m]

    bars_model = ax2.bar([i - bar_w for i in rep_idx],
                         df_plot["err_model"].iloc[rep_idx],
                         bar_w, color=c_model, alpha=0.82, label=f"Modell   (MAE={mae_model:.1f})")
    bars_brent = ax2.bar(rep_idx,
                         df_plot["err_brent"].iloc[rep_idx],
                         bar_w, color=c_spot, alpha=0.82, label=f"Brent spot (MAE={mae_brent:.1f})")
    bars_m2    = ax2.bar([i + bar_w for i in rep_idx],
                         df_plot["err_m2"].iloc[rep_idx],
                         bar_w, color=c_m2, alpha=0.82, label=f"Brent M2   (MAE={mae_m2:.1f})")

    ax2.axhline(0, color="#333333", lw=0.9)
    ax2.set_ylabel("Feil (USD/boe)", fontsize=10, labelpad=8)
    ax2.set_xlabel("Kvartal", fontsize=10)
    ax2.set_xlim(-0.5, len(x) - 0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}"))
    ax2.grid(axis="y", alpha=0.25, linestyle=":", color="gray")
    ax2.set_facecolor("#FAFAFA")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=3,
               title="Prediksjonsfeil = Predikert − Rapportert", title_fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=0.5)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\n  Graf lagret: {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
