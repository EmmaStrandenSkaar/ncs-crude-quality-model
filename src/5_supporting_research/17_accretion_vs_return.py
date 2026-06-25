"""
17_accretion_vs_return.py
Scatter: EV/2P-akkresjon vs 50-dagers kursutvikling etter hvert oppkjøp.

Tesen: Jo større rabatt (akkresjon) paa maalets reserver vs Aker BPs egne,
jo bedre mottok markedet dealen. Ekstrapolerer hva en OKEA-deal ville gitt.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRENT_CSV    = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

C_BG     = "#f7f9fc"
C_REG    = "#2c3e50"
C_ACC    = "#27ae60"
C_PRED   = "#e74c3c"

# ── Deal-data (fra script 14 + 16) ────────────────────────────────────────
# accretion = Aker BPs EV/2P paa deal-dato minus maalets implied EV/2P
# Positivt tall = "billige reserver" = akkretivt
DEALS = [
    dict(name="BP Norge",      date="2016-06-10", akrbp_ev2p=8.0,  target_ev2p=4.4,
         color="#8e44ad",   value_bn=1.44, brent=50),
    dict(name="Hess Norge",    date="2017-05-26", akrbp_ev2p=18.0, target_ev2p=13.3,
         color="#2980b9",   value_bn=2.0,  brent=52),
    dict(name="Total E&P",     date="2018-07-31", akrbp_ev2p=17.0, target_ev2p=2.5,
         color="#16a085",   value_bn=0.205, brent=75),
    dict(name="King Lear",     date="2018-10-15", akrbp_ev2p=17.0, target_ev2p=3.2,
         color="#1abc9c",   value_bn=0.250, brent=81),
    dict(name="Lundin Energy", date="2021-12-21", akrbp_ev2p=17.0, target_ev2p=21.9,
         color="#c0392b",   value_bn=14.0,  brent=73),
]

# Potensielle maal (for ekstrapolering)
FUTURE_TARGETS = [
    dict(name="OKEA",              akrbp_ev2p=19.7, target_ev2p=4.3,  color="#e74c3c"),
    dict(name="INPEX Valhall 10%", akrbp_ev2p=19.7, target_ev2p=11.4, color="#d4ac0d"),
    dict(name="Harbour NCS",       akrbp_ev2p=19.7, target_ev2p=5.8,  color="#16a085"),
    dict(name="Vaar Energi",       akrbp_ev2p=19.7, target_ev2p=13.6, color="#2980b9"),
]


def fetch_50d_return(ticker, deal_date):
    """Hent 50-dagers avkastning etter deal-dato."""
    data = yf.download(ticker, start="2016-01-01", end="2026-05-10",
                       progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(level=1, axis=1)
    close = data["Close"].dropna()
    close.index = pd.to_datetime(close.index)

    dt = pd.Timestamp(deal_date)
    idx = close.index.get_indexer([dt], method="ffill")[0]
    if idx < 0:
        idx = close.index.get_indexer([dt], method="bfill")[0]

    end_idx = min(len(close) - 1, idx + 50)
    base = close.iloc[idx]
    end = close.iloc[end_idx]
    return ((end / base) - 1) * 100


def fetch_brent_50d_return(deal_date):
    """Hent 50-dagers Brent-avkastning for oljepris-justering."""
    brent = pd.read_csv(BRENT_CSV, parse_dates=["date"]).set_index("date")["brent_usd"]
    dt = pd.Timestamp(deal_date)
    idx = brent.index.get_indexer([dt], method="ffill")[0]
    end_idx = min(len(brent) - 1, idx + 50)
    base = brent.iloc[idx]
    end = brent.iloc[end_idx]
    return ((end / base) - 1) * 100


def make_analysis():
    if not HAS_YF:
        print("yfinance trengs!")
        return

    print("Laster AKRBP.OL data...")
    # Beregn 50-dagers returns
    for d in DEALS:
        d["return_50d"] = fetch_50d_return("AKRBP.OL", d["date"])
        d["brent_50d"]  = fetch_brent_50d_return(d["date"])
        d["accretion"]  = d["akrbp_ev2p"] - d["target_ev2p"]
        d["excess_return"] = d["return_50d"] - d["brent_50d"]
        print(f"  {d['name']}: akkresjon ${d['accretion']:.1f}/boe, "
              f"retur {d['return_50d']:+.1f}%, "
              f"Brent {d['brent_50d']:+.1f}%, "
              f"meravkastning {d['excess_return']:+.1f}%")

    # ── Regresjon ──────────────────────────────────────────────────────────
    x = np.array([d["accretion"] for d in DEALS])
    y_raw = np.array([d["return_50d"] for d in DEALS])
    y_excess = np.array([d["excess_return"] for d in DEALS])

    # Lineær regresjon paa raa-avkastning
    coeffs_raw = np.polyfit(x, y_raw, 1)
    slope_raw, intercept_raw = coeffs_raw
    r2_raw = 1 - np.sum((y_raw - np.polyval(coeffs_raw, x))**2) / np.sum((y_raw - y_raw.mean())**2)

    # Lineær regresjon paa meravkastning (justert for Brent)
    coeffs_ex = np.polyfit(x, y_excess, 1)
    slope_ex, intercept_ex = coeffs_ex
    r2_ex = 1 - np.sum((y_excess - np.polyval(coeffs_ex, x))**2) / np.sum((y_excess - y_excess.mean())**2)

    print(f"\n  Raa-regresjon: y = {slope_raw:.2f}x + {intercept_raw:.1f}, R² = {r2_raw:.2f}")
    print(f"  Meravkastning-regresjon: y = {slope_ex:.2f}x + {intercept_ex:.1f}, R² = {r2_ex:.2f}")

    # ── Figur ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor=C_BG)

    fig.text(0.04, 0.975,
             "Aker BP — Akkresjon vs kursutvikling etter oppkjøp",
             fontsize=19, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.04, 0.950,
             "Tese: Jo billigere reserver (høyere akkresjon), jo bedre kursutvikling. "
             "Stiplet = prediksjon for fremtidige maal.",
             fontsize=10.5, color="#566573", va="top")

    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.07, right=0.96, wspace=0.22)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Raa 50-dagers avkastning
    # ════════════════════════════════════════════════════════════════════════
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    # Regresjonslinje
    x_line = np.linspace(-8, 20, 100)
    y_line = np.polyval(coeffs_raw, x_line)
    ax1.plot(x_line, y_line, color=C_REG, lw=1.5, ls="-", alpha=0.3, zorder=2)
    ax1.fill_between(x_line, y_line - 10, y_line + 10,
                     alpha=0.04, color=C_REG, zorder=1)

    # Historiske deals
    for d in DEALS:
        size = max(d["value_bn"] * 80, 100)
        ax1.scatter(d["accretion"], d["return_50d"], s=size,
                    c=d["color"], edgecolors="white", lw=1.5, zorder=6, alpha=0.9)
        # Label
        offset_y = 3 if d["return_50d"] > 0 else -4
        va = "bottom" if d["return_50d"] > 0 else "top"
        ax1.annotate(
            f"{d['name']}\n({d['return_50d']:+.1f}%)",
            (d["accretion"], d["return_50d"]),
            xytext=(0, offset_y), textcoords="offset points",
            fontsize=9, fontweight="bold", color=d["color"],
            ha="center", va=va, linespacing=1.2,
        )

    # Fremtidige maal (ekstrapolert)
    for ft in FUTURE_TARGETS:
        acc = ft["akrbp_ev2p"] - ft["target_ev2p"]
        predicted = np.polyval(coeffs_raw, acc)
        ax1.scatter(acc, predicted, s=150, marker="D",
                    c=ft["color"], edgecolors=ft["color"], lw=2,
                    facecolors="none", zorder=7, alpha=0.9)
        ax1.annotate(
            f"{ft['name']}\n(pred: {predicted:+.1f}%)",
            (acc, predicted),
            xytext=(12, -5), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color=ft["color"],
            ha="left", va="center", fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color=ft["color"], lw=1.0),
        )

    ax1.axhline(0, color="#2c3e50", lw=0.8, ls="-", alpha=0.3)
    ax1.axvline(0, color="#c0392b", lw=1.2, ls="--", alpha=0.5)
    ax1.text(-0.5, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -35,
             "← PREMIUM\n(dyrere enn egne)", fontsize=7.5, color="#c0392b",
             ha="right", va="bottom", fontstyle="italic")
    ax1.text(0.5, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -35,
             "AKKRETIVT →\n(billigere enn egne)", fontsize=7.5, color=C_ACC,
             ha="left", va="bottom", fontstyle="italic")

    ax1.set_xlabel("Akkresjon: AKRBP EV/2P − maalets EV/2P ($/boe)", fontsize=10)
    ax1.set_ylabel("AKRBP 50-dagers kursutvikling (%)", fontsize=10)
    ax1.set_title(
        f"Raa kursutvikling (ikke justert for oljepris)\n"
        f"Regresjon: {slope_raw:+.2f}% per $/boe akkresjon   (R² = {r2_raw:.2f})",
        fontsize=11, fontweight="bold", loc="left", pad=8)
    ax1.grid(True, alpha=0.15, color="#b2bec3")

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Meravkastning (justert for Brent)
    # ════════════════════════════════════════════════════════════════════════
    ax2.set_facecolor("#fdfefe")
    for sp in ax2.spines.values(): sp.set_color("#dce0e6")

    # Regresjonslinje
    y_line_ex = np.polyval(coeffs_ex, x_line)
    ax2.plot(x_line, y_line_ex, color=C_REG, lw=1.5, ls="-", alpha=0.3, zorder=2)
    ax2.fill_between(x_line, y_line_ex - 8, y_line_ex + 8,
                     alpha=0.04, color=C_REG, zorder=1)

    for d in DEALS:
        size = max(d["value_bn"] * 80, 100)
        ax2.scatter(d["accretion"], d["excess_return"], s=size,
                    c=d["color"], edgecolors="white", lw=1.5, zorder=6, alpha=0.9)
        offset_y = 3 if d["excess_return"] > 0 else -4
        va = "bottom" if d["excess_return"] > 0 else "top"
        ax2.annotate(
            f"{d['name']}\n({d['excess_return']:+.1f}%)",
            (d["accretion"], d["excess_return"]),
            xytext=(0, offset_y), textcoords="offset points",
            fontsize=9, fontweight="bold", color=d["color"],
            ha="center", va=va, linespacing=1.2,
        )

    # Fremtidige maal
    for ft in FUTURE_TARGETS:
        acc = ft["akrbp_ev2p"] - ft["target_ev2p"]
        predicted = np.polyval(coeffs_ex, acc)
        ax2.scatter(acc, predicted, s=150, marker="D",
                    c=ft["color"], edgecolors=ft["color"], lw=2,
                    facecolors="none", zorder=7, alpha=0.9)
        ax2.annotate(
            f"{ft['name']}\n(pred: {predicted:+.1f}%)",
            (acc, predicted),
            xytext=(12, -5), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color=ft["color"],
            ha="left", va="center", fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color=ft["color"], lw=1.0),
        )

    ax2.axhline(0, color="#2c3e50", lw=0.8, ls="-", alpha=0.3)
    ax2.axvline(0, color="#c0392b", lw=1.2, ls="--", alpha=0.5)

    ax2.set_xlabel("Akkresjon: AKRBP EV/2P − maalets EV/2P ($/boe)", fontsize=10)
    ax2.set_ylabel("Meravkastning vs Brent (%)", fontsize=10)
    ax2.set_title(
        f"Oljeprisjustert meravkastning (AKRBP − Brent-endring)\n"
        f"Regresjon: {slope_ex:+.2f}% per $/boe akkresjon   (R² = {r2_ex:.2f})",
        fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.grid(True, alpha=0.15, color="#b2bec3")

    # Legend
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#566573",
               markersize=10, label="Historiske oppkjøp"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
               markeredgecolor=C_PRED, markersize=10, markeredgewidth=2,
               label="Predikerte maal (ekstrapolert)"),
        Line2D([0], [0], color=C_REG, lw=1.5, alpha=0.4, label="Regresjonslinje"),
    ]
    ax1.legend(handles=legend_items, fontsize=8.5, loc="lower right",
               framealpha=0.92, edgecolor="#dce0e6")
    ax2.legend(handles=legend_items, fontsize=8.5, loc="lower right",
               framealpha=0.92, edgecolor="#dce0e6")

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(0.04, 0.015,
             "Kilder: Yahoo Finance (AKRBP.OL), EIA (Brent), Aker BP børsmeldinger. "
             "Marathon (2014) utelatt — pre-AKRBP ticker.\n"
             "Akkresjon = AKRBP EV/2P paa deal-dato minus maalets implisitte EV/2P. "
             "Prediksjon er basert paa lineaer ekstrapolering av 5 historiske datapunkter — "
             "kun illustrativt.",
             fontsize=7.5, color="#95a5a6", style="italic")

    # ── Lagre ───────────────────────────────────────────────────────────────
    out = OUT_DIR / "17_accretion_vs_return.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nAnalyse lagret: {out}")


if __name__ == "__main__":
    make_analysis()
