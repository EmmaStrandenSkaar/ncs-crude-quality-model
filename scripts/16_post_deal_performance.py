"""
16_post_deal_performance.py
Indeksert kursutvikling AKRBP.OL — 50 handelsdager etter hvert oppkjøp.

Klassisk event-study: Dag 0 = annonsering, indeksert til 100.
Viser hvordan markedet reagerte paa hver deal over tid.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

PROJECT_ROOT = Path(__file__).parent.parent
BRENT_CSV    = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

# ── Farger ──────────────────────────────────────────────────────────────────
C_BG = "#f7f9fc"

# ── Oppkjøp med annonsedatoer ──────────────────────────────────────────────
# Marathon (2014) og BP Norge (2016) er før AKRBP-ticker eksisterte
# BP Norge fusjon: Det norske -> Aker BP skjedde sept 2016, vi bruker DETNOR/AKRBP
DEALS = [
    dict(
        name     = "BP Norge fusjon",
        date     = "2016-06-10",
        color    = "#8e44ad",
        ls       = "-",
        note     = "$1.4bn — Valhall, Skarv, Ula",
    ),
    dict(
        name     = "Hess Norge",
        date     = "2017-05-26",
        color    = "#2980b9",
        ls       = "-",
        note     = "$2.0bn — Valhall 64%→100%",
    ),
    dict(
        name     = "Total E&P lisenser",
        date     = "2018-07-31",
        color    = "#16a085",
        ls       = "-",
        note     = "$205m — Trell/Trine (Tyrving)",
    ),
    dict(
        name     = "King Lear (Fenris)",
        date     = "2018-10-15",
        color    = "#1abc9c",
        ls       = "--",
        note     = "$250m — King Lear gass",
    ),
    dict(
        name     = "Lundin Energy",
        date     = "2021-12-21",
        color    = "#c0392b",
        ls       = "-",
        note     = "$14bn — J.Sverdrup 20%, Ed. Grieg",
    ),
]

WINDOW_BEFORE = 10   # handelsdager før
WINDOW_AFTER  = 50   # handelsdager etter


def fetch_akrbp():
    """Last AKRBP.OL fra Yahoo Finance."""
    print("  Laster AKRBP.OL fra Yahoo Finance...")
    data = yf.download("AKRBP.OL", start="2016-01-01", end="2026-05-10",
                       progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(level=1, axis=1)
    close = data["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    print(f"  {len(close)} handelsdager lastet")
    return close


def fetch_brent():
    """Last Brent fra CSV for sammenligning."""
    brent = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    brent = brent.set_index("date")["brent_usd"]
    return brent


def get_indexed_window(series, event_date, before=10, after=50):
    """
    Hent indeksert vindu rundt en event-dato.
    Returnerer (relative_days, indexed_values) der dag 0 = 100.
    """
    event_dt = pd.Timestamp(event_date)

    # Finn nærmeste handelsdag
    idx = series.index.get_indexer([event_dt], method="ffill")[0]
    if idx < 0:
        idx = series.index.get_indexer([event_dt], method="bfill")[0]

    start_idx = max(0, idx - before)
    end_idx   = min(len(series) - 1, idx + after)

    window = series.iloc[start_idx:end_idx + 1]
    base_price = series.iloc[idx]

    # Indekser til 100
    indexed = (window / base_price) * 100

    # Relative handelsdager (dag 0 = annonsering)
    rel_days = np.arange(-(idx - start_idx), end_idx - idx + 1)

    return rel_days, indexed.values, base_price


def make_analysis():
    if not HAS_YF:
        print("yfinance ikke installert!")
        return

    akrbp = fetch_akrbp()
    brent = fetch_brent()

    # ── Figur ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=C_BG)

    fig.text(0.05, 0.975,
             "Aker BP — Kursutvikling etter oppkjøpsannonseringer",
             fontsize=19, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.05, 0.953,
             "Indeksert til 100 paa annonseringsdagen (dag 0).  "
             "50 handelsdager etter = ~2.5 maaneder.  "
             "Viser markedets dom over hver deal.",
             fontsize=10.5, color="#566573", va="top")

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.62, 0.38],
        top=0.920, bottom=0.06, left=0.08, right=0.75, hspace=0.30,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Indeksert AKRBP aksje etter hver deal
    # ════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    # Referanselinje = 100
    ax1.axhline(100, color="#2c3e50", lw=1.0, ls="-", alpha=0.3, zorder=1)
    ax1.axvline(0, color="#2c3e50", lw=1.2, ls="--", alpha=0.4, zorder=1)

    # Skygge over positive/negative soner
    ax1.axhspan(100, 130, alpha=0.03, color="#27ae60")
    ax1.axhspan(70, 100, alpha=0.03, color="#c0392b")
    ax1.text(WINDOW_AFTER - 1, 100.5, "Annonsering = 100",
             fontsize=7.5, color="#7f8c8d", ha="right", va="bottom")

    results = {}
    for d in DEALS:
        try:
            days, vals, base = get_indexed_window(
                akrbp, d["date"], WINDOW_BEFORE, WINDOW_AFTER)
            ax1.plot(days, vals, color=d["color"], lw=2.2, ls=d["ls"],
                     alpha=0.85, label=d["name"], zorder=4)

            # Sluttverdien etter 50 dager
            end_val = vals[-1]
            end_day = days[-1]
            change = end_val - 100

            # Annotér sluttverdien
            ax1.annotate(
                f"{change:+.1f}%",
                (end_day, end_val),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=d["color"],
                va="center",
            )

            results[d["name"]] = dict(
                base_nok=base, end_indexed=end_val, change_pct=change,
                color=d["color"], note=d["note"],
            )
            print(f"  {d['name']}: dag 0 = {base:.1f} NOK, "
                  f"dag {WINDOW_AFTER} = {change:+.1f}%")
        except Exception as e:
            print(f"  {d['name']}: FEIL — {e}")

    ax1.set_xlim(-WINDOW_BEFORE - 1, WINDOW_AFTER + 8)
    ax1.set_ylim(70, 130)
    ax1.set_xlabel("Handelsdager relativt til annonsering (dag 0)", fontsize=10)
    ax1.set_ylabel("Indeksert kurs (dag 0 = 100)", fontsize=10)
    ax1.set_title(
        "AKRBP.OL — indeksert kursutvikling etter hvert oppkjøp\n"
        "Dag 0 = annonseringsdag, indeksert til 100",
        fontsize=12, fontweight="bold", loc="left", pad=8)
    ax1.grid(True, alpha=0.15, color="#b2bec3")

    ax1.legend(fontsize=9, loc="upper left", framealpha=0.92,
               edgecolor="#dce0e6", ncol=1)

    # Dag-markører
    for day_mark in [10, 20, 30, 40, 50]:
        ax1.axvline(day_mark, color="#dce0e6", lw=0.5, ls=":", alpha=0.5)
        ax1.text(day_mark, ax1.get_ylim()[0] + 0.5, f"d+{day_mark}",
                 fontsize=6.5, color="#b2bec3", ha="center", va="bottom")

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Brent etter hvert oppkjøp (kontekst)
    # ════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#fdfefe")
    for sp in ax2.spines.values(): sp.set_color("#dce0e6")

    ax2.axhline(100, color="#2c3e50", lw=1.0, ls="-", alpha=0.3, zorder=1)
    ax2.axvline(0, color="#2c3e50", lw=1.2, ls="--", alpha=0.4, zorder=1)

    for d in DEALS:
        try:
            days, vals, base = get_indexed_window(
                brent, d["date"], WINDOW_BEFORE, WINDOW_AFTER)
            ax2.plot(days, vals, color=d["color"], lw=1.8, ls=d["ls"],
                     alpha=0.70, label=d["name"], zorder=4)

            end_val = vals[-1]
            end_day = days[-1]
            change = end_val - 100
            ax2.annotate(
                f"{change:+.1f}%",
                (end_day, end_val),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7.5, fontweight="bold", color=d["color"],
                va="center",
            )
        except Exception as e:
            print(f"  Brent {d['name']}: FEIL — {e}")

    ax2.set_xlim(-WINDOW_BEFORE - 1, WINDOW_AFTER + 8)
    ax2.set_ylim(75, 135)
    ax2.set_xlabel("Handelsdager relativt til annonsering (dag 0)", fontsize=10)
    ax2.set_ylabel("Indeksert Brent (dag 0 = 100)", fontsize=10)
    ax2.set_title(
        "Brent-utvikling etter annonsering (kontekst — var det oljepris som drev aksjen?)",
        fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.grid(True, alpha=0.15, color="#b2bec3")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.92,
               edgecolor="#dce0e6", ncol=2)

    for day_mark in [10, 20, 30, 40, 50]:
        ax2.axvline(day_mark, color="#dce0e6", lw=0.5, ls=":", alpha=0.5)

    # ════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Oppsummering per deal
    # ════════════════════════════════════════════════════════════════════════
    ax_side = fig.add_axes([0.77, 0.06, 0.22, 0.86])
    ax_side.set_facecolor(C_BG)
    for sp in ax_side.spines.values(): sp.set_visible(False)
    ax_side.set_xlim(0, 10)
    ax_side.set_ylim(0, len(DEALS) + 0.5)
    ax_side.invert_yaxis()
    ax_side.set_xticks([])
    ax_side.set_yticks([])

    ax_side.text(5, 0.15, "Oppsummering per deal",
                 fontsize=12, fontweight="bold", color="#2c3e50",
                 ha="center", va="center")

    for i, d in enumerate(DEALS):
        y_base = i + 0.7
        r = results.get(d["name"])
        if r is None:
            continue

        change = r["change_pct"]
        verdict_color = "#27ae60" if change > 2 else ("#c0392b" if change < -2 else "#f39c12")
        verdict = "POSITIV" if change > 2 else ("NEGATIV" if change < -2 else "NØYTRAL")

        # Dealfarget markør
        ax_side.plot([0.3], [y_base], marker="s", ms=10, color=d["color"], zorder=5)

        # Dealnavn
        ax_side.text(1.0, y_base - 0.08, d["name"],
                     fontsize=9.5, fontweight="bold", color=d["color"], va="center")
        ax_side.text(1.0, y_base + 0.15, d["note"],
                     fontsize=7, color="#7f8c8d", va="center")

        # Dag 0 kurs
        ax_side.text(1.0, y_base + 0.35,
                     f"Dag 0: {r['base_nok']:.0f} NOK",
                     fontsize=7.5, color="#566573", va="center")

        # 50-dagers endring
        ax_side.text(7.0, y_base + 0.05,
                     f"{change:+.1f}%",
                     fontsize=14, fontweight="bold", color=verdict_color,
                     va="center", ha="center")
        ax_side.text(7.0, y_base + 0.35,
                     verdict,
                     fontsize=7, fontweight="bold", color=verdict_color,
                     va="center", ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec=verdict_color, lw=0.7))

        # Skillelinje
        if i < len(DEALS) - 1:
            ax_side.axhline(y_base + 0.55, color="#dce0e6", lw=0.5, alpha=0.7)

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(0.05, 0.015,
             "Kilder: Yahoo Finance (AKRBP.OL), EIA (Brent). "
             "Marathon Oil Norge (2014) er utelatt — aksjen handlet som Det norske (DETNOR) og er ikke sammenlignbar.\n"
             "Indeksering: Dag 0 lukkekurs = 100. Endring viser total kursutvikling 50 handelsdager (~2.5 mnd) etter annonsering.",
             fontsize=7.5, color="#95a5a6", style="italic")

    # ── Lagre ───────────────────────────────────────────────────────────────
    out = OUT_DIR / "16_post_deal_performance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nAnalyse lagret: {out}")


if __name__ == "__main__":
    make_analysis()
