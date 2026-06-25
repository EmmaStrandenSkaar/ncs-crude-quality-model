"""
19_ma_window_timing.py
M&A-vindu timing: Hvor langt inn i oppgangssyklusen slo Aker BP til?

Definisjon av "vinduet åpner":
  - Siste lokale bunn i aksjen FØR oppkjøpet (>15% drawdown fra forrige topp)
  - Antall handelsdager fra bunn til deal = "vinduets alder"

Viser:
  Panel 1: AKRBP med alle vindu-perioder markert
  Panel 2: Aksjeoppgang fra bunn til deal (%) vs vinduets alder (dager)
  Panel 3: Dagens vindu — hvor er vi nå sammenlignet med historikk?
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

C_BG = "#f7f9fc"

DEALS = [
    dict(name="BP Norge",      date="2016-06-10", color="#8e44ad",
         value="$1.44bn", note="Aksjebytte + kontant"),
    dict(name="Hess Norge",    date="2017-05-26", color="#2980b9",
         value="$2.0bn",  note="Emisjon + bank"),
    dict(name="Total E&P",     date="2018-07-31", color="#16a085",
         value="$205m",   note="Kontant"),
    dict(name="King Lear",     date="2018-10-15", color="#1abc9c",
         value="$250m",   note="Kontant"),
    dict(name="Lundin Energy", date="2021-12-21", color="#c0392b",
         value="$14.0bn", note="Aksjer + kontant"),
]


def find_cycle_trough(close, deal_date, lookback_days=500, drawdown_threshold=0.12):
    """
    Finn siste signifikante bunn STRENGT FØR deal-datoen.
    Gaar bakover fra deal-datoen og finner siste lokale minimum
    som er starten paa rallyet opp til deal-datoen.
    """
    dt = pd.Timestamp(deal_date)
    idx = close.index.get_indexer([dt], method="ffill")[0]

    # Kun data FØR deal-datoen
    start = max(0, idx - lookback_days)
    pre_deal = close.iloc[start:idx + 1]

    # Strategi: Gaa bakover fra deal-datoen.
    # For hvert punkt, sjekk om aksjen steg monotont (med litt støy) derfra.
    # Finn det laveste punktet som starter den siste sammenhengende oppgangen.

    deal_price = pre_deal.iloc[-1]

    # Enkel tilnaerming: Finn laveste kurs i siste 60-300 dager FØR deal
    # Start med aa søke i 300d-vinduet, finn den absolutte bunnen
    best_trough_date = None
    best_trough_price = None

    # Søk etter signifikant bunn — rulerende min
    # Vi vil finne det siste punktet som representerer start paa rallyet
    prices = pre_deal.values
    dates = pre_deal.index

    # Gaa bakover fra deal-dato og finn der rallyet startet
    # Rallyet "startet" ved det laveste punktet i trailing window
    for lookback in [60, 90, 120, 180, 250, 350]:
        if len(prices) < lookback:
            continue
        window_prices = prices[-lookback:]
        window_dates = dates[-lookback:]
        min_idx = np.argmin(window_prices)
        min_price = window_prices[min_idx]
        min_date = window_dates[min_idx]

        rally_from_min = (deal_price / min_price - 1)

        # Aksepter denne bunnen hvis rallyet er signifikant (>15%)
        if rally_from_min > 0.15:
            best_trough_date = min_date
            best_trough_price = min_price
            break

    # Fallback: bruk laveste i hele vinduet
    if best_trough_date is None:
        min_idx = np.argmin(prices)
        best_trough_date = dates[min_idx]
        best_trough_price = prices[min_idx]

    # Sørg for at bunnen er FØR deal (ikke paa deal-datoen)
    if best_trough_date >= dt:
        # Søk bredere
        min_idx = np.argmin(prices[:-5])  # ekskluder siste 5 dager
        best_trough_date = dates[min_idx]
        best_trough_price = prices[min_idx]

    return best_trough_date, float(best_trough_price)


def make_analysis():
    if not HAS_YF:
        print("yfinance trengs!")
        return

    print("  Laster AKRBP.OL...")
    data = yf.download("AKRBP.OL", start="2015-06-01", end="2026-05-10",
                       progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(level=1, axis=1)
    close = data["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    print(f"  {len(close)} handelsdager")

    # ── Beregn vindu-parametre for hver deal ───────────────────────────────
    for d in DEALS:
        dt = pd.Timestamp(d["date"])
        deal_idx = close.index.get_indexer([dt], method="ffill")[0]
        deal_price = close.iloc[deal_idx]

        trough_date, trough_price = find_cycle_trough(close, d["date"])
        trough_idx = close.index.get_loc(trough_date)

        d["deal_price"]   = deal_price
        d["trough_date"]  = trough_date
        d["trough_price"] = trough_price
        d["window_days"]  = (dt - trough_date).days
        d["window_trading_days"] = deal_idx - trough_idx
        d["rally_pct"]    = ((deal_price / trough_price) - 1) * 100

        print(f"  {d['name']}: bunn {trough_date.strftime('%Y-%m-%d')} ({trough_price:.0f} NOK) "
              f"→ deal {d['date']} ({deal_price:.0f} NOK), "
              f"vindu {d['window_days']} dager, rally +{d['rally_pct']:.0f}%")

    # ── Dagens vindu ───────────────────────────────────────────────────────
    today_date = close.index[-1]
    today_price = close.iloc[-1]
    today_trough_date, today_trough_price = find_cycle_trough(
        close, today_date.strftime("%Y-%m-%d"), lookback_days=600)

    today_window = {
        "trough_date": today_trough_date,
        "trough_price": today_trough_price,
        "today_price": today_price,
        "window_days": (today_date - today_trough_date).days,
        "rally_pct": ((today_price / today_trough_price) - 1) * 100,
    }
    print(f"\n  I DAG: bunn {today_trough_date.strftime('%Y-%m-%d')} ({today_trough_price:.0f} NOK) "
          f"→ naa ({today_price:.0f} NOK), "
          f"vindu {today_window['window_days']} dager, rally +{today_window['rally_pct']:.0f}%")

    # ── Figur ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16), facecolor=C_BG)

    fig.text(0.04, 0.980,
             "Aker BP — Timing av M&A-vinduet",
             fontsize=20, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.04, 0.958,
             "Hvor langt inn i aksjeoppgangen var Aker BP da de slo til paa hvert oppkjøp?  "
             "Og hvor er vi i dagens syklus?",
             fontsize=11, color="#566573", va="top")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[0.55, 0.45],
        width_ratios=[0.60, 0.40],
        top=0.920, bottom=0.06, left=0.06, right=0.97, hspace=0.28, wspace=0.18,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — AKRBP med vindu-perioder markert
    # ════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    ax1.plot(close.index, close.values, color="#2c3e50", lw=1.4, alpha=0.8, zorder=3)
    ax1.fill_between(close.index, close.values, alpha=0.04, color="#2c3e50")

    # Marker vindu-perioder (bunn → deal)
    for d in DEALS:
        dt = pd.Timestamp(d["date"])
        tr = d["trough_date"]

        # Fyll vinduperioden
        mask = (close.index >= tr) & (close.index <= dt)
        window_dates = close.index[mask]
        window_vals = close.values[mask]
        ax1.fill_between(window_dates, window_vals, alpha=0.12, color=d["color"], zorder=2)

        # Bunn-markør
        ax1.scatter(tr, d["trough_price"], s=120, c=d["color"],
                    marker="v", edgecolors="white", lw=1.5, zorder=6)

        # Deal-markør
        ax1.scatter(dt, d["deal_price"], s=120, c=d["color"],
                    marker="^", edgecolors="white", lw=1.5, zorder=6)

        # Deal-linje
        ax1.axvline(dt, color=d["color"], lw=1.2, ls="--", alpha=0.4)

        # Annotasjon
        mid_date = tr + (dt - tr) / 2
        y_ann = d["deal_price"] * 1.06
        ax1.annotate(
            f"{d['name']}\n{d['window_days']}d | +{d['rally_pct']:.0f}%",
            (mid_date, y_ann),
            fontsize=7.5, fontweight="bold", color=d["color"],
            ha="center", va="bottom", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=d["color"],
                      lw=0.7, alpha=0.92),
        )

    # Dagens vindu
    mask_today = (close.index >= today_trough_date)
    ax1.fill_between(close.index[mask_today], close.values[mask_today],
                     alpha=0.08, color="#27ae60", zorder=2)
    ax1.scatter(today_trough_date, today_trough_price, s=150, c="#27ae60",
                marker="v", edgecolors="white", lw=2, zorder=6)
    ax1.scatter(today_date, today_price, s=200, c="#27ae60",
                marker="*", edgecolors="white", lw=2, zorder=6)

    ax1.annotate(
        f"NAAVAERENDE VINDU\n{today_window['window_days']}d | +{today_window['rally_pct']:.0f}%",
        (today_date - pd.Timedelta(days=60), today_price * 1.05),
        fontsize=9, fontweight="bold", color="#27ae60",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", lw=1.0),
    )

    ax1.set_ylabel("AKRBP.OL (NOK)", fontsize=10)
    ax1.set_title(
        "AKRBP aksje med M&A-vinduer markert (▼ = syklusbunn, ▲ = deal-annonsering)\n"
        "Fargede omraader viser perioden fra bunn til deal — \"vinduets levetid\"",
        fontsize=11, fontweight="bold", loc="left", pad=8)
    ax1.grid(True, alpha=0.15, color="#b2bec3")
    ax1.set_xlim(pd.Timestamp("2015-10-01"), pd.Timestamp("2026-08-01"))

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Scatter: Vinduets alder vs rally-størrelse
    # ════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#fdfefe")
    for sp in ax2.spines.values(): sp.set_color("#dce0e6")

    for d in DEALS:
        size = 180
        ax2.scatter(d["window_days"], d["rally_pct"], s=size,
                    c=d["color"], edgecolors="white", lw=1.5, zorder=5, alpha=0.9)
        ax2.annotate(
            f"  {d['name']}\n  ({d['value']})",
            (d["window_days"], d["rally_pct"]),
            fontsize=8, fontweight="bold", color=d["color"],
            va="center", linespacing=1.2,
        )

    # "I dag"-markør
    ax2.scatter(today_window["window_days"], today_window["rally_pct"],
                s=300, c="#27ae60", marker="*", edgecolors="white", lw=2, zorder=6)
    ax2.annotate(
        f"  I DAG\n  ({today_window['window_days']}d, +{today_window['rally_pct']:.0f}%)",
        (today_window["window_days"], today_window["rally_pct"]),
        fontsize=9, fontweight="bold", color="#27ae60",
        va="center", linespacing=1.2,
    )

    # Gjennomsnittlig vindu
    avg_days = np.mean([d["window_days"] for d in DEALS])
    avg_rally = np.mean([d["rally_pct"] for d in DEALS])
    ax2.axvline(avg_days, color="#7f8c8d", lw=1.2, ls=":", alpha=0.5)
    ax2.axhline(avg_rally, color="#7f8c8d", lw=1.2, ls=":", alpha=0.5)
    ax2.text(avg_days + 5, ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else 0,
             f"Snitt: {avg_days:.0f}d",
             fontsize=8, color="#7f8c8d", va="bottom", fontstyle="italic")

    ax2.set_xlabel("Vinduets alder (kalenderdager fra bunn til deal)", fontsize=10)
    ax2.set_ylabel("Aksjeoppgang fra bunn (%)", fontsize=10)
    ax2.set_title(
        "Vinduets alder vs aksjeoppgang ved deal-tidspunkt",
        fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.grid(True, alpha=0.15, color="#b2bec3")

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Tidslinje-sammenligning
    # ════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#fdfefe")
    for sp in ax3.spines.values(): sp.set_color("#dce0e6")

    # Horisontale staver: vinduets varighet for hver deal
    all_items = DEALS + [dict(name="I DAG", window_days=today_window["window_days"],
                              rally_pct=today_window["rally_pct"],
                              color="#27ae60", value="?")]
    all_items_sorted = sorted(all_items, key=lambda x: x["window_days"])

    y_pos = np.arange(len(all_items_sorted))
    for yi, item in enumerate(all_items_sorted):
        bar_color = item["color"]
        alpha = 0.9 if item["name"] == "I DAG" else 0.7
        lw = 2.0 if item["name"] == "I DAG" else 0.8
        ec = item["color"] if item["name"] == "I DAG" else "white"

        ax3.barh(yi, item["window_days"], 0.55,
                 color=bar_color, alpha=alpha, edgecolor=ec, lw=lw)
        ax3.text(item["window_days"] + 5, yi,
                 f"{item['window_days']}d (+{item['rally_pct']:.0f}%)",
                 fontsize=9, fontweight="bold", color=bar_color, va="center")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([item["name"] for item in all_items_sorted],
                        fontsize=9.5, fontweight="bold")
    for yi, item in enumerate(all_items_sorted):
        ax3.get_yticklabels()[yi].set_color(item["color"])
    ax3.set_xlabel("Kalenderdager fra syklusbunn til deal", fontsize=10)
    ax3.set_title(
        "Vinduets varighet — sortert\n(hvor mange dager var vinduet aapent?)",
        fontsize=11, fontweight="bold", loc="left", pad=8)
    ax3.grid(True, axis="x", alpha=0.15, color="#b2bec3")
    ax3.invert_yaxis()

    # Gjennomsnittslinje
    ax3.axvline(avg_days, color="#c0392b", lw=1.5, ls="--", alpha=0.6)
    ax3.text(avg_days, len(all_items_sorted) - 0.3,
             f"Snitt: {avg_days:.0f} dager",
             fontsize=8.5, color="#c0392b", ha="center", va="top",
             fontweight="bold")

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(0.04, 0.015,
             "Kilder: Yahoo Finance (AKRBP.OL). "
             "Syklusbunn definert som siste punkt med >12% drawdown fra foregaaende topp, "
             "innenfor 500 handelsdager før deal. "
             "\"Vinduets alder\" = kalenderdager fra bunn til annonseringsdag. "
             "Marathon (2014) utelatt — pre-AKRBP ticker.",
             fontsize=7.5, color="#95a5a6", style="italic")

    # ── Lagre ───────────────────────────────────────────────────────────────
    out = OUT_DIR / "19_ma_window_timing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nAnalyse lagret: {out}")


if __name__ == "__main__":
    make_analysis()
