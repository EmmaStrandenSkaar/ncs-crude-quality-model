"""
18_deal_comparison_summary.py
Samlet oversikt — historisk rabatt + kursutvikling + kandidatenes rabatt.

Ett enkelt sammenligningsdokument:
  Øverst: Historiske deals med rabatt og faktisk kursutvikling
  Nederst: Kandidater med estimert rabatt og predikert kurseffekt
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
BRENT_CSV    = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

C_BG = "#f7f9fc"


def fetch_50d_return(close, deal_date):
    """50-dagers avkastning fra lukkekurs-serie."""
    dt = pd.Timestamp(deal_date)
    idx = close.index.get_indexer([dt], method="ffill")[0]
    if idx < 0:
        idx = close.index.get_indexer([dt], method="bfill")[0]
    end_idx = min(len(close) - 1, idx + 50)
    return ((close.iloc[end_idx] / close.iloc[idx]) - 1) * 100


def make_analysis():
    # ── Hent data ──────────────────────────────────────────────────────────
    print("  Laster AKRBP.OL...")
    akrbp = yf.download("AKRBP.OL", start="2016-01-01", end="2026-05-10",
                        progress=False, auto_adjust=True)
    if isinstance(akrbp.columns, pd.MultiIndex):
        akrbp = akrbp.droplevel(level=1, axis=1)
    close = akrbp["Close"].dropna()
    close.index = pd.to_datetime(close.index)

    brent_df = pd.read_csv(BRENT_CSV, parse_dates=["date"]).set_index("date")["brent_usd"]

    # ── Historiske deals ───────────────────────────────────────────────────
    hist = [
        dict(name="BP Norge fusjon",    date="2016-06-10", value="$1.44bn",
             akrbp_ev2p=8.0,  target_ev2p=4.4,  assets="Valhall, Skarv, Ula, Hod",
             financing="Aksjebytte + $140m", color="#8e44ad"),
        dict(name="Hess Norge",         date="2017-05-26", value="$2.0bn",
             akrbp_ev2p=18.0, target_ev2p=13.3, assets="Valhall 64%→100%, Hod",
             financing="$500m emisjon + bank", color="#2980b9"),
        dict(name="Total E&P lisenser", date="2018-07-31", value="$205m",
             akrbp_ev2p=17.0, target_ev2p=2.5,  assets="Trell/Trine (Tyrving), Alve",
             financing="Kontant", color="#16a085"),
        dict(name="King Lear (Fenris)", date="2018-10-15", value="$250m",
             akrbp_ev2p=17.0, target_ev2p=3.2,  assets="King Lear gass/kondensat",
             financing="Kontant", color="#1abc9c"),
        dict(name="Lundin Energy",      date="2021-12-21", value="$14.0bn",
             akrbp_ev2p=17.0, target_ev2p=21.9, assets="J.Sverdrup 20%, Ed. Grieg",
             financing="$2.2bn + 272m aksjer", color="#c0392b"),
    ]

    for d in hist:
        d["rabatt"] = d["akrbp_ev2p"] - d["target_ev2p"]
        d["rabatt_pct"] = (d["rabatt"] / d["akrbp_ev2p"]) * 100
        d["return_50d"] = fetch_50d_return(close, d["date"])
        print(f"  {d['name']}: rabatt ${d['rabatt']:.1f}/boe ({d['rabatt_pct']:.0f}%), "
              f"50d retur {d['return_50d']:+.1f}%")

    # ── Kandidater ─────────────────────────────────────────────────────────
    akrbp_ev2p_now = 19.7
    candidates = [
        dict(name="OKEA ASA",          ev2p=4.3,  ev_bn="$0.33bn", prod="33 kboepd",
             assets="Draugen, Gjøa, Brage",          prob="HØY",     color="#e74c3c"),
        dict(name="INPEX Valhall 10%", ev2p=11.4, ev_bn="$0.4bn",  prod="12 kboepd",
             assets="Valhall 10%, Hod 10%",           prob="HØY",     color="#d4ac0d"),
        dict(name="Harbour NCS",       ev2p=5.8,  ev_bn="~$3.5bn", prod="169 kboepd",
             assets="Skarv (op.), Gjøa, Hansteen",    prob="MIDDELS", color="#16a085"),
        dict(name="Lime Petroleum",    ev2p=7.5,  ev_bn="~$0.15bn",prod="10 kboepd",
             assets="Brage 34%, Yme 25%",             prob="MIDDELS", color="#95a5a6"),
        dict(name="Vaar Energi",       ev2p=13.6, ev_bn="~$17.6bn",prod="406 kboepd",
             assets="J.Sverdrup 12%, Balder, Castberg",prob="LAV",    color="#2980b9"),
        dict(name="DNO NCS",           ev2p=13.2, ev_bn="~$2.5bn", prod="80 kboepd",
             assets="Kvitebjørn, Visund, Fram",       prob="LAV",     color="#8e44ad"),
    ]

    for c in candidates:
        c["rabatt"] = akrbp_ev2p_now - c["ev2p"]
        c["rabatt_pct"] = (c["rabatt"] / akrbp_ev2p_now) * 100

    # ── Figur ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 16), facecolor=C_BG)

    fig.text(0.03, 0.980,
             "Aker BP M&A — Rabatt og kursutvikling: historikk vs kandidater",
             fontsize=20, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.03, 0.958,
             "Rabatt = Aker BPs EV/2P minus maalets EV/2P (høyere = billigere reserver).  "
             "Historisk mønster: Større rabatt → bedre kursutvikling.",
             fontsize=11, color="#566573", va="top")

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.47, 0.53],
        top=0.925, bottom=0.05, left=0.03, right=0.97, hspace=0.15,
    )

    # ════════════════════════════════════════════════════════════════════════
    # TABELL 1 — Historiske deals
    # ════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_visible(False)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, len(hist) + 0.5)
    ax1.invert_yaxis()
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title("HISTORISKE OPPKJØP — faktisk rabatt og kursutvikling",
                  fontsize=13, fontweight="bold", loc="left", pad=12,
                  color="#2c3e50")

    # Kolonner
    col_x = [1, 12, 20, 30, 40, 52, 64, 77, 89]
    headers = ["Oppkjøp", "Aar", "Dealverdi", "AKRBP\nEV/2P", "Maal\nEV/2P",
               "Rabatt\n($/boe)", "Rabatt\n(%)", "50d kurs-\nutvikling", "Vurdering"]

    for x, h in zip(col_x, headers):
        ax1.text(x, -0.35, h, fontsize=9, fontweight="bold",
                 color="#2c3e50", va="center", ha="center", linespacing=1.1)

    ax1.axhline(-0.15, color="#2c3e50", lw=1.0, alpha=0.3)

    for yi, d in enumerate(hist):
        # Bakgrunn
        if yi % 2 == 0:
            ax1.axhspan(yi - 0.45, yi + 0.45, alpha=0.03, color="#2c3e50")

        # Fargebar for rabatt visuell
        rabatt_width = max(d["rabatt_pct"] * 0.20, 0)  # skalert
        bar_color = "#27ae60" if d["rabatt"] > 0 else "#c0392b"

        # Data
        ax1.text(col_x[0], yi, d["name"], fontsize=9.5, fontweight="bold",
                 color=d["color"], va="center", ha="center")
        ax1.text(col_x[1], yi, d["date"][:4], fontsize=9.5,
                 color="#2c3e50", va="center", ha="center")
        ax1.text(col_x[2], yi, d["value"], fontsize=9.5,
                 color="#2c3e50", va="center", ha="center")
        ax1.text(col_x[3], yi, f"${d['akrbp_ev2p']:.0f}", fontsize=9.5,
                 color="#7f8c8d", va="center", ha="center")
        ax1.text(col_x[4], yi, f"${d['target_ev2p']:.1f}", fontsize=10,
                 fontweight="bold", color="#2c3e50", va="center", ha="center")

        # Rabatt $/boe
        rabatt_color = "#27ae60" if d["rabatt"] > 0 else "#c0392b"
        ax1.text(col_x[5], yi, f"${d['rabatt']:+.1f}", fontsize=11,
                 fontweight="bold", color=rabatt_color, va="center", ha="center")

        # Rabatt %
        ax1.text(col_x[6], yi, f"{d['rabatt_pct']:+.0f}%", fontsize=10,
                 fontweight="bold", color=rabatt_color, va="center", ha="center")

        # 50d kursutvikling
        ret_color = "#27ae60" if d["return_50d"] > 2 else (
            "#c0392b" if d["return_50d"] < -2 else "#f39c12")
        ax1.text(col_x[7], yi, f"{d['return_50d']:+.1f}%", fontsize=12,
                 fontweight="bold", color=ret_color, va="center", ha="center")

        # Vurdering
        if d["return_50d"] > 10:
            verdict = "STERK ↑"
            vc = "#27ae60"
        elif d["return_50d"] > 2:
            verdict = "POSITIV ↑"
            vc = "#27ae60"
        elif d["return_50d"] > -2:
            verdict = "FLAT →"
            vc = "#f39c12"
        else:
            verdict = "NEGATIV ↓"
            vc = "#c0392b"

        ax1.text(col_x[8], yi, verdict, fontsize=9.5, fontweight="bold",
                 color=vc, va="center", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=vc, lw=0.8))

    # ════════════════════════════════════════════════════════════════════════
    # TABELL 2 — Kandidater
    # ════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#fdfefe")
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, len(candidates) + 1.0)
    ax2.invert_yaxis()
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_title(
        f"OPPKJØPSKANDIDATER — estimert rabatt vs Aker BPs EV/2P i dag (${akrbp_ev2p_now}/boe)",
        fontsize=13, fontweight="bold", loc="left", pad=12, color="#2c3e50")

    col_x2 = [1, 12, 22, 33, 46, 58, 70, 83, 93]
    headers2 = ["Kandidat", "Est. EV", "Produksjon", "Nøkkelaktiva",
                "Maal\nEV/2P", "Rabatt\n($/boe)", "Rabatt\n(%)",
                "Sannsynlighet", "Rabatt-\nvisualisering"]

    for x, h in zip(col_x2, headers2):
        ax2.text(x, -0.35, h, fontsize=9, fontweight="bold",
                 color="#2c3e50", va="center", ha="center", linespacing=1.1)

    ax2.axhline(-0.15, color="#2c3e50", lw=1.0, alpha=0.3)

    # Sortert etter rabatt (høyest først)
    candidates_sorted = sorted(candidates, key=lambda c: -c["rabatt"])

    for yi, c in enumerate(candidates_sorted):
        if yi % 2 == 0:
            ax2.axhspan(yi - 0.45, yi + 0.45, alpha=0.03, color="#2c3e50")

        ax2.text(col_x2[0], yi, c["name"], fontsize=9.5, fontweight="bold",
                 color=c["color"], va="center", ha="center")
        ax2.text(col_x2[1], yi, c["ev_bn"], fontsize=9,
                 color="#2c3e50", va="center", ha="center")
        ax2.text(col_x2[2], yi, c["prod"], fontsize=9,
                 color="#2c3e50", va="center", ha="center")
        ax2.text(col_x2[3], yi, c["assets"], fontsize=7.5,
                 color="#566573", va="center", ha="center")
        ax2.text(col_x2[4], yi, f"${c['ev2p']:.1f}", fontsize=10,
                 fontweight="bold", color="#2c3e50", va="center", ha="center")

        # Rabatt
        ax2.text(col_x2[5], yi, f"+${c['rabatt']:.1f}", fontsize=11,
                 fontweight="bold", color="#27ae60", va="center", ha="center")
        ax2.text(col_x2[6], yi, f"{c['rabatt_pct']:.0f}%", fontsize=10,
                 fontweight="bold", color="#27ae60", va="center", ha="center")

        # Sannsynlighet
        prob_colors = {"HØY": "#27ae60", "MIDDELS": "#f39c12", "LAV": "#c0392b"}
        pc = prob_colors.get(c["prob"], "#7f8c8d")
        ax2.text(col_x2[7], yi, c["prob"], fontsize=10, fontweight="bold",
                 color=pc, va="center", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=pc, lw=0.9))

        # Rabatt-bar (visuell)
        bar_start = col_x2[8] - 4
        bar_width = c["rabatt_pct"] * 0.09  # skalert for plass
        bar_y = yi
        ax2.barh(bar_y, bar_width, 0.45, left=bar_start,
                 color="#27ae60", alpha=0.6, edgecolor="white", lw=0.5)
        ax2.text(bar_start + bar_width + 0.3, bar_y,
                 f"{c['rabatt_pct']:.0f}%", fontsize=7.5, color="#27ae60",
                 va="center", fontweight="bold")

    # Kontekstlinje
    ax2.text(50, len(candidates_sorted) + 0.3,
             "↑ Sortert etter rabatt (største rabatt øverst). "
             "Historikk viser at BP Norge (55% rabatt → +26%) og Total E&P (85% rabatt → +15%) "
             "ga sterkest kursutvikling.",
             fontsize=8.5, color="#566573", ha="center", va="center",
             fontstyle="italic")

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(0.03, 0.015,
             "Kilder: Yahoo Finance (AKRBP.OL), EIA (Brent), selskapenes IR-sider, SODIR/NPD. "
             "EV/2P = Enterprise Value / 2P-reserver (proven + probable). "
             "Rabatt = Aker BPs EV/2P minus maalets EV/2P paa deal-/naavaerende tidspunkt. "
             "50d kursutvikling = AKRBP.OL indeksert avkastning 50 handelsdager etter annonsering.",
             fontsize=7.5, color="#95a5a6", style="italic")

    # ── Lagre ───────────────────────────────────────────────────────────────
    out = OUT_DIR / "18_deal_comparison_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nAnalyse lagret: {out}")


if __name__ == "__main__":
    make_analysis()
