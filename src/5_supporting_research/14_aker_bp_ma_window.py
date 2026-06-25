"""
14_aker_bp_ma_window.py
Aker BP M&A-vindusanalyse — EV/2P, Brent og USDNOK under historiske oppkjøp.

Citi-tesen (mai 2026):
  "Aker BP handles til ~19x EV/2P — samme nivå som ved Hess Norge (2017) og
   Lundin (2021). M&A-vinduet er åpent igjen."

Vi plotter:
  Panel 1: AKRBP aksje + USDNOK (dobbel y-akse) med deal-markører
  Panel 2: Brent med deal-markører
  Panel 3: Deal-sammenligning — implisitt reservepris vs Aker BP egen EV/2P
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
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

# ── Farger ──────────────────────────────────────────────────────────────────
C_AKRBP = "#2c3e50"
C_BRENT = "#e67e22"
C_NOK   = "#16a085"
C_DEAL  = "#c0392b"
C_OWN   = "#2980b9"
C_ACC   = "#27ae60"
C_BG    = "#f7f9fc"

# ── Alle materielle oppkjøp (>$100m) ─────────────────────────────────────────
DEALS = [
    dict(
        name       = "Marathon Oil\nNorge",
        short      = "Marathon",
        date       = "2014-06-02",
        value_bn   = 2.1,
        reserves_2p= None,       # vanskelig å isolere
        implied_2p = 10.4,       # estimert: $2.1bn / ~200 mmboe
        akrbp_ev2p = None,       # Det norske, ikke sammenlignbart
        brent      = 109,
        usdnok     = 5.97,
        share_nok  = None,       # DETNOR, ingen sammenligning
        financing  = "Kontant",
        color      = "#95a5a6",  # grå — pre-Aker BP
        key_assets = "Alvheim (FPSO), Bøyla",
    ),
    dict(
        name       = "BP Norge\nfusjon",
        short      = "BP Norge",
        date       = "2016-06-10",
        value_bn   = 1.44,
        reserves_2p= 225,
        implied_2p = 4.4,        # ekstremt billig i lav oljepris
        akrbp_ev2p = 8.0,        # estimert: Det norske/AKRBP ca $4bn EV / 498 mmboe
        brent      = 50,
        usdnok     = 8.2,
        share_nok  = 80,
        financing  = "Aksjebytte + $140m",
        color      = "#8e44ad",
        key_assets = "Valhall, Skarv, Ula, Hod",
    ),
    dict(
        name       = "Hess\nNorge",
        short      = "Hess Norge",
        date       = "2017-05-26",
        value_bn   = 2.0,
        reserves_2p= 150,
        implied_2p = 13.3,
        akrbp_ev2p = 18.0,      # ~$16.5bn EV / 914 mmboe
        brent      = 52,
        usdnok     = 8.5,
        share_nok  = 175,        # mai 2017 estimat
        financing  = "$500m emisjon + bank",
        color      = "#2980b9",
        key_assets = "Valhall 64%, Hod → 100%",
    ),
    dict(
        name       = "Total E&P\nlisenser",
        short      = "Total E&P",
        date       = "2018-07-31",
        value_bn   = 0.205,
        reserves_2p= 83,        # 2C ressurser
        implied_2p = 2.5,
        akrbp_ev2p = 17.0,      # estimert
        brent      = 75,
        usdnok     = 8.2,
        share_nok  = 260,
        financing  = "Kontant",
        color      = "#16a085",
        key_assets = "Trell/Trine (Tyrving), Alve Nord",
    ),
    dict(
        name       = "King Lear\n(Fenris)",
        short      = "King Lear",
        date       = "2018-10-15",
        value_bn   = 0.250,
        reserves_2p= 77,
        implied_2p = 3.2,
        akrbp_ev2p = 17.0,
        brent      = 81,
        usdnok     = 8.2,
        share_nok  = 270,
        financing  = "Kontant",
        color      = "#1abc9c",
        key_assets = "King Lear gass/kondensat",
    ),
    dict(
        name       = "Lundin\nEnergy",
        short      = "Lundin",
        date       = "2021-12-21",
        value_bn   = 14.0,
        reserves_2p= 639,
        implied_2p = 21.9,
        akrbp_ev2p = 17.0,      # ~$13.3bn EV / 802 mmboe
        brent      = 73,
        usdnok     = 9.0,
        share_nok  = 290,
        financing  = "$2.2bn + 272m aksjer",
        color      = "#c0392b",
        key_assets = "J.Sverdrup 20%, Ed. Grieg",
    ),
]

# Dagens nivåer (mai 2026)
TODAY = dict(
    date      = "2026-05-08",
    brent     = 102,
    usdnok    = 9.26,
    share_nok = 355,
    ev_bn     = 30.0,
    reserves  = 1526,
    ev2p      = 19.7,
    label     = "I DAG\n(mai 2026)",
)


def fetch_yf_data():
    """Last ned AKRBP.OL og USDNOK historikk fra Yahoo Finance."""
    print("  Laster AKRBP.OL og NOK=X fra Yahoo Finance...")

    akrbp = yf.download("AKRBP.OL", start="2014-01-01", end="2026-05-10",
                         progress=False, auto_adjust=True)
    nok   = yf.download("NOK=X", start="2014-01-01", end="2026-05-10",
                         progress=False, auto_adjust=True)

    # AKRBP: daglig lukke
    if isinstance(akrbp.columns, pd.MultiIndex):
        akrbp = akrbp.droplevel(level=1, axis=1)
    akrbp_s = akrbp["Close"].dropna()
    akrbp_s.index = pd.to_datetime(akrbp_s.index)

    # NOK=X er USDNOK
    if isinstance(nok.columns, pd.MultiIndex):
        nok = nok.droplevel(level=1, axis=1)
    nok_s = nok["Close"].dropna()
    nok_s.index = pd.to_datetime(nok_s.index)

    return akrbp_s, nok_s


def make_analysis():
    # ── Data ────────────────────────────────────────────────────────────────
    brent = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    brent = brent[brent["date"] >= "2014-01-01"]

    akrbp_s = nok_s = None
    if HAS_YF:
        try:
            akrbp_s, nok_s = fetch_yf_data()
            print(f"  AKRBP: {len(akrbp_s)} datapunkter, NOK: {len(nok_s)} datapunkter")
        except Exception as e:
            print(f"  Yahoo Finance feilet: {e}")

    # ── Figur ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 22), facecolor=C_BG)

    fig.text(0.05, 0.980,
             "Aker BP ASA — M&A-vindusanalyse",
             fontsize=18, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.05, 0.960,
             "Er M&A-vinduet aapent?  Citi: \"EV/2P ~19x — samme nivaa som ved "
             "Hess Norge (2017) og Lundin (2021)\"",
             fontsize=10, color="#566573", va="top")

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.36, 0.30, 0.34],
        top=0.930, bottom=0.04, left=0.08, right=0.93, hspace=0.32,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — AKRBP aksje + USDNOK (dobbel y-akse)
    # ════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    if akrbp_s is not None and len(akrbp_s) > 100:
        ax1.plot(akrbp_s.index, akrbp_s.values,
                 color=C_AKRBP, lw=1.6, alpha=0.9, label="AKRBP.OL (NOK)")
        ax1.fill_between(akrbp_s.index, akrbp_s.values, alpha=0.06, color=C_AKRBP)
    else:
        # Fallback: plott nøkkelpunkter fra deals
        dates = [pd.Timestamp(d["date"]) for d in DEALS if d.get("share_nok")]
        prices = [d["share_nok"] for d in DEALS if d.get("share_nok")]
        dates.append(pd.Timestamp(TODAY["date"]))
        prices.append(TODAY["share_nok"])
        ax1.plot(dates, prices, color=C_AKRBP, lw=2, marker="o", ms=6,
                 label="AKRBP.OL (NOK) — nøkkelpunkter")

    ax1.set_ylabel("AKRBP aksje (NOK)", fontsize=9, color=C_AKRBP)
    ax1.tick_params(axis="y", labelcolor=C_AKRBP)

    # USDNOK på sekundærakse
    ax1b = ax1.twinx()
    if nok_s is not None and len(nok_s) > 100:
        ax1b.plot(nok_s.index, nok_s.values,
                  color=C_NOK, lw=1.2, alpha=0.6, label="USD/NOK")
    else:
        dates_nok = [pd.Timestamp(d["date"]) for d in DEALS]
        vals_nok  = [d["usdnok"] for d in DEALS]
        dates_nok.append(pd.Timestamp(TODAY["date"]))
        vals_nok.append(TODAY["usdnok"])
        ax1b.plot(dates_nok, vals_nok, color=C_NOK, lw=2, ls="--",
                  marker="s", ms=5, label="USD/NOK — nøkkelpunkter")
    ax1b.set_ylabel("USD/NOK", fontsize=9, color=C_NOK)
    ax1b.tick_params(axis="y", labelcolor=C_NOK)

    # Deal-markører
    for d in DEALS:
        dt = pd.Timestamp(d["date"])
        ax1.axvline(dt, color=d["color"], lw=1.6, ls="--", alpha=0.7, zorder=3)

    # "I dag"-markør
    dt_today = pd.Timestamp(TODAY["date"])
    ax1.axvline(dt_today, color=C_ACC, lw=2.2, ls="-", alpha=0.85, zorder=4)

    # Annotasjoner for deals — sykler gjennom høyder
    ann_heights = [0.92, 0.78, 0.64, 0.50]
    for i, d in enumerate(DEALS):
        dt = pd.Timestamp(d["date"])
        y_frac = ann_heights[i % len(ann_heights)]
        ax1.annotate(
            d["name"],
            xy=(dt, 0), xycoords=("data", "axes fraction"),
            xytext=(dt + pd.Timedelta(days=12), y_frac),
            textcoords=("data", "axes fraction"),
            fontsize=7.5, color=d["color"], fontweight="bold",
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=d["color"],
                      lw=0.7, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.8),
        )

    # "I dag"-annotasjon
    ax1.annotate(
        TODAY["label"],
        xy=(dt_today, 0), xycoords=("data", "axes fraction"),
        xytext=(dt_today + pd.Timedelta(days=12), 0.85),
        textcoords=("data", "axes fraction"),
        fontsize=8, color=C_ACC, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="#eafaf1", ec=C_ACC, lw=0.9),
        arrowprops=dict(arrowstyle="-", color=C_ACC, lw=0.9),
    )

    ax1.set_title(
        "AKRBP aksje (NOK) + USD/NOK — med alle oppkjøp markert\n"
        "Hoy aksje + svak NOK = sterk oppkjopsvaluta (kan betale i dyre aksjer)",
        fontsize=10, fontweight="bold", loc="left")
    ax1.set_xlim(pd.Timestamp("2014-01-01"), pd.Timestamp("2026-09-01"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(True, alpha=0.15, color="#b2bec3")

    # Samlet legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="upper left", framealpha=0.88, edgecolor="#dce0e6")

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Brent + deal-markører
    # ════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#fdfefe")
    for sp in ax2.spines.values(): sp.set_color("#dce0e6")

    ax2.plot(brent["date"], brent["brent_usd"],
             color=C_BRENT, lw=1.3, alpha=0.8)
    ax2.fill_between(brent["date"], brent["brent_usd"], alpha=0.08, color=C_BRENT)

    for d in DEALS:
        dt = pd.Timestamp(d["date"])
        ax2.axvline(dt, color=d["color"], lw=1.6, ls="--", alpha=0.7, zorder=3)
        ax2.scatter(dt, d["brent"], s=90, c=d["color"],
                    edgecolors="white", lw=1.0, zorder=6)
        ax2.annotate(f"${d['brent']}", (dt, d["brent"]),
                     xytext=(0, 10), textcoords="offset points",
                     fontsize=7.5, color=d["color"], fontweight="bold", ha="center")

    # "I dag"
    ax2.axvline(dt_today, color=C_ACC, lw=2.2, ls="-", alpha=0.85, zorder=4)
    ax2.scatter(dt_today, TODAY["brent"], s=120, c=C_ACC,
                edgecolors="white", lw=1.2, zorder=6, marker="*")
    ax2.annotate(f"${TODAY['brent']}", (dt_today, TODAY["brent"]),
                 xytext=(0, 12), textcoords="offset points",
                 fontsize=8.5, color=C_ACC, fontweight="bold", ha="center")

    ax2.set_ylabel("Brent (USD/fat)", fontsize=9)
    ax2.set_title(
        "Brent oljepris ved hvert oppkjop\n"
        "Hoeyere Brent = bedre kontantstrøm for gjeldsbetjening + hoeyere aksje",
        fontsize=10, fontweight="bold", loc="left")
    ax2.set_xlim(pd.Timestamp("2014-01-01"), pd.Timestamp("2026-09-01"))
    ax2.set_ylim(10, 145)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, alpha=0.15, color="#b2bec3")

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Implisitt reservepris per deal vs Aker BPs egen EV/2P
    # ════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#fdfefe")
    for sp in ax3.spines.values(): sp.set_color("#dce0e6")

    # Bare deals med implied_2p data (alle har det)
    deal_names   = [d["short"] for d in DEALS]
    deal_implied = [d["implied_2p"] for d in DEALS]
    deal_ev2p    = [d.get("akrbp_ev2p") for d in DEALS]
    deal_values  = [d["value_bn"] for d in DEALS]
    deal_colors  = [d["color"] for d in DEALS]

    y_pos = np.arange(len(DEALS))
    bar_h = 0.35

    # Implisitt reservepris (hva de BETALTE per fat 2P)
    bars1 = ax3.barh(y_pos - bar_h/2, deal_implied, bar_h,
                     color=deal_colors, alpha=0.85,
                     edgecolor="white", lw=0.8, label="Betalt ($/2P boe)")

    # Aker BPs egen EV/2P på deal-tidspunkt (hva MARKEDET priser deres fat)
    for yi, ev2p in enumerate(deal_ev2p):
        if ev2p is not None:
            ax3.barh(yi + bar_h/2, ev2p, bar_h,
                     color=C_OWN, alpha=0.50,
                     edgecolor=C_OWN, lw=1.0)

    # "I dag" — Aker BPs nåværende EV/2P
    # Legg til en ekstra rad
    today_y = len(DEALS)
    ax3.barh(today_y, TODAY["ev2p"], bar_h * 2,
             color=C_ACC, alpha=0.80, edgecolor="white", lw=0.8)
    ax3.text(TODAY["ev2p"] + 0.3, today_y,
             f"  ${TODAY['ev2p']:.1f}/boe — \"M&A-vinduet\"",
             va="center", fontsize=8.5, color=C_ACC, fontweight="bold")

    # Annotér deal-verdier og differanser
    for yi, d in enumerate(DEALS):
        # Betalt pris
        ax3.text(d["implied_2p"] + 0.3, yi - bar_h/2,
                 f"${d['implied_2p']:.1f}  (${d['value_bn']:.1f}bn)",
                 va="center", fontsize=7.5, color="#2c3e50", fontweight="bold")
        # Aker BPs egen
        if d.get("akrbp_ev2p"):
            ax3.text(d["akrbp_ev2p"] + 0.3, yi + bar_h/2,
                     f"AKRBP: ${d['akrbp_ev2p']:.0f}",
                     va="center", fontsize=7, color=C_OWN, alpha=0.75)
            # Akkresjon?
            diff = d["akrbp_ev2p"] - d["implied_2p"]
            if diff > 0:
                ax3.text(28, yi,
                         f"AKKRETIVT\n(+${diff:.1f}/boe)",
                         va="center", ha="center", fontsize=7,
                         color=C_ACC, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", fc="#eafaf1",
                                   ec=C_ACC, lw=0.6))
            else:
                ax3.text(28, yi,
                         f"PREMIUM\n(${diff:+.1f}/boe)",
                         va="center", ha="center", fontsize=7,
                         color=C_DEAL, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", fc="#fdedec",
                                   ec=C_DEAL, lw=0.6))

    # Y-akse labels
    all_labels = deal_names + [f"I DAG\n(EV/2P)"]
    ax3.set_yticks(np.arange(len(all_labels)))
    ax3.set_yticklabels(all_labels, fontsize=8.5)
    ax3.set_xlabel("USD per fat 2P-reserver", fontsize=9)
    ax3.set_title(
        "Implisitt reservepris per oppkjop vs Aker BPs egen EV/2P\n"
        "Groenn = akkretivt (betaler mindre per fat enn markedet verdsetter egne reserver)\n"
        "Roed = betalt premium for kvalitetsreserver",
        fontsize=9.5, fontweight="bold", loc="left")
    ax3.axvline(0, color="#2c3e50", lw=0.5, alpha=0.3)
    ax3.grid(True, axis="x", alpha=0.15, color="#b2bec3")
    ax3.set_xlim(-1, 34)
    ax3.invert_yaxis()

    lp_leg = [
        mpatches.Patch(color="#566573", alpha=0.85, label="Betalt per fat ($/2P boe)"),
        mpatches.Patch(color=C_OWN, alpha=0.50, label="Aker BP egen EV/2P paa deal-dato"),
        mpatches.Patch(color=C_ACC, alpha=0.80, label="Aker BP EV/2P i dag ($19.7)"),
    ]
    ax3.legend(handles=lp_leg, fontsize=7.8, loc="lower right",
               framealpha=0.90, edgecolor="#dce0e6")

    # ── Tabell nederst ─────────────────────────────────────────────────────
    fig.text(0.05, 0.018,
             "Kilder: Aker BP borsmeldinger, EIA (Brent), Yahoo Finance (AKRBP.OL, USDNOK). "
             "Citi-analyse mai 2026: Tianhong Bi. Reservetall er 2P (proven + probable). "
             "EV = markedsverdi + netto gjeld. Implisitt 2P-pris = dealverdi / 2P-reserver.",
             fontsize=7, color="#95a5a6", style="italic")

    # ── Lagre ────────────────────────────────────────────────────────────────
    out = OUT_DIR / "14_aker_bp_ma_window.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Analyse lagret: {out}")


if __name__ == "__main__":
    make_analysis()
