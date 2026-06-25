"""
Pre-rapport mal v3 — Norske oljeselskaper, Q2 2026
Stående layout. Fokus: realiserte priser, produksjon, kvalitet.

Layout:
  1. Toppbanner (enkel tekst, ingen mørk boks)
  2. Historisk pris + hendelsestidslinje  (realisert pris per region + Brent + events)
  3. [Feltscatter]  +  [Produksjon 8 kvartal]
  4. Felt-for-felt differensial  (lollipop, full bredde)

VIKTIG — Hormuz-klassifisering for DNO:
  Ingen DNO-felt eksporterer via Hormuz-stredet.
  - Kurdistan: Kirkuk–Ceyhan pipeline → Tyrkia → Middelhav.
    Disrupsjon skyldes IRAN-KRIGEN (produksjon stanset 28.feb–9.apr 2026),
    IKKE Hormuz-ruten. Markeres som "Krig-eksponert".
  - NCS / Vest-Afrika: Nordsjø-/Atlanterhavs-ruter — ikke eksponert.
    Fikk PREMIUM under krisen fordi de kan levere der Gulf-olje ikke kan.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_CSV  = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
BRENT_CSV = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
OUT_DIR   = PROJECT_ROOT / "data" / "processed"

C_KRD = "#c0392b"
C_NCS = "#2980b9"
C_WA  = "#27ae60"
C_BRT = "#2c3e50"
C_ACC = "#e67e22"
REGION_COLORS = {"Kurdistan": C_KRD, "Nordsjøen": C_NCS, "Vest-Afrika": C_WA}

# ─────────────────────────────────────────────────────────────────────────────
COMPANY = dict(
    name        = "DNO ASA",
    ticker      = "DNO.OL",
    report_date = "7. mai 2026",
    brent_now   = 103.0,

    # ── Realiserte priser per kvartal + region (bekreftet fra rapporter) ──
    # Utvidet til 2022 for å vise pipeline-closure-effekten
    realized = [
        # (år, kvartal, region, realisert_usd, brent_snitt_usd, bekreftet)
        (2022, 1, "Kurdistan",  96.0,  99.1, True),
        (2022, 2, "Kurdistan", 111.0, 114.2, True),
        (2022, 3, "Kurdistan",  88.0,  99.0, True),
        (2022, 4, "Kurdistan",  72.0,  85.2, True),
        (2023, 1, "Kurdistan",  54.0,  81.3, True),   # pipeline begynner å stenges
        (2023, 2, "Kurdistan",  44.0,  78.0, True),
        (2023, 3, "Kurdistan",  38.0,  85.9, True),
        (2023, 4, "Kurdistan",  33.0,  80.3, True),
        (2024, 1, "Kurdistan",  31.0,  82.1, True),
        (2024, 2, "Kurdistan",  34.0,  85.0, True),
        (2024, 3, "Kurdistan",  35.0,  80.4, True),
        (2024, 4, "Kurdistan",  38.0,  74.5, True),
        (2025, 1, "Kurdistan",  34.7,  76.2, True),
        (2025, 2, "Kurdistan",  32.0,  64.5, False),  # estimert
        (2025, 3, "Kurdistan",  33.0,  67.0, False),
        (2025, 4, "Kurdistan",  31.6,  62.5, True),   # pipeline restart sept 2025
        (2026, 1, "Kurdistan",  31.0,  80.7, True),   # Iran-krig, produksjon stanset

        # NCS: minimal before Sval (jun 2025), viser kun fra Q3-25
        (2025, 3, "Nordsjøen",  66.4,  67.0, False),
        (2025, 4, "Nordsjøen",  63.6,  62.5, True),
        (2026, 1, "Nordsjøen",  87.0,  80.7, True),
    ],

    # ── Historiske hendelser ──────────────────────────────────────────────
    # (dato, kort_label, type: pos/neg/neutral, y_side: +1/-1 over/under akse)
    events = [
        ("2022-02-24", "Russland\nangrir Ukraina", "neg",  -1),
        ("2023-03-25", "Pipeline\nstengt",          "neg",  -1),
        ("2025-06-15", "Sval-oppkjop\n(NCS x10)",   "pos",  +1),
        ("2025-09-27", "Pipeline\nrestart",          "pos",  +1),
        ("2026-02-28", "Iran-krig\n28.feb",          "neg",  -1),
        ("2026-04-07", "Symra-\noppstart",            "pos",  +1),
        ("2026-05-07", "Q1-26\nRapport",             "neutral", +1),
    ],

    # ── Feltdata ─────────────────────────────────────────────────────────
    # war_exposed: True = direkte disrupsjon fra Iran-krigen
    # hormuz_route: True = ville brukt Hormuz (ingen DNO-felt!)
    fields = [
        dict(name="Tawke",        region="Kurdistan",  api=28.0, sulfur=3.50,
             net_kbpd=33.75, war_exp=True,
             diff_hist=-10.0, diff_now=-50.0,  # nå = lokalsalg under krig
             note="Ceyhan-rute\n(krig stanset prod.)"),
        dict(name="Peshkabir",    region="Kurdistan",  api=26.0, sulfur=3.80,
             net_kbpd=11.25, war_exp=True,
             diff_hist=-12.0, diff_now=-50.0,
             note="Ceyhan-rute\n(krig stanset prod.)"),
        dict(name="Ekofisk",      region="Nordsjøen",  api=37.5, sulfur=0.25,
             net_kbpd=3.73,  war_exp=False,
             diff_hist=+1.79, diff_now=+5.0,   # estimert Hormuz-krise premium
             note="BFOET +$1.8"),
        dict(name="Eldfisk",      region="Nordsjøen",  api=37.5, sulfur=0.25,
             net_kbpd=2.93,  war_exp=False,
             diff_hist=+1.79, diff_now=+5.0,
             note="Ekofisk-system"),
        dict(name="Martin Linge", region="Nordsjøen",  api=41.0, sulfur=0.04,
             net_kbpd=2.06,  war_exp=False,
             diff_hist=+1.35, diff_now=+6.0,
             note="Lett kondensat"),
        dict(name="Nova",         region="Nordsjøen",  api=33.0, sulfur=0.25,
             net_kbpd=2.79,  war_exp=False,
             diff_hist=+0.0,  diff_now=+3.0,
             note="Ingen hist.data"),
        dict(name="Maria",        region="Nordsjøen",  api=28.5, sulfur=0.50,
             net_kbpd=1.49,  war_exp=False,
             diff_hist=-0.05, diff_now=+2.0,
             note="Heidrun-area"),
        dict(name="Verdande",     region="Nordsjøen",  api=32.0, sulfur=0.16,
             net_kbpd=1.17,  war_exp=False,
             diff_hist=+1.36, diff_now=+3.0,
             note="Norne-stream"),
        dict(name="Gudrun",       region="Nordsjøen",  api=39.0, sulfur=0.30,
             net_kbpd=1.01,  war_exp=False,
             diff_hist=-5.13, diff_now=-3.0,
             note="Kondensat"),
        dict(name="Kvitebjorn",   region="Nordsjøen",  api=49.0, sulfur=0.13,
             net_kbpd=0.57,  war_exp=False,
             diff_hist=-3.76, diff_now=-2.0,
             note="Kondensat"),
        dict(name="Symra",        region="Nordsjøen",  api=45.0, sulfur=0.05,
             net_kbpd=0.80,  war_exp=False,
             diff_hist=+0.5,  diff_now=+4.0,
             note="Apr 2026"),
        dict(name="Embla/Tor",    region="Nordsjøen",  api=37.5, sulfur=0.25,
             net_kbpd=0.53,  war_exp=False,
             diff_hist=+1.79, diff_now=+5.0,
             note="Ekofisk-syst."),
        dict(name="CI-26",        region="Vest-Afrika", api=35.0, sulfur=0.20,
             net_kbpd=3.30,  war_exp=False,
             diff_hist=-0.10, diff_now=+2.0,
             note="Cote d'Ivoire"),
    ],

    # ── Produksjon per kvartal i kboepd ──────────────────────────────────
    production = [
        ("Q2-24", "2024-05-15", 50.0,  5.5,  3.4),
        ("Q3-24", "2024-08-15", 50.0,  6.0,  3.4),
        ("Q4-24", "2024-11-15", 50.0,  8.0,  3.4),
        ("Q1-25", "2025-02-15", 61.6, 19.3,  3.4),  # bekreftet
        ("Q2-25", "2025-05-15", 60.0, 52.0,  3.4),  # Sval delvis
        ("Q3-25", "2025-08-15", 58.0, 86.0,  3.5),  # Sval fullt
        ("Q4-25", "2024-11-15", 58.0, 88.3,  3.5),  # bekreftet
        ("Q1-26", "2026-02-15", 39.6, 88.6,  3.4),  # bekreftet — krig -6 uker
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
def quarter_to_date(year, q):
    return pd.Timestamp(f"{year}-{q*3-1:02d}-15")

def make_report(co):
    fields_df = pd.DataFrame(co["fields"])
    prod_df   = pd.DataFrame(co["production"],
                             columns=["label","date","Kurdistan","Nordsjøen","Vest-Afrika"])
    prod_df["date"] = pd.to_datetime(prod_df["date"])

    # Realiserte priser → datoer
    real_df = pd.DataFrame(co["realized"],
                           columns=["year","q","region","realized","brent","confirmed"])
    real_df["date"] = real_df.apply(lambda r: quarter_to_date(r.year, r.q), axis=1)
    real_df = real_df.sort_values("date")

    # Brent daily
    brent_d = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    brent_d = brent_d[brent_d["date"] >= "2022-01-01"]

    # ── Figur ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 22), facecolor="#f7f9fc")

    # Enkel tittel — ingen mørk boks
    fig.text(0.05, 0.975,
             f"{co['name']}  ({co['ticker']})",
             fontsize=17, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.05, 0.954,
             f"Pre-rapport analyse Q2 2026  |  Rapport: {co['report_date']}  "
             f"|  Brent: ~${co['brent_now']:.0f}/fat  (mai 2026)",
             fontsize=10, color="#566573", va="top")
    fig.text(0.97, 0.954,
             f"Det ene sporsmalet:\nKurdistan via Ceyhan eller\nlokal kjoper ($31/fat)?",
             fontsize=8.5, color=C_KRD, fontweight="bold",
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fdedec", ec=C_KRD, lw=1.0))

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.30, 0.33, 0.33],
        top=0.922, bottom=0.035, left=0.09, right=0.94, hspace=0.30,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PANEL 1 — Realisert pris + Brent + hendelsestidslinje
    # ══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    # Brent daglig — tynt og dempet
    ax1.plot(brent_d["date"], brent_d["brent_usd"],
             color=C_BRT, lw=1.0, alpha=0.35, zorder=2, label="Brent spot (daglig)")
    ax1.fill_between(brent_d["date"], brent_d["brent_usd"],
                     alpha=0.04, color=C_BRT)

    # Realiserte priser per region
    for region, color, marker in [("Kurdistan", C_KRD, "D"), ("Nordsjøen", C_NCS, "o")]:
        sub = real_df[real_df["region"] == region].copy()
        if sub.empty: continue
        # Skille bekreftet vs estimert
        conf  = sub[sub["confirmed"]]
        estim = sub[~sub["confirmed"]]
        ax1.plot(sub["date"], sub["realized"],
                 color=color, lw=2.2, zorder=5, label=f"{region} realisert")
        ax1.scatter(conf["date"], conf["realized"],
                    s=55, color=color, marker=marker,
                    edgecolors="white", lw=0.8, zorder=6)
        ax1.scatter(estim["date"], estim["realized"],
                    s=45, color=color, marker=marker,
                    facecolors="none", edgecolors=color, lw=1.5, zorder=6)

    # Annotér siste punkt Kurdistan
    krd_last = real_df[real_df["region"]=="Kurdistan"].iloc[-1]
    ax1.annotate(f"Q1-26: ${krd_last.realized:.0f}",
                 (krd_last.date, krd_last.realized),
                 xytext=(8, -18), textcoords="offset points",
                 fontsize=8.5, color=C_KRD, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_KRD, lw=0.8))
    ncs_last = real_df[real_df["region"]=="Nordsjøen"].iloc[-1]
    ax1.annotate(f"Q1-26: ${ncs_last.realized:.0f}",
                 (ncs_last.date, ncs_last.realized),
                 xytext=(8, 6), textcoords="offset points",
                 fontsize=8.5, color=C_NCS, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_NCS, lw=0.8))

    # Hendelsesmarkører — sykler gjennom flere høyder for å unngå overlapp
    ev_styles = {
        "pos":     dict(color=C_NCS,  lw=1.4, ls="--"),
        "neg":     dict(color=C_KRD,  lw=1.4, ls="--"),
        "neutral": dict(color=C_ACC,  lw=1.4, ls="--"),
    }
    y_max = 142
    # Tre høydnivåer per side — roterer slik at nærliggende events ikke overlapper.
    # NB: pos_heights starter på 0.80 (ikke 0.92) slik at to-linje tekst
    # med va="bottom" får nok luft til toppen av ylim.
    pos_heights = [y_max * 0.80, y_max * 0.65, y_max * 0.50]
    neg_heights = [y_max * 0.08, y_max * 0.23, y_max * 0.38]
    pos_i = neg_i = 0

    for dt_str, label, etype, yside in co["events"]:
        dt = pd.Timestamp(dt_str)
        st = ev_styles[etype]
        ax1.axvline(dt, color=st["color"], lw=st["lw"], ls=st["ls"], alpha=0.65, zorder=3)
        if yside > 0:
            y_pos = pos_heights[pos_i % len(pos_heights)]
            pos_i += 1
            va = "bottom"
        else:
            y_pos = neg_heights[neg_i % len(neg_heights)]
            neg_i += 1
            va = "top"
        ax1.text(dt + pd.Timedelta(days=7), y_pos, label,
                 fontsize=7.0, color=st["color"], va=va,
                 fontweight="bold", linespacing=1.35,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=st["color"],
                           lw=0.7, alpha=0.90))

    ax1.set_ylabel("Realisert / Brent (USD/fat)", fontsize=9)
    ax1.set_title("Realiserte priser per region + historiske aksjetriggere\n"
                  "● fylt = bekreftet  ○ aapen = estimert",
                  fontsize=9.5, fontweight="bold", loc="left")
    ax1.set_ylim(12, y_max)
    ax1.set_xlim(brent_d["date"].min(), pd.Timestamp("2026-07-01"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(True, alpha=0.15, color="#b2bec3")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=8.5, loc="upper left",
               framealpha=0.88, edgecolor="#dce0e6")

    # ══════════════════════════════════════════════════════════════════════
    # RAD 2 — [Feltscatter | Produksjon 8 kvartal]
    # ══════════════════════════════════════════════════════════════════════
    inner2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], wspace=0.38)
    ax_sc = fig.add_subplot(inner2[0])
    ax_pr = fig.add_subplot(inner2[1])

    for ax in [ax_sc, ax_pr]:
        ax.set_facecolor("#fdfefe")
        for sp in ax.spines.values(): sp.set_color("#dce0e6")

    # ── Feltscatter ───────────────────────────────────────────────────────
    # Felt med identisk posisjon (Ekofisk/Eldfisk/Embla/Tor = 37.5, 0.25)
    # — vises som én klynge med felles label for å unngå overlapp.
    # Felter som ikke skal ha eget label (dekket av gruppe-label nedenfor):
    no_label = {"Eldfisk", "Embla/Tor"}
    # Manuelt justerte offsets (x_pts, y_pts) per felt
    label_offsets = {
        "Tawke":        (+5,  +10),
        "Peshkabir":    (+5,  -15),
        "Ekofisk":      (+5,   +8),   # vises som "Ekofisk system" gruppe
        "Martin Linge": (+5,   +8),
        "Nova":         (-5,  +10),
        "Maria":        (-26,  +8),
        "Verdande":     (+5,  -14),
        "Gudrun":       (+5,   +8),
        "Kvitebjorn":   (+5,  -14),
        "Symra":        (+5,   +8),
        "CI-26":        (+5,   +8),
    }

    for _, r in fields_df.iterrows():
        col = C_KRD if r["war_exp"] else REGION_COLORS.get(r["region"], "gray")
        mk  = "D" if r["war_exp"] else "o"
        sz  = 50 + r["net_kbpd"] * 30
        ax_sc.scatter(r["api"], r["sulfur"],
                      s=sz, c=col, marker=mk,
                      edgecolors="white", lw=1.0, alpha=0.88, zorder=5)

        if r["name"] in no_label:
            continue   # feltes del av Ekofisk-klyngen — ikke separat label

        if r["net_kbpd"] > 0.5:
            lbl = "Ekofisk system\n(Ekofisk/Eldfisk\nEmbla/Tor)" if r["name"] == "Ekofisk" else r["name"]
            dx, dy = label_offsets.get(r["name"], (+5, +6))
            ax_sc.annotate(lbl,
                           (r["api"], r["sulfur"]),
                           xytext=(dx, dy), textcoords="offset points",
                           fontsize=7.0, color="#2c3e50", zorder=6,
                           linespacing=1.3)

    # Decorasjon
    ax_sc.add_patch(plt.Rectangle((30, 0), 12, 0.5,
                                   alpha=0.06, color="green"))
    ax_sc.text(30.3, 0.02, "Sweet spot",
               fontsize=7, color="darkgreen", alpha=0.75)
    ax_sc.axhline(0.5, color="#aab7b8", ls="--", lw=0.7)
    ax_sc.axvspan(22, 30.5, alpha=0.05, color=C_KRD)

    # Forklaringsboks
    ax_sc.text(22.4, 4.0,
               "Kurdistan: Ceyhan-pipeline\n(ikke Hormuz, men Iran-krig\nstanset produksjon)",
               fontsize=6.8, color=C_KRD, style="italic",
               bbox=dict(boxstyle="round,pad=0.25", fc="#fdedec",
                         ec=C_KRD, lw=0.7, alpha=0.9))

    ax_sc.set_xlabel("API-grad  (hoeyere = lettere)", fontsize=9)
    ax_sc.set_ylabel("Svovel (%)", fontsize=9)
    ax_sc.set_title("Feltportefolje — Oljekvalitet\n"
                    "◆ Krig-eksponert (Iran)   ● Ikke eksponert",
                    fontsize=9.5, fontweight="bold", loc="left")
    ax_sc.set_xlim(21, 56); ax_sc.set_ylim(-0.2, 4.5)
    ax_sc.grid(True, alpha=0.15, color="#b2bec3")

    sc_legend = [
        mpatches.Patch(color=C_KRD, label="Kurdistan (Krig-eksponert)"),
        mpatches.Patch(color=C_NCS, label="Nordsjoen (Ikke eksponert)"),
        mpatches.Patch(color=C_WA,  label="Vest-Afrika (Ikke eksponert)"),
    ]
    ax_sc.legend(handles=sc_legend, fontsize=7.5, loc="upper right",
                 framealpha=0.88, edgecolor="#dce0e6")

    # ── Produksjon stacked bar ────────────────────────────────────────────
    x = np.arange(len(prod_df))
    bw = 0.60
    b1 = ax_pr.bar(x, prod_df["Kurdistan"],
                   bw, color=C_KRD, alpha=0.82,
                   label="Kurdistan", edgecolor="white", lw=0.5)
    b2 = ax_pr.bar(x, prod_df["Nordsjøen"],
                   bw, bottom=prod_df["Kurdistan"],
                   color=C_NCS, alpha=0.82,
                   label="Nordsjoen", edgecolor="white", lw=0.5)
    b3 = ax_pr.bar(x, prod_df["Vest-Afrika"],
                   bw,
                   bottom=prod_df["Kurdistan"] + prod_df["Nordsjøen"],
                   color=C_WA, alpha=0.82,
                   label="Vest-Afrika", edgecolor="white", lw=0.5)

    # Total over søylen
    totals = prod_df["Kurdistan"] + prod_df["Nordsjøen"] + prod_df["Vest-Afrika"]
    for xi, tot in enumerate(totals):
        ax_pr.text(xi, tot + 1.5, f"{tot:.0f}",
                   ha="center", fontsize=7.5, color="#2c3e50", fontweight="bold")

    # Hendelsesmarkører
    ax_pr.axvline(3.5, color=C_ACC, lw=1.4, ls="--", alpha=0.8)
    ax_pr.text(3.56, 172, "Sval-\noppkjop", fontsize=7.2,
               color=C_ACC, fontweight="bold")
    ax_pr.axvline(6.5, color=C_KRD, lw=1.4, ls="--", alpha=0.75)
    ax_pr.text(6.56, 155, "Iran-\nkrig", fontsize=7.2,
               color=C_KRD, fontweight="bold")

    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(prod_df["label"], fontsize=8.5)
    ax_pr.set_ylabel("Netto produksjon (kboepd)", fontsize=9)
    ax_pr.set_title("Produksjon siste 8 kvartal (kboepd)\n"
                    "* Estimert der ikke bekreftet fra rapport",
                    fontsize=9.5, fontweight="bold", loc="left")
    ax_pr.legend(fontsize=8.5, loc="upper left",
                 framealpha=0.88, edgecolor="#dce0e6")
    ax_pr.grid(True, axis="y", alpha=0.15, color="#b2bec3")
    ax_pr.set_ylim(0, 192)

    # ══════════════════════════════════════════════════════════════════════
    # PANEL 3 — Felt-for-felt differensial (lollipop, full bredde)
    # ══════════════════════════════════════════════════════════════════════
    ax_lp = fig.add_subplot(gs[2])
    ax_lp.set_facecolor("#fdfefe")
    for sp in ax_lp.spines.values(): sp.set_color("#dce0e6")

    # Sorter: Kurdistan nederst (størst negativ), NCS over, WA øverst
    lp_df = fields_df.sort_values(
        ["region", "diff_hist"],
        ascending=[False, True]
    ).reset_index(drop=True)

    y_pos = np.arange(len(lp_df))

    # Bakgrunnssoner
    ax_lp.axvspan(-55, 0, alpha=0.02, color=C_KRD)
    ax_lp.axvspan(0, 12, alpha=0.02, color=C_NCS)
    ax_lp.text(-30, len(lp_df) - 0.5, "RABATT TIL BRENT",
               fontsize=8.5, color=C_KRD, alpha=0.40, fontweight="bold", va="top")
    ax_lp.text(1.5, len(lp_df) - 0.5, "PREMIUM",
               fontsize=8.5, color=C_NCS, alpha=0.40, fontweight="bold", va="top")

    for yi, (_, row) in enumerate(lp_df.iterrows()):
        col = C_KRD if row["war_exp"] else REGION_COLORS.get(row["region"], "gray")
        sz  = 60 + row["net_kbpd"] * 20

        # Forbindelseslinje mellom hist og nå
        ax_lp.hlines(yi,
                     min(row["diff_hist"], row["diff_now"]),
                     max(row["diff_hist"], row["diff_now"]),
                     color=col, lw=1.8, alpha=0.30, zorder=2)

        # Historisk (fylt)
        ax_lp.scatter(row["diff_hist"], yi,
                      s=sz, c=col, marker="o",
                      edgecolors="white", lw=0.8, alpha=0.90, zorder=5)
        # Nåværende (åpen)
        ax_lp.scatter(row["diff_now"], yi,
                      s=sz, facecolors="none", edgecolors=col,
                      lw=2.0, alpha=0.90, zorder=5)

        # Produksjonsvolum til venstre av y-akse
        ax_lp.text(-59.5, yi, f"{row['net_kbpd']:.1f}k",
                   va="center", ha="left",
                   fontsize=7.5, color=col, fontweight="bold")

        # Note til høyre — kun for felt med stor endring eller spesielt interessante
        # Hopper over felt med identiske diff-verdier som andre (f.eks Eldfisk = Ekofisk)
        show_note = (
            row["war_exp"]                          # Kurdistan alltid
            or row["name"] in ("Gudrun", "Kvitebjorn", "Martin Linge", "CI-26")
            or abs(row["diff_now"] - row["diff_hist"]) > 2.5
        )
        if show_note:
            x_note = max(row["diff_hist"], row["diff_now"]) + 0.55
            ax_lp.text(x_note, yi, row["note"],
                       va="center", fontsize=7.0,
                       color=col, alpha=0.82, linespacing=1.3)

    ax_lp.axvline(0, color=C_BRT, lw=0.9, alpha=0.45, zorder=3)
    ax_lp.set_yticks(y_pos)
    ax_lp.set_yticklabels(lp_df["name"], fontsize=9.0)
    ax_lp.set_xlabel("Differensial mot Brent (USD/fat)", fontsize=9)
    ax_lp.set_title(
        "Felt-for-felt differensial mot Brent  (post Iran-krig)\n"
        "● fylt = hist. snitt   ○ aapen = naav. estimat   |   kbpd-volum til venstre",
        fontsize=9.5, fontweight="bold", loc="left")
    ax_lp.grid(True, axis="x", alpha=0.15, color="#b2bec3")
    ax_lp.set_xlim(-62, 18)
    ax_lp.set_ylim(-0.8, len(lp_df) - 0.2)

    # Legend
    lp_leg = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#566573",
               markeredgecolor="#566573", markersize=8,
               label="Historisk snitt"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#566573", markersize=8, markeredgewidth=1.8,
               label="Naavarende estimat"),
        mpatches.Patch(color=C_KRD, label="Kurdistan (Krig-eksponert)"),
        mpatches.Patch(color=C_NCS, label="Nordsjoen (Ikke eksponert)"),
        mpatches.Patch(color=C_WA,  label="Vest-Afrika (Ikke eksponert)"),
    ]
    ax_lp.legend(handles=lp_leg, fontsize=7.8, loc="lower right",
                 framealpha=0.90, edgecolor="#dce0e6", ncol=2)

    # ── Lagre ─────────────────────────────────────────────────────────────
    safe = co["ticker"].replace(".", "_")
    out  = OUT_DIR / f"12_prerap_{safe}_Q2_2026.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Rapport lagret: {out}")


if __name__ == "__main__":
    make_report(COMPANY)
