"""
Pre-rapport mal — Aker BP ASA, Q2 2026  (rapport 7. mai 2026)

Alle felt er på NCS (norsk sokkel).
- Ingen Hormuz-eksponering.
- Iran-krigen ga NCS-PREMIUM (positiv for Aker BP, i motsetning til DNO Kurdistan).
- Nøkkelspørsmål: Realiseres full Hormuz-krise-premium i Q2 2026?
  (Vår Energi bekreftet kvartalsforsinkelse i NCS-cargoprisingen)

Produksjonsområder:
  Johan Sverdrup — medium sour (API 28, S 0.80 %) → dominerer volum (~53 %)
  Alvheim omr.   — lett søt (API 34, S 0.18 %)  → høyest premium
  Valhall omr.   — lett søt (API 36, S 0.10 %)  → inkl. Hod
  Grieg/Aasen    — medium (API 32, S 0.42 %)    → inkl. Symra fra apr 2026
  Skarv omr.     — lett kondensat (API 43, S 0.21 %) → norsk hav
  Ula omr.       — moden (API 35, S 0.09 %)     → liten andel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

PROJECT_ROOT   = Path(__file__).parent.parent
BRENT_CSV      = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
SODIR_MONTHLY  = PROJECT_ROOT / "data" / "raw" / "sodir" / "sodir_field_production_monthly.csv"
OUT_DIR        = PROJECT_ROOT / "data" / "processed"

# ── Farger ──────────────────────────────────────────────────────────────────
C_JSVD  = "#2c3e50"   # mørk navy   — Johan Sverdrup (medium sour, dominerende)
C_ALVM  = "#16a085"   # teal        — Alvheim omr. (lett søt, høyest premium)
C_VLHH  = "#2980b9"   # blå         — Valhall omr. (inkl. Hod)
C_GRGS  = "#6c3483"   # lilla       — Grieg/Aasen (inkl. Symra fra apr 2026)
C_SKARV = "#ca6f1e"   # brent oransj— Skarv omr. (norsk hav, lett kondensat)
C_ULA   = "#95a5a6"   # grå         — Ula omr. (moden, liten)
C_BRT   = "#7f8c8d"   # grå         — Brent
C_ACC   = "#e67e22"   # oransje     — aksent/nøytrale events
C_POS   = "#27ae60"   # grønn       — positive events
C_NEG   = "#c0392b"   # rød         — negative events

PROD_COLORS = {
    "Johan Sverdrup": C_JSVD,
    "Alvheim omr.":   C_ALVM,
    "Valhall omr.":   C_VLHH,
    "Grieg/Aasen":    C_GRGS,
    "Skarv omr.":     C_SKARV,
    "Ula omr.":       C_ULA,
}

# Felt → produksjonsområde (for fargekobling til stacked bar og lollipop)
FIELD_AREA = {
    "Johan Sverdrup":     "Johan Sverdrup",
    "Alvheim blend":      "Alvheim omr.",
    "Valhall/Hod":        "Valhall omr.",
    "Edvard Grieg/Aasen": "Grieg/Aasen",
    "Skarv blend":        "Skarv omr.",
    "Ula":                "Ula omr.",
    "Symra":              "Grieg/Aasen",
}

# ── SODIR-datagrunnlag for lollipop ─────────────────────────────────────────
# Aker BP eierandeler per felt (kilde: selskapets egne presentasjoner)
AKER_BP_WI = {
    "JOHAN SVERDRUP": 0.3157,
    "VALHALL":        0.900,
    "SKARV":          0.238,
    "ALVHEIM":        0.650,
    "EDVARD GRIEG":   0.650,
    "TYRVING":        0.650,
    "HOD":            0.900,
    "IVAR AASEN":     0.348,
    "TAMBAR":         0.550,
    "TAMBAR ØST":     0.550,
    "ULA":            0.800,
    "BØYLA":          0.650,
    "SKOGUL":         0.650,
}

# Lesevennlige feltnavn + differensial + område per felt
FIELD_CONFIG = {
    "JOHAN SVERDRUP": dict(display="Johan Sverdrup",  area="Johan Sverdrup",
                           diff_hist=-2.0, diff_now=+0.5,
                           note="Medium sour — NCS\nnon-Hormuz premium"),
    "VALHALL":        dict(display="Valhall",          area="Valhall omr.",
                           diff_hist=+2.0, diff_now=+5.5,
                           note="Kritt-reservoar"),
    "SKARV":          dict(display="Skarv/Aerfugl",   area="Skarv omr.",
                           diff_hist=+1.5, diff_now=+5.0,
                           note="Lett kondensat\nnorsk hav"),
    "ALVHEIM":        dict(display="Alvheim (FPSO)",   area="Alvheim omr.",
                           diff_hist=+3.5, diff_now=+7.0,
                           note="Urals-erstatter\nhoy premium"),
    "EDVARD GRIEG":   dict(display="Edvard Grieg",     area="Grieg/Aasen",
                           diff_hist=-0.5, diff_now=+3.5,
                           note="Grane-stream"),
    "TYRVING":        dict(display="Tyrving",          area="Alvheim omr.",
                           diff_hist=+3.5, diff_now=+7.0,
                           note="Sep 2024"),
    "HOD":            dict(display="Hod",              area="Valhall omr.",
                           diff_hist=+2.0, diff_now=+5.5,
                           note=""),
    "IVAR AASEN":     dict(display="Ivar Aasen",       area="Grieg/Aasen",
                           diff_hist=-0.3, diff_now=+3.5,
                           note=""),
    "TAMBAR":         dict(display="Tambar",           area="Ula omr.",
                           diff_hist=+1.5, diff_now=+4.5,
                           note=""),
    "TAMBAR ØST":     dict(display="Tambar Øst",       area="Ula omr.",
                           diff_hist=+1.5, diff_now=+4.5,
                           note=""),
    "ULA":            dict(display="Ula",              area="Ula omr.",
                           diff_hist=+1.5, diff_now=+4.5,
                           note="Moden felt"),
    "BØYLA":          dict(display="Bøyla",            area="Alvheim omr.",
                           diff_hist=+3.5, diff_now=+7.0,
                           note=""),
    "SKOGUL":         dict(display="Skogul",           area="Alvheim omr.",
                           diff_hist=+3.5, diff_now=+7.0,
                           note=""),
}

# Symra: etter SODIR-cutoff (apr 2026), legges til manuelt
SYMRA_MANUAL = dict(name="Symra", area="Grieg/Aasen", net_kboepd=14.0,
                    diff_hist=+0.5, diff_now=+4.5,
                    note="Apr 2026\n(Aasen tieback)")


def load_aker_bp_lollipop_fields(n=12, add_symra=True):
    """
    Les SODIR månedlig produksjon, påfør Aker BP-eierandel og
    returner topp-n felt som liste med dicts klar for lollipop-plot.

    Konvertering: Mill Sm3 BOE × 6290 kbbl/Mill Sm3 / 365 dager = kboepd
    """
    sodir = pd.read_csv(SODIR_MONTHLY)

    # Siste 12 måneder av tilgjengelig data
    sodir["date"] = pd.to_datetime(
        {"year": sodir["prfYear"], "month": sodir["prfMonth"], "day": 1}
    )
    latest_month = sodir["date"].max()
    start_month  = latest_month - pd.DateOffset(months=11)

    recent = sodir[
        (sodir["date"] >= start_month) &
        (sodir["prfInformationCarrier"].isin(AKER_BP_WI))
    ].copy()

    # Summer felt og konverter til kboepd
    agg = (
        recent
        .groupby("prfInformationCarrier")["prfPrdOeNetMillSm3"]
        .sum()
        .reset_index()
        .rename(columns={"prfPrdOeNetMillSm3": "mill_sm3_12m"})
    )
    agg["gross_kboepd"] = agg["mill_sm3_12m"] * 6290 / 365
    agg["wi"]           = agg["prfInformationCarrier"].map(AKER_BP_WI)
    agg["net_kboepd"]   = agg["gross_kboepd"] * agg["wi"]

    top = agg.nlargest(n, "net_kboepd").reset_index(drop=True)

    records = []
    for _, row in top.iterrows():
        cfg = FIELD_CONFIG[row["prfInformationCarrier"]]
        records.append(dict(
            name      = cfg["display"],
            area      = cfg["area"],
            net_kboepd= round(row["net_kboepd"], 1),
            diff_hist = cfg["diff_hist"],
            diff_now  = cfg["diff_now"],
            note      = cfg["note"],
        ))

    if add_symra:
        records.append(SYMRA_MANUAL)

    print(f"  [SODIR] Periode: {start_month:%b %Y} – {latest_month:%b %Y}")
    for r in sorted(records, key=lambda x: -x["net_kboepd"]):
        print(f"    {r['name']:<22} {r['net_kboepd']:>6.1f} kboepd")
    return records


# ─────────────────────────────────────────────────────────────────────────────
COMPANY = dict(
    name        = "Aker BP ASA",
    ticker      = "AKRBP.OL",
    report_date = "7. mai 2026",
    brent_now   = 103.0,

    # ── Realiserte priser — én blandet linje (bekreftet fra trading-oppdateringer) ──
    # (år, kvartal, realisert_usd, brent_snitt_usd, bekreftet)
    realized = [
        (2022, 1,  100.9,  99.1, True),
        (2022, 2,  117.5, 114.2, True),
        (2022, 3,  101.1,  99.0, True),
        (2022, 4,   86.6,  85.2, True),
        (2023, 1,   78.4,  81.3, True),
        (2023, 2,   76.8,  78.0, True),
        (2023, 3,   87.6,  85.9, True),
        (2023, 4,   83.6,  80.3, True),
        (2024, 1,   82.7,  82.1, True),
        (2024, 2,   83.1,  85.0, True),
        (2024, 3,   80.3,  80.4, True),
        (2024, 4,   74.1,  74.5, True),
        (2025, 1,   75.0,  76.2, True),
        (2025, 2,   66.9,  64.5, True),
        (2025, 3,   70.3,  67.0, True),
        (2025, 4,   63.1,  62.5, True),
        (2026, 1,   82.2,  80.7, True),   # Iran-krig NCS-premium
    ],

    # ── Historiske hendelser ─────────────────────────────────────────────────
    # (dato, label, type: pos/neg/neutral, y_side: +1 over / -1 under akse)
    events = [
        ("2022-02-24", "Russland\nangrir Ukraina",    "neg",     -1),
        ("2022-06-30", "Lundin-fusjon\nfullfort",     "pos",     +1),
        ("2022-12-15", "J.Sverdrup\nFase 2 opp",      "pos",     +1),
        ("2024-09-03", "Tyrving\nforste olje",         "pos",     +1),
        ("2026-02-28", "Iran-krig\n(NCS premium!)",    "pos",     -1),
        ("2026-04-07", "Symra-\noppstart",             "pos",     +1),
        ("2026-05-07", "Q1-26\nRapport",               "neutral", +1),
    ],

    # ── Feltdata (for scatter + lollipop) ────────────────────────────────────
    # medium_sour=True → diamant-markør (Johan Sverdrup skiller seg ut)
    fields = [
        dict(name="Johan Sverdrup",     api=28.0, sulfur=0.80,
             net_kboepd=210, medium_sour=True,
             diff_hist=-2.0, diff_now=+0.5,
             note="Medium sour — men\nNCS non-Hormuz\npremium kompenserte"),
        dict(name="Alvheim blend",      api=33.9, sulfur=0.18,
             net_kboepd=55, medium_sour=False,
             diff_hist=+3.5, diff_now=+7.0,
             note="Lett sort — Urals-\nerstatter, hoy premium"),
        dict(name="Valhall/Hod",        api=36.0, sulfur=0.10,
             net_kboepd=47, medium_sour=False,
             diff_hist=+2.0, diff_now=+5.5,
             note="Kritt-reservoar\nlett sort"),
        dict(name="Edvard Grieg/Aasen", api=32.0, sulfur=0.42,
             net_kboepd=43, medium_sour=False,
             diff_hist=-0.4, diff_now=+4.0,
             note="Blandet via\nGrane-stroem"),
        dict(name="Skarv blend",        api=43.0, sulfur=0.21,
             net_kboepd=30, medium_sour=False,
             diff_hist=+1.5, diff_now=+5.0,
             note="Lett kondensat\n(norsk hav)"),
        dict(name="Symra",              api=45.0, sulfur=0.05,
             net_kboepd=3,  medium_sour=False,
             diff_hist=+0.5, diff_now=+4.5,
             note="Apr 2026\n(20-25k plateau)"),
        dict(name="Ula",                api=35.0, sulfur=0.09,
             net_kboepd=6,  medium_sour=False,
             diff_hist=+1.5, diff_now=+4.5,
             note="Moden felt"),
    ],

    # ── Produksjon siste 8 kvartal (kboepd netto egenandel) ─────────────────
    # (label, dato, Johan Sverdrup, Alvheim omr., Valhall omr., Grieg/Aasen, Skarv omr., Ula omr.)
    # Totaler bekreftet fra trading-oppdateringer; area-split estimert fra investor-presentasjoner.
    production = [
        ("Q2-24", "2024-05-15", 241, 50, 49, 58, 36, 10),  # total 444 ✓
        ("Q3-24", "2024-08-15", 237, 50, 47, 50, 24,  7),  # total 415 ✓
        ("Q4-24", "2024-11-15", 239, 68, 47, 55, 34,  6),  # total 449 ✓  Tyrving ramp-up
        ("Q1-25", "2025-02-15", 236, 68, 48, 48, 35,  6),  # total 441 ✓
        ("Q2-25", "2025-05-15", 238, 62, 30, 46, 32,  7),  # total 415 ✓  Valhall 1-mnd. nedstenging
        ("Q3-25", "2025-08-15", 235, 60, 35, 45, 33,  6),  # total 414 ✓
        ("Q4-25", "2024-11-15", 217, 58, 43, 42, 45,  6),  # total 411 ✓  J.Sverdrup-nedgang starter
        ("Q1-26", "2026-02-15", 210, 55, 42, 50, 35,  6),  # total 398 ✓  Symra i Grieg/Aasen apr26
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
def quarter_to_date(year, q):
    return pd.Timestamp(f"{year}-{q*3-1:02d}-15")


def make_report(co):
    fields_df = pd.DataFrame(co["fields"])
    prod_cols  = ["label", "date",
                  "Johan Sverdrup", "Alvheim omr.", "Valhall omr.",
                  "Grieg/Aasen", "Skarv omr.", "Ula omr."]
    prod_df    = pd.DataFrame(co["production"], columns=prod_cols)
    prod_df["date"] = pd.to_datetime(prod_df["date"])

    # Realiserte priser
    real_df = pd.DataFrame(co["realized"],
                           columns=["year", "q", "realized", "brent", "confirmed"])
    real_df["date"] = real_df.apply(lambda r: quarter_to_date(r.year, r.q), axis=1)
    real_df = real_df.sort_values("date")

    # Brent daglig
    brent_d = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    brent_d = brent_d[brent_d["date"] >= "2022-01-01"]

    # ── Figur ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 22), facecolor="#f7f9fc")

    # Tittel
    fig.text(0.05, 0.975,
             f"{co['name']}  ({co['ticker']})",
             fontsize=17, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.05, 0.954,
             f"Pre-rapport analyse Q2 2026  |  Rapport: {co['report_date']}  "
             f"|  Brent: ~${co['brent_now']:.0f}/fat  (mai 2026)  "
             f"|  Alle felt: NCS — ingen Hormuz-eksponering",
             fontsize=10, color="#566573", va="top")
    fig.text(0.97, 0.954,
             "Det ene sporsmalet:\nRealiseres full krise-\npremium i Q2 2026?\n"
             "(NCS cargo-lag — bekreft-\net av Var Energi Q1-26)",
             fontsize=8.5, color=C_ALVM, fontweight="bold",
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec=C_ALVM, lw=1.0))

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.30, 0.33, 0.33],
        top=0.922, bottom=0.035, left=0.09, right=0.94, hspace=0.30,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Realisert pris + Brent + hendelsestidslinje
    # ══════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    # Brent daglig
    ax1.plot(brent_d["date"], brent_d["brent_usd"],
             color=C_BRT, lw=1.0, alpha=0.35, zorder=2, label="Brent spot (daglig)")
    ax1.fill_between(brent_d["date"], brent_d["brent_usd"], alpha=0.04, color=C_BRT)

    # Realisert pris — én enkelt linje (blandet NCS)
    conf  = real_df[real_df["confirmed"]]
    estim = real_df[~real_df["confirmed"]]
    ax1.plot(real_df["date"], real_df["realized"],
             color=C_JSVD, lw=2.4, zorder=5, label="Aker BP realisert (blandet)")
    ax1.scatter(conf["date"], conf["realized"],
                s=55, color=C_JSVD, marker="o",
                edgecolors="white", lw=0.8, zorder=6)
    ax1.scatter(estim["date"], estim["realized"],
                s=45, color=C_JSVD, marker="o",
                facecolors="none", edgecolors=C_JSVD, lw=1.5, zorder=6)

    # Annotér Q1-26 og Q4-25
    last = real_df.iloc[-1]
    prev = real_df.iloc[-2]
    ax1.annotate(f"Q1-26: ${last.realized:.1f}",
                 (last.date, last.realized),
                 xytext=(8, 6), textcoords="offset points",
                 fontsize=8.5, color=C_JSVD, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_JSVD, lw=0.8))
    ax1.annotate(f"Q4-25: ${prev.realized:.1f}",
                 (prev.date, prev.realized),
                 xytext=(8, -18), textcoords="offset points",
                 fontsize=8, color=C_BRT,
                 arrowprops=dict(arrowstyle="-", color=C_BRT, lw=0.7))

    # Hendelsesmarkorer — sykler gjennom høydenivåer
    ev_styles = {
        "pos":     dict(color=C_POS, lw=1.4, ls="--"),
        "neg":     dict(color=C_NEG, lw=1.4, ls="--"),
        "neutral": dict(color=C_ACC, lw=1.4, ls="--"),
    }
    y_max = 142
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

    ax1.set_ylabel("Realisert / Brent (USD/boe)", fontsize=9)
    ax1.set_title("Realiserte priser (blandet NCS) + historiske aksjetriggere\n"
                  "● fylt = bekreftet  ○ aapen = estimert",
                  fontsize=9.5, fontweight="bold", loc="left")
    ax1.set_ylim(12, y_max)
    ax1.set_xlim(brent_d["date"].min(), pd.Timestamp("2026-07-01"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(True, alpha=0.15, color="#b2bec3")
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.88, edgecolor="#dce0e6")

    # ══════════════════════════════════════════════════════════════════════════
    # RAD 2 — [Feltscatter | Produksjon 8 kvartal]
    # ══════════════════════════════════════════════════════════════════════════
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.38)
    ax_sc = fig.add_subplot(inner2[0])
    ax_pr = fig.add_subplot(inner2[1])

    for ax in [ax_sc, ax_pr]:
        ax.set_facecolor("#fdfefe")
        for sp in ax.spines.values(): sp.set_color("#dce0e6")

    # ── Feltscatter ──────────────────────────────────────────────────────────
    label_offsets = {
        "Johan Sverdrup":      (+6,  +8),
        "Alvheim blend":       (+6,  +8),
        "Valhall/Hod":         (+6,  +8),
        "Edvard Grieg/Aasen":  (+6, -16),
        "Skarv blend":         (+6,  +8),
        "Symra":               (-58, +8),
        "Ula":                 (+6, -16),
    }

    for _, r in fields_df.iterrows():
        area_col = PROD_COLORS.get(FIELD_AREA.get(r["name"], "Grieg/Aasen"), C_JSVD)
        mk = "D" if r["medium_sour"] else "o"
        sz = 50 + r["net_kboepd"] * 1.8
        ax_sc.scatter(r["api"], r["sulfur"],
                      s=sz, c=area_col, marker=mk,
                      edgecolors="white", lw=1.0, alpha=0.88, zorder=5)
        dx, dy = label_offsets.get(r["name"], (+6, +6))
        ax_sc.annotate(r["name"],
                       (r["api"], r["sulfur"]),
                       xytext=(dx, dy), textcoords="offset points",
                       fontsize=7.2, color="#2c3e50", zorder=6, linespacing=1.3)

    # Dekorasjon
    ax_sc.add_patch(plt.Rectangle((30, 0), 12, 0.5,
                                  alpha=0.07, color="green"))
    ax_sc.text(30.3, 0.02, "Sweet spot", fontsize=7, color="darkgreen", alpha=0.75)
    ax_sc.axhline(0.5, color="#aab7b8", ls="--", lw=0.7)

    # NCS-premium-boks (erstatter Kurdistan-boks fra DNO)
    ax_sc.text(37.0, 0.72,
               "Alle felt: NCS — fikk\nnon-Hormuz premium\nunder Iran-krigen",
               fontsize=7.0, color=C_POS, style="italic",
               bbox=dict(boxstyle="round,pad=0.25", fc="#eafaf1",
                         ec=C_POS, lw=0.7, alpha=0.9))

    ax_sc.set_xlabel("API-grad  (hoeyere = lettere)", fontsize=9)
    ax_sc.set_ylabel("Svovel (%)", fontsize=9)
    ax_sc.set_title("Feltportefolje — Oljekvalitet (NCS)\n"
                    "◆ Medium sour (J.Sverdrup)   ● Lett/sort (oevrige)",
                    fontsize=9.5, fontweight="bold", loc="left")
    ax_sc.set_xlim(24, 52)
    ax_sc.set_ylim(-0.05, 1.05)
    ax_sc.grid(True, alpha=0.15, color="#b2bec3")

    sc_legend = [
        mpatches.Patch(color=C_JSVD,  label="Johan Sverdrup (med. sour)"),
        mpatches.Patch(color=C_ALVM,  label="Alvheim omr."),
        mpatches.Patch(color=C_VLHH,  label="Valhall omr."),
        mpatches.Patch(color=C_GRGS,  label="Grieg/Aasen"),
        mpatches.Patch(color=C_SKARV, label="Skarv omr."),
        mpatches.Patch(color=C_ULA,   label="Ula omr."),
    ]
    ax_sc.legend(handles=sc_legend, fontsize=7.0, loc="upper right",
                 framealpha=0.88, edgecolor="#dce0e6", ncol=2)

    # ── Produksjon stacked bar (6 områder) ────────────────────────────────────
    x  = np.arange(len(prod_df))
    bw = 0.60

    # Kumulativ bunn for stacking
    bot = pd.Series([0] * len(prod_df))
    bar_specs = [
        ("Johan Sverdrup", C_JSVD),
        ("Alvheim omr.",   C_ALVM),
        ("Valhall omr.",   C_VLHH),
        ("Grieg/Aasen",    C_GRGS),
        ("Skarv omr.",     C_SKARV),
        ("Ula omr.",       C_ULA),
    ]
    for col, color in bar_specs:
        ax_pr.bar(x, prod_df[col], bw, bottom=bot,
                  color=color, alpha=0.87, label=col,
                  edgecolor="white", lw=0.5)
        bot = bot + prod_df[col]

    # Total over søyle
    for xi, tot in enumerate(bot):
        ax_pr.text(xi, tot + 4, f"{tot:.0f}",
                   ha="center", fontsize=7.5, color="#2c3e50", fontweight="bold")

    # Hendelsesmarkorer
    # Tyrving (sep 2024) — mellom Q3-24 (idx 1) og Q4-24 (idx 2)
    ax_pr.axvline(1.5, color=C_ALVM, lw=1.4, ls="--", alpha=0.80)
    ax_pr.text(1.56, 450, "Tyrving\nforste olje", fontsize=7.2,
               color=C_ALVM, fontweight="bold")
    # J.Sverdrup naturlig nedgang starter Q4-25 — mellom Q3-25 (idx 5) og Q4-25 (idx 6)
    ax_pr.axvline(5.5, color=C_JSVD, lw=1.2, ls=":", alpha=0.70)
    ax_pr.text(5.56, 430, "J.Sverdrup\nnedgang", fontsize=7.0,
               color=C_JSVD, fontweight="bold")
    # Iran-krig (feb 2026) — mellom Q4-25 (idx 6) og Q1-26 (idx 7)
    ax_pr.axvline(6.5, color=C_NEG, lw=1.4, ls="--", alpha=0.70)
    ax_pr.text(6.56, 370, "Iran-krig\n(pris-hopp)", fontsize=7.2,
               color=C_NEG, fontweight="bold")

    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(prod_df["label"], fontsize=8.5)
    ax_pr.set_ylabel("Netto produksjon (kboepd)", fontsize=9)
    ax_pr.set_title("Produksjon siste 8 kvartal (kboepd)\n"
                    "* Bekreftet totaler; area-split estimert",
                    fontsize=9.5, fontweight="bold", loc="left")
    ax_pr.legend(fontsize=7.5, loc="upper right",
                 framealpha=0.88, edgecolor="#dce0e6", ncol=2)
    ax_pr.grid(True, axis="y", alpha=0.15, color="#b2bec3")
    ax_pr.set_ylim(0, 540)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Felt-for-felt differensial (lollipop)
    # ══════════════════════════════════════════════════════════════════════════
    ax_lp = fig.add_subplot(gs[2])
    ax_lp.set_facecolor("#fdfefe")
    for sp in ax_lp.spines.values(): sp.set_color("#dce0e6")

    # Last topp-12 felt fra SODIR (faktisk produksjon, Aker BP-eierandel påført)
    lp_records = load_aker_bp_lollipop_fields(n=12, add_symra=True)
    lp_df = pd.DataFrame(lp_records).sort_values("net_kboepd", ascending=True).reset_index(drop=True)
    y_pos  = np.arange(len(lp_df))

    # Bakgrunnssoner
    ax_lp.axvspan(-5, 0, alpha=0.03, color=C_NEG)
    ax_lp.axvspan(0, 12, alpha=0.03, color=C_POS)
    ax_lp.text(-3.5, len(lp_df) - 0.5, "RABATT",
               fontsize=8.5, color=C_NEG, alpha=0.40, fontweight="bold", va="top")
    ax_lp.text(0.4, len(lp_df) - 0.5, "PREMIUM",
               fontsize=8.5, color=C_POS, alpha=0.40, fontweight="bold", va="top")

    for yi, (_, row) in enumerate(lp_df.iterrows()):
        area_col = PROD_COLORS.get(row.get("area", "Grieg/Aasen"), C_JSVD)
        # Skaler boble: Johan Sverdrup er 210k — begrens maks størrelse for leselighet
        sz = 40 + min(row["net_kboepd"], 80) * 2.5

        # Forbindelseslinje
        ax_lp.hlines(yi,
                     min(row["diff_hist"], row["diff_now"]),
                     max(row["diff_hist"], row["diff_now"]),
                     color=area_col, lw=1.8, alpha=0.30, zorder=2)

        # Historisk (fylt)
        ax_lp.scatter(row["diff_hist"], yi,
                      s=sz, c=area_col, marker="o",
                      edgecolors="white", lw=0.8, alpha=0.90, zorder=5)
        # Nåværende (åpen)
        ax_lp.scatter(row["diff_now"], yi,
                      s=sz, facecolors="none", edgecolors=area_col,
                      lw=2.0, alpha=0.90, zorder=5)

        # Volum til venstre
        ax_lp.text(-5.8, yi, f"{row['net_kboepd']:.0f}k",
                   va="center", ha="left",
                   fontsize=7.5, color=area_col, fontweight="bold")

        # Note til høyre — bare der det er noe å si
        note = row.get("note", "")
        if note:
            x_note = max(row["diff_hist"], row["diff_now"]) + 0.35
            ax_lp.text(x_note, yi, note,
                       va="center", fontsize=6.8,
                       color=area_col, alpha=0.85, linespacing=1.3)

    ax_lp.axvline(0, color=C_BRT, lw=0.9, alpha=0.45, zorder=3)
    ax_lp.set_yticks(y_pos)
    ax_lp.set_yticklabels(lp_df["name"], fontsize=8.5)
    ax_lp.set_xlabel("Differensial mot Brent (USD/fat)", fontsize=9)
    ax_lp.set_title(
        "Felt-for-felt differensial mot Brent  (topp 11 felt, post Iran-krig)\n"
        "● fylt = hist. snitt   ○ aapen = naav. estimat   |   kboepd til venstre",
        fontsize=9.5, fontweight="bold", loc="left")
    ax_lp.grid(True, axis="x", alpha=0.15, color="#b2bec3")
    ax_lp.set_xlim(-6.5, 15)
    ax_lp.set_ylim(-0.8, len(lp_df) - 0.2)

    # Innsikt-boks: selv medium sour (J.Sverdrup) gikk fra rabatt til premium
    ax_lp.text(6.5, 0.3,
               "Geografisk tilgjengelighet trumfer kvalitet:\n"
               "Selv Johan Sverdrup (medium sour) fikk premium\n"
               "fordi NCS = ikke Hormuz-eksponert.",
               fontsize=7.8, color="#2c3e50",
               bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7",
                         ec=C_ACC, lw=0.8, alpha=0.92))

    lp_leg = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#566573",
               markeredgecolor="#566573", markersize=8, label="Historisk snitt"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#566573", markersize=8, markeredgewidth=1.8,
               label="Naav. estimat (post krise)"),
        mpatches.Patch(color=C_JSVD,  label="Johan Sverdrup"),
        mpatches.Patch(color=C_ALVM,  label="Alvheim omr."),
        mpatches.Patch(color=C_VLHH,  label="Valhall omr."),
        mpatches.Patch(color=C_GRGS,  label="Grieg/Aasen"),
        mpatches.Patch(color=C_SKARV, label="Skarv omr."),
        mpatches.Patch(color=C_ULA,   label="Ula omr."),
    ]
    ax_lp.legend(handles=lp_leg, fontsize=7.5, loc="lower right",
                 framealpha=0.90, edgecolor="#dce0e6", ncol=2)

    # ── Lagre ──────────────────────────────────────────────────────────────────
    safe = co["ticker"].replace(".", "_")
    out  = OUT_DIR / f"13_prerap_{safe}_Q2_2026.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Rapport lagret: {out}")


if __name__ == "__main__":
    make_report(COMPANY)
