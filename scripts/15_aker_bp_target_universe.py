"""
15_aker_bp_target_universe.py
Aker BP M&A — Potensielle oppkjøpskandidater paa NCS

Bygger videre paa script 14 (M&A-vindusanalyse).
Kartlegger alle realistiske NCS-maal, beregner akkresjon,
og viser passform med Aker BPs portefølje.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

# ── Farger ──────────────────────────────────────────────────────────────────
C_BG     = "#f7f9fc"
C_AKRBP  = "#2c3e50"
C_ACC    = "#27ae60"
C_PREM   = "#c0392b"
C_NEUT   = "#7f8c8d"

# ── Aker BP (kjøper) ───────────────────────────────────────────────────────
AKRBP = dict(
    name        = "Aker BP",
    ev_bn_usd   = 31.0,
    reserves_2p  = 1526,      # mmboe
    ev2p        = 19.7,       # $/boe
    prod_kboepd = 398,
    share_nok   = 340,
    mcap_bn_nok = 224,
)

# ── Kandidater ─────────────────────────────────────────────────────────────
# Sortert etter strategisk passform / sannsynlighet

TARGETS = [
    dict(
        name         = "OKEA ASA",
        ticker       = "OSL: OKEA",
        ev_bn_usd    = 0.33,       # mcap ~$450m - net cash $123m
        reserves_2p  = 76,          # mmboe (redusert fra 93)
        ev2p         = 4.3,         # $/boe — ekstremt billig
        prod_kboepd  = 33,          # 31-35k guide 2026
        key_assets   = "Draugen (45%)\nGjøa (12%)\nBrage (35%)\nHasselmus",
        overlap      = "Ingen direkte, men\nNCS operatørkompetanse",
        owner        = "Spredt (børsnotert)",
        deal_type    = "Kontant (liten deal)",
        probability  = "HØY",
        prob_color   = "#27ae60",
        note         = "Lavest EV/2P på NCS.\nReservenedjustering = billig.\nDraugen er langt nok nord\ntil å gi diversifisering.",
        color        = "#e74c3c",
    ),
    dict(
        name         = "Vår Energi\n(Eni 63%)",
        ticker       = "OSL: VAR",
        ev_bn_usd    = 17.6,       # mcap $12.4bn + net debt $5.2bn
        reserves_2p  = 1294,        # mmboe
        ev2p         = 13.6,        # $/boe — akkretivt!
        prod_kboepd  = 406,         # Q1 2026 record
        key_assets   = "J.Sverdrup (12%)\nBalder/Ringhorne\nJohan Castberg\nGoliat",
        overlap      = "Johan Sverdrup!\n+20% → 51.6% total\n= operatørkontroll?",
        owner        = "Eni SpA (63.1%)",
        deal_type    = "Aksjer + kontant\n(~$18bn mega-deal)",
        probability  = "LAV",
        prob_color   = "#c0392b",
        note         = "Eni er neppe selger.\nMen mest akkretiv deal:\n$13.6 vs $19.7/boe.\nVille gi 51% J.Sverdrup.",
        color        = "#2980b9",
    ),
    dict(
        name         = "Harbour Energy\nNorge (utskilt)",
        ticker       = "LSE: HBR",
        ev_bn_usd    = 3.5,        # estimert: Norge ~40% av $8.4bn EV
        reserves_2p  = 600,         # estimert NCS-andel
        ev2p         = 5.8,         # estimert
        prod_kboepd  = 169,         # 2025 Norge
        key_assets   = "Skarv (operatør!)\nGjøa (28%)\nAasta Hansteen\nNjord, Dvalin",
        overlap      = "Skarv! AKRBP har\n23.8%, HBR operatør.\nKonsoliderer Skarv-hub.",
        owner        = "Børsnotert (UK)",
        deal_type    = "Kontant + aksjer\n(NCS carve-out)",
        probability  = "MIDDELS",
        prob_color   = "#f39c12",
        note         = "Skarv-synergi er nøkkelen.\nHBR kan selge NCS for å\nredusere gjeld post-Wintershall.\n5 subsea-prosjekter i pipeline.",
        color        = "#16a085",
    ),
    dict(
        name         = "DNO NCS-\nportefølje",
        ticker       = "OSL: DNO",
        ev_bn_usd    = 2.5,        # estimert NCS-andel av DNO
        reserves_2p  = 189,         # NCS 2P pro forma m/ Sval
        ev2p         = 13.2,        # estimert
        prod_kboepd  = 80,          # NCS post-Sval
        key_assets   = "Kvitebjørn\nVisund\nFram\nValemon (Equinor-op.)",
        overlap      = "Begrenset direkte,\nmen Nordsjøen-fokus\nkomplementært",
        owner        = "RAK/Bijan Mossavar-\nRahmani (54%)",
        deal_type    = "Kontant\n(NCS carve-out)",
        probability  = "LAV",
        prob_color   = "#c0392b",
        note         = "DNO kjøpte nettopp Sval\nfor å bygge NCS. Neppe selger.\nMen ved strategiskifte\nkan NCS frigjøres.",
        color        = "#8e44ad",
    ),
    dict(
        name         = "Lime Petroleum\n(Rex Int'l)",
        ticker       = "Privat (SGX: 5WH)",
        ev_bn_usd    = 0.15,       # estimert
        reserves_2p  = 20,          # estimert: Brage + Yme-andeler
        ev2p         = 7.5,         # estimert
        prod_kboepd  = 10,
        key_assets   = "Brage (34%)\nYme (25%)\nVette (utvikling)",
        overlap      = "Ingen direkte",
        owner        = "Rex International\nHoldings (Singapore)",
        deal_type    = "Kontant (bolt-on)",
        probability  = "MIDDELS",
        prob_color   = "#f39c12",
        note         = "Typisk bolt-on.\nRex kan ønske exit.\nYme + Brage er modne\nmen Vette gir oppsida.",
        color        = "#95a5a6",
    ),
    dict(
        name         = "INPEX Idemitsu\nNorge",
        ticker       = "TSE: 1605",
        ev_bn_usd    = 0.4,        # estimert NCS-andel
        reserves_2p  = 35,          # estimert: Valhall 10% + Snorre etc.
        ev2p         = 11.4,        # estimert
        prod_kboepd  = 12,          # estimert
        key_assets   = "Valhall (10%)\nHod (10%)\nSnorre (9.6%)\nMistral/Slagugle",
        overlap      = "Valhall! AKRBP\nallerede 90% operatør.\nKonsoliderer til 100%.",
        owner        = "INPEX Corp (Japan)",
        deal_type    = "Kontant (bolt-on)",
        probability  = "HØY",
        prob_color   = "#27ae60",
        note         = "Valhall 10% er den mest\nopplagte bolt-on.\nAKRBP opererer allerede.\nINPEX kan ønske Japan-fokus.",
        color        = "#d4ac0d",
    ),
]


def make_target_universe():
    fig = plt.figure(figsize=(22, 18), facecolor=C_BG)

    # ── Tittel ──────────────────────────────────────────────────────────────
    fig.text(0.04, 0.975,
             "Aker BP — Potensielle oppkjøpskandidater paa NCS",
             fontsize=20, fontweight="bold", color="#1a252f", va="top")
    fig.text(0.04, 0.955,
             "Hvem kan Aker BP kjøpe akkretivt med EV/2P ~$19.7/boe?  "
             "Alle kandidater under denne grensen gir verdiøkning per aksje.",
             fontsize=11, color="#566573", va="top")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[0.55, 0.45],
        width_ratios=[0.55, 0.45],
        top=0.920, bottom=0.05, left=0.06, right=0.97, hspace=0.32, wspace=0.22,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — EV/2P sammenligning (hovedpanel)
    # ════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#fdfefe")
    for sp in ax1.spines.values(): sp.set_color("#dce0e6")

    names = [t["name"] for t in TARGETS]
    ev2ps = [t["ev2p"] for t in TARGETS]
    colors = [t["color"] for t in TARGETS]
    prods = [t["prod_kboepd"] for t in TARGETS]

    y_pos = np.arange(len(TARGETS))
    bar_h = 0.6

    bars = ax1.barh(y_pos, ev2ps, bar_h, color=colors, alpha=0.80,
                    edgecolor="white", lw=1.2)

    # Aker BP EV/2P-linje
    ax1.axvline(AKRBP["ev2p"], color=C_AKRBP, lw=2.5, ls="--", alpha=0.8, zorder=5)
    ax1.text(AKRBP["ev2p"] + 0.3, len(TARGETS) - 0.3,
             f"Aker BP EV/2P = ${AKRBP['ev2p']}/boe",
             fontsize=9, color=C_AKRBP, fontweight="bold", va="bottom")

    # Akkresjonssone
    ax1.axvspan(0, AKRBP["ev2p"], alpha=0.04, color=C_ACC)
    ax1.text(AKRBP["ev2p"] / 2, -0.7, "AKKRETIV SONE",
             fontsize=9, color=C_ACC, fontweight="bold", ha="center", alpha=0.7)

    # Annotér bars
    for yi, t in enumerate(TARGETS):
        # EV/2P verdi
        ax1.text(t["ev2p"] + 0.3, yi,
                 f"${t['ev2p']:.1f}/boe",
                 va="center", fontsize=9, fontweight="bold", color="#2c3e50")
        # Akkresjon
        diff = AKRBP["ev2p"] - t["ev2p"]
        ax1.text(t["ev2p"] + 4.0, yi,
                 f"+${diff:.1f} akkresjon",
                 va="center", fontsize=7.5, color=C_ACC, fontstyle="italic")
        # Produksjon
        ax1.text(-0.5, yi + 0.28,
                 f"{t['prod_kboepd']} kboepd",
                 va="center", ha="right", fontsize=7, color="#7f8c8d")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("EV/2P (USD per fat 2P-reserver)", fontsize=10)
    ax1.set_xlim(-1, 25)
    ax1.invert_yaxis()
    ax1.grid(True, axis="x", alpha=0.15, color="#b2bec3")
    ax1.set_title(
        "Implisitt reservepris — alle under Aker BPs EV/2P er akkretive\n"
        "Stiplet linje = Aker BPs nåværende verdsettelse ($19.7/boe)",
        fontsize=11, fontweight="bold", loc="left", pad=10)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Bobler: Produksjon vs 2P-reserver (størrelsesfokus)
    # ════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#fdfefe")
    for sp in ax2.spines.values(): sp.set_color("#dce0e6")

    # Manuell offset for å unngå label-overlapp
    label_offsets = {
        "OKEA ASA":                (15, 12),
        "Vår Energi\n(Eni 63%)":  (-20, 15),
        "Harbour Energy\nNorge (utskilt)": (-30, 10),
        "DNO NCS-\nportefølje":   (15, 12),
        "Lime Petroleum\n(Rex Int'l)":  (15, 18),
        "INPEX Idemitsu\nNorge":  (15, -18),
    }
    for t in TARGETS:
        size = max(t["ev_bn_usd"] * 150, 60)  # boblestørrelse ~ dealverdi
        ax2.scatter(t["reserves_2p"], t["prod_kboepd"], s=size,
                    c=t["color"], alpha=0.75, edgecolors="white", lw=1.5, zorder=5)
        # Label
        ox, oy = label_offsets.get(t["name"], (15, 8))
        ha = "left" if ox > 0 else "right"
        ax2.annotate(
            t["name"].replace("\n", " "),
            (t["reserves_2p"], t["prod_kboepd"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=8, fontweight="bold", color=t["color"],
            ha=ha,
            arrowprops=dict(arrowstyle="-", color=t["color"], lw=0.5, alpha=0.5),
        )

    # Aker BP som referanse
    ax2.scatter(AKRBP["reserves_2p"], AKRBP["prod_kboepd"], s=800,
                c=C_AKRBP, alpha=0.25, edgecolors=C_AKRBP, lw=2, zorder=3)
    ax2.text(AKRBP["reserves_2p"], AKRBP["prod_kboepd"] + 15,
             "Aker BP", fontsize=10, fontweight="bold", color=C_AKRBP, ha="center")

    ax2.set_xlabel("2P-reserver (mmboe)", fontsize=10)
    ax2.set_ylabel("Produksjon (kboepd)", fontsize=10)
    ax2.set_title(
        "Størrelse: Produksjon vs reserver\n"
        "Boblestørrelse = estimert EV (dealverdi)",
        fontsize=11, fontweight="bold", loc="left", pad=10)
    ax2.grid(True, alpha=0.15, color="#b2bec3")
    ax2.set_xlim(-50, 1700)
    ax2.set_ylim(-10, 460)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Strategisk passform (kvalitativ)
    # ════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#fdfefe")
    for sp in ax3.spines.values(): sp.set_visible(False)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-0.5, len(TARGETS) + 0.2)
    ax3.invert_yaxis()
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Kolonneoverskrifter
    col_x = [0.15, 1.8, 3.7, 5.5, 7.0, 8.6]
    headers = ["Kandidat", "Nøkkelaktiva", "Porteføljeoverlapp",
               "Eier / Dealtype", "Sannsynlighet", "Vurdering"]

    for x, h in zip(col_x, headers):
        ax3.text(x, -0.35, h, fontsize=9.5, fontweight="bold",
                 color="#2c3e50", va="center")

    # Horisontale skillelinjer
    for yi in range(len(TARGETS) + 1):
        ax3.axhline(yi - 0.5, color="#dce0e6", lw=0.5, alpha=0.7)

    # Data
    for yi, t in enumerate(TARGETS):
        # Bakgrunnsfarge annenhver rad
        if yi % 2 == 0:
            ax3.axhspan(yi - 0.5, yi + 0.5, alpha=0.03, color="#2c3e50")

        # Navn + ticker
        ax3.text(col_x[0], yi, t["name"].replace("\n", " "),
                 fontsize=8.5, fontweight="bold", color=t["color"], va="center")
        ax3.text(col_x[0], yi + 0.25, t["ticker"],
                 fontsize=6.5, color="#95a5a6", va="center")

        # Nøkkelaktiva
        ax3.text(col_x[1], yi, t["key_assets"],
                 fontsize=7, color="#2c3e50", va="center", linespacing=1.2)

        # Overlap
        ax3.text(col_x[2], yi, t["overlap"],
                 fontsize=7, color="#2c3e50", va="center", linespacing=1.2)

        # Eier / Dealtype
        owner_text = f"{t['owner']}\n{t['deal_type']}"
        ax3.text(col_x[3], yi, owner_text,
                 fontsize=7, color="#2c3e50", va="center", linespacing=1.2)

        # Sannsynlighet
        ax3.text(col_x[4], yi, t["probability"],
                 fontsize=9, fontweight="bold", color=t["prob_color"],
                 va="center", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec=t["prob_color"], lw=1.0))

        # Vurdering
        ax3.text(col_x[5], yi, t["note"],
                 fontsize=6.5, color="#566573", va="center", linespacing=1.3)

    ax3.set_title(
        "Strategisk passform og sannsynlighetsvurdering",
        fontsize=12, fontweight="bold", loc="left", pad=12)

    # ── Footer ──────────────────────────────────────────────────────────────
    fig.text(0.04, 0.015,
             "Kilder: Yahoo Finance, OKEA IR, Vår Energi IR, Harbour Energy IR, "
             "DNO IR, Lime Petroleum IR, SODIR/NPD, Pareto Sec., Citi Research.\n"
             "EV/2P-estimater er basert paa offentlig tilgjengelig data per mai 2026. "
             "Sannsynlighetsvurdering er kvalitativ og gjenspeiler forfatterens syn.",
             fontsize=7.5, color="#95a5a6", style="italic")

    # ── Lagre ───────────────────────────────────────────────────────────────
    out = OUT_DIR / "15_aker_bp_target_universe.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Analyse lagret: {out}")


if __name__ == "__main__":
    make_target_universe()
