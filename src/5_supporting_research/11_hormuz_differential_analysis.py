"""
Steg 11: Hormuz-effekten — endringer i råolje-differensialer og
          realiseringspriser under Hormuz-stengingen mars 2026.

HENDELSESFORLØP:
  28. feb 2026 : USA/Israel-luftangrep mot Iran starter
  1–4. mars    : IRGC angriper skip, de facto stengsel av Hormuz-stredet
  12. mars     : Brent over $100/fat for første gang siden 2022
  19. mars     : Dubai crude ALL-TIME HIGH $166/fat
  31. mars     : Brent-WTI spread peak $25/fat (høyest på 5+ år)
  Mnd slutt    : Brent avslutter kvartalet på $118/fat — størst kvartalsstigning
                 på inflasjonsjustert basis siden 1988 (EIA)

NØKKELDATA FRA SELSKAPENE (Q1 2026, bekreftet):
  Aker BP     : Liquids $82.2/boe, Q4-25 var $63.1 (+$19.1)
  DNO Norge   : Oil $87.0/boe, Q4-25 var $63.6 (+$23.4)
  DNO Kurdistan: Oil $31.0/boe, Q4-25 var $31.6 (FLAT — produksjon stanset + lokalsalg)
  Vår Energi  : $77/boe blended; sier "North Sea premium differentials to be realised
                in second quarter" — Q1 fanget ikke fullt premien pga. prising-timing

HORMUZ-EFFEKTEN PÅ KVALITETS-DIFFERENSIALER (Reuters / IEA / Argus, mars-april 2026):
  WTI Midland til Asia      : peak +$40/fat over Dubai
  WTI til Europa            : peak +$22/fat over dated Brent
  Vest-afrikanske grader    : peak ~+$10/fat over dated Brent
  Nordsjø-grader (NCS)      : peak >+$10/fat over dated Brent
  Dubai (Medium Gulf Sour)  : $166/fat 19.mars (knapphet = kortvarig premium)
  Kurdistan (ikke-Hormuz)   : $31/fat realisert — fortsatt lokalsalg, krig-disrupsjon

INNSIKT:
  Under en Hormuz-stengsel overskygger GEOGRAFISK TILGJENGELIGHET kvalitet.
  Ikke-Hormuz olje (uansett kvalitet) er verdt mer fordi den KAN leveres.
  Sour discount/sweet premium = sekundær; Hormuz/ikke-Hormuz = primær.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRENT_CSV  = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
WTI_CSV    = PROJECT_ROOT / "data" / "raw" / "wti_monthly.csv"
MARS_CSV   = PROJECT_ROOT / "data" / "raw" / "mars_blend_monthly.csv"
DIFF_CSV   = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR    = PROJECT_ROOT / "data" / "processed"

# ─── Bekreftet selskapdata (fra trading-oppdateringer og kvartalrapporter) ───
COMPANY_REALIZED = {
    "Aker BP (Liquids)": [
        # (kvartal-label, dato, realisert_usd, brent_snitt_usd)
        ("Q2-24", "2024-05-15", 83.4, 85.0),
        ("Q3-24", "2024-08-15", 78.0, 80.4),
        ("Q4-24", "2024-11-15", 72.6, 74.5),
        ("Q1-25", "2025-02-15", 69.8, 76.2),
        ("Q2-25", "2025-05-15", 63.8, 72.5),
        ("Q3-25", "2025-08-15", 66.7, 70.0),
        ("Q4-25", "2024-11-15", 63.1, 62.5),
        ("Q1-26", "2026-02-15", 82.2, 80.7),   # bekreftet trading update 16.april
    ],
    "DNO North Sea (Oil)": [
        ("Q2-24", "2024-05-15", 83.6, 85.0),
        ("Q3-24", "2024-08-15", 77.6, 80.4),
        ("Q4-24", "2024-11-15", 67.4, 74.5),
        ("Q1-25", "2025-02-15", 77.9, 76.2),
        ("Q2-25", "2025-05-15", 68.2, 72.5),
        ("Q3-25", "2025-08-15", 66.4, 70.0),
        ("Q4-25", "2024-11-15", 63.6, 62.5),
        ("Q1-26", "2026-02-15", 87.0, 80.7),   # bekreftet trading update 20.april
    ],
    "DNO Kurdistan (Oil)": [
        ("Q2-24", "2024-05-15", 34.0, 85.0),
        ("Q3-24", "2024-08-15", 35.0, 80.4),
        ("Q4-24", "2024-11-15", 38.0, 74.5),
        ("Q1-25", "2025-02-15", 34.7, 76.2),
        ("Q2-25", "2025-05-15", 32.0, 72.5),
        ("Q3-25", "2025-08-15", 33.0, 70.0),
        # Q4-25: pipeline restart sept 2025, bedring
        ("Q4-25", "2024-11-15", 31.6, 62.5),
        # Q1-26: produksjon stanset 28.feb-9.april, lokalsalg
        ("Q1-26", "2026-02-15", 31.0, 80.7),
    ],
}

# Hormuz-effekten på differensialer — bekreftet fra Reuters / IEA / Argus
# Format: (crude_type, region, api, sulfur, pre_hormuz_diff, during_hormuz_diff, kilde)
HORMUZ_DIFFERENTIALS = [
    # (navn, API, S%, pre-krise diff mot Brent, krise-peak diff mot Brent, kilde)
    ("WTI Midland (Til EU)",        42, 0.3, +2.5,  +22.0, "Reuters, peak mars 2026"),
    ("Bonny Light (Nigeria)",       37, 0.1, +3.5,  +10.0, "Reuters, peak mars 2026"),
    ("Ekofisk (NCS)",               37, 0.3, +1.8,  +10.0, "Reuters, NCS estimat"),
    ("Johan Sverdrup (NCS sour)",   28, 0.8, -0.6,   +5.0, "Reuters, NCS sour estimat"),
    ("Mars Blend (US Gulf sour)",   29, 1.8, -5.0,   +3.0, "EIA/Argus estimat"),
    ("Arab Light (Saudi, Hormuz)",  33, 1.8, -2.0,  +19.5, "SOMO/Bloomberg, knapphet"),
    ("Dubai (UAE, Hormuz)",         31, 2.0, -3.0,  +50.0, "Dubai hit $166 19.mars"),
    ("Kurdistan Tawke (Iraq war)",  28, 3.5, -8.0,  -78.0, "DNO Q1-2026 rapport"),
]

# Periodisering
HORMUZ_CLOSE  = pd.Timestamp("2026-03-01")
WAR_START     = pd.Timestamp("2026-02-28")
PIPELINE_OPEN = pd.Timestamp("2025-09-27")


def build_brent_wti(brent_csv: Path, wti_csv: Path) -> pd.DataFrame:
    brent = pd.read_csv(brent_csv, parse_dates=["date"])
    brent_m = brent.set_index("date")["brent_usd"].resample("MS").mean().reset_index()
    brent_m.columns = ["date", "brent"]

    wti = pd.read_csv(wti_csv, parse_dates=["date"])
    wti["ym"] = wti["date"].dt.to_period("M")
    brent_m["ym"] = brent_m["date"].dt.to_period("M")
    df = brent_m.merge(wti[["ym", "wti_usd"]], on="ym", how="left")
    df["brent_wti_spread"] = df["brent"] - df["wti_usd"]
    return df.dropna(subset=["brent_wti_spread"])


def company_df(data: dict) -> pd.DataFrame:
    rows = []
    for company, records in data.items():
        for q_label, date_str, realized, brent in records:
            rows.append(dict(
                company=company, quarter=q_label,
                date=pd.Timestamp(date_str),
                realized=realized, brent=brent,
                vs_brent=realized - brent,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    print("Laster data …")
    spread_df = build_brent_wti(BRENT_CSV, WTI_CSV)
    brent_daily = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    brent_daily = brent_daily[brent_daily["date"] >= "2024-01-01"]

    cdfs = company_df(COMPANY_REALIZED)
    hormuz_df = pd.DataFrame(HORMUZ_DIFFERENTIALS,
                             columns=["crude","api","sulfur","pre_diff","peak_diff","kilde"])
    hormuz_df["diff_change"] = hormuz_df["peak_diff"] - hormuz_df["pre_diff"]

    # ─── Print nøkkeltall ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("HORMUZ-EFFEKTEN: DIFFERENSIALENDRINGER (PRE-KRISE vs PEAK MARS 2026)")
    print("=" * 90)
    print(f"{'Råolje':<30} {'API':>5} {'S%':>5} {'Pre':>8} {'Krise-peak':>12} {'Endring':>10} Kilde")
    print("-" * 90)
    for _, r in hormuz_df.sort_values("diff_change", ascending=False).iterrows():
        print(f"{r['crude']:<30} {r['api']:>5.0f} {r['sulfur']:>5.1f} "
              f"{r['pre_diff']:>+8.1f} {r['peak_diff']:>+12.1f} "
              f"{r['diff_change']:>+10.1f}  {r['kilde']}")

    print("\n" + "=" * 90)
    print("SELSKAPERS REALISERTE PRISER (Q1 2026 vs Q4 2025)")
    print("=" * 90)
    q1 = cdfs[cdfs["quarter"] == "Q1-26"]
    q4 = cdfs[cdfs["quarter"] == "Q4-25"]
    for company in COMPANY_REALIZED.keys():
        r_q1 = q1[q1["company"] == company]
        r_q4 = q4[q4["company"] == company]
        if not r_q1.empty and not r_q4.empty:
            p1 = r_q1.iloc[0]
            p4 = r_q4.iloc[0]
            print(f"\n  {company}")
            print(f"    Q4-25: ${p4.realized:.1f}/boe  (vs Brent {p4.brent:.1f} = {p4.vs_brent:+.1f})")
            print(f"    Q1-26: ${p1.realized:.1f}/boe  (vs Brent {p1.brent:.1f} = {p1.vs_brent:+.1f})")

    # ─── PLOTT ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])    # Brent + events (full bredde)
    ax2 = fig.add_subplot(gs[1, 0])    # Brent-WTI spread
    ax3 = fig.add_subplot(gs[1, 1])    # Differensial-endring per crude (lollipop)
    ax4 = fig.add_subplot(gs[2, 0])    # Selskapspriser
    ax5 = fig.add_subplot(gs[2, 1])    # API vs delta-diff scatter

    COL_WAR   = "#c0392b"
    COL_OPEN  = "#27ae60"
    COL_BLUE  = "#2980b9"

    # ── Plott 1: Brent daglig + event-merking ────────────────────────────
    ax1.plot(brent_daily["date"], brent_daily["brent_usd"],
             color=COL_BLUE, lw=1.5, alpha=0.9, label="Brent spot daglig")
    ax1.fill_between(brent_daily["date"], brent_daily["brent_usd"],
                     alpha=0.08, color=COL_BLUE)

    # Korleis-enn: grønt (pipeline åpen), rødt (war/Hormuz)
    ax1.axvspan(brent_daily["date"].min(), WAR_START,
                alpha=0.05, color=COL_OPEN)
    ax1.axvspan(WAR_START, brent_daily["date"].max(),
                alpha=0.07, color=COL_WAR)

    ax1.axvline(PIPELINE_OPEN, color=COL_OPEN, lw=1.8, ls="--", alpha=0.8)
    ax1.axvline(WAR_START,     color=COL_WAR,  lw=1.8, ls="--", alpha=0.8)

    # Event-annotasjonspiler
    events = [
        (PIPELINE_OPEN, 70, "Pipeline restart\n(27. sept 2025)"),
        (WAR_START,     90, "US/Israel-angrep\nIran (28. feb 2026)"),
        (pd.Timestamp("2026-03-12"), 102, "Brent > $100"),
        (pd.Timestamp("2026-03-19"), 114, "Dubai $166/fat\n(all-time high)"),
        (pd.Timestamp("2026-03-31"), 130, "Brent $118\nrekord-kvartal"),
    ]
    for date, y_base, label in events:
        brent_val_row = brent_daily[brent_daily["date"] <= date]
        brent_val = brent_val_row["brent_usd"].iloc[-1] if not brent_val_row.empty else y_base
        ax1.annotate(
            label,
            xy=(date, brent_val),
            xytext=(date + pd.Timedelta(days=10), y_base),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="gray"),
        )

    ax1.axhline(100, color="red", lw=0.8, ls=":", alpha=0.5)
    ax1.set_ylabel("USD/fat")
    ax1.set_title("Brent spot 2024–2026: pipeline-restart, Iran-krig og Hormuz-stengsel",
                  fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.set_ylim(50, 145)
    ax1.legend(loc="upper left", fontsize=9)

    # ── Plott 2: Brent-WTI spread ────────────────────────────────────────
    sp = spread_df[spread_df["date"] >= "2024-01-01"]
    bar_colors = [COL_WAR if d >= WAR_START else COL_BLUE for d in sp["date"]]
    ax2.bar(sp["date"], sp["brent_wti_spread"],
            width=20, color=bar_colors, alpha=0.75, edgecolor="none")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(4, color="gray",  lw=0.8, ls="--", alpha=0.5)
    ax2.text(sp["date"].iloc[0], 4.3, "Historisk snitt ~$4", fontsize=8, color="gray")
    ax2.axvline(WAR_START, color=COL_WAR, lw=1.5, ls="--", alpha=0.7)
    ax2.text(WAR_START + pd.Timedelta(days=8), 20,
             "Krig", fontsize=8, color=COL_WAR, fontweight="bold")
    ax2.set_ylabel("Brent – WTI (USD/fat)")
    ax2.set_title("Brent-WTI spread: Hormuz-eksponering\n"
                  "Brent steg mer pga. geografisk nærhet til Hormuz-disrupsjon")
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    # Merk peak
    peak_row = sp.loc[sp["brent_wti_spread"].idxmax()]
    ax2.annotate(f"Peak ${peak_row['brent_wti_spread']:.1f}\n({peak_row['date'].strftime('%b %Y')})",
                 (peak_row["date"], peak_row["brent_wti_spread"]),
                 xytext=(-30, -20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    ax2.set_ylim(-5, 30)

    # ── Plott 3: Lollipop — pre vs krise differensial per crude ──────────
    h = hormuz_df.sort_values("pre_diff")
    y_pos = np.arange(len(h))
    ax3.hlines(y_pos, h["pre_diff"], h["peak_diff"].clip(-20, 25),
               color="gray", lw=2, alpha=0.5)
    ax3.scatter(h["pre_diff"], y_pos, s=80, color=COL_BLUE,
                zorder=5, label="Pre-krise")
    # Klipp Dubai-peak for plot (egentlig $50, men vi viser -20 til 25 range)
    peak_clipped = h["peak_diff"].clip(-20, 25)
    ax3.scatter(peak_clipped, y_pos, s=80, color=COL_WAR,
                zorder=5, label="Krise-peak mars 2026")

    # Merk Kurdistan spesielt
    krd_idx = h.index[h["crude"].str.contains("Kurdistan")].tolist()
    if krd_idx:
        ki = list(h.index).index(krd_idx[0])
        ax3.annotate("Kurdistan:\n$31/fat realisert\n(krig + lokalsalg)",
                     (h.loc[krd_idx[0], "peak_diff"].clip(-20, 25), ki),
                     xytext=(-15, -15), textcoords="offset points",
                     fontsize=7.5, color=COL_WAR,
                     arrowprops=dict(arrowstyle="->", color=COL_WAR, lw=0.8))
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(h["crude"], fontsize=8)
    ax3.set_xlabel("Differensial mot Dated Brent (USD/fat)")
    ax3.set_title("Hormuz-effekten per råoljegrad\n"
                  "● blå = pre-krise  ● rød = krise-peak mars 2026\n"
                  "(Note: Dubai-krise-peak egentlig +$50, Dubai $166/fat 19.mars)")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(True, alpha=0.25, axis="x")

    # ── Plott 4: Selskapspriser per kvartal ──────────────────────────────
    companies = list(COMPANY_REALIZED.keys())
    colors_c = {
        "Aker BP (Liquids)":      "#2980b9",
        "DNO North Sea (Oil)":    "#27ae60",
        "DNO Kurdistan (Oil)":    "#c0392b",
    }
    markers = {"Aker BP (Liquids)": "o", "DNO North Sea (Oil)": "s",
               "DNO Kurdistan (Oil)": "^"}

    for company in companies:
        sub = cdfs[cdfs["company"] == company].sort_values("date")
        color = colors_c.get(company, "gray")
        marker = markers.get(company, "o")
        ax4.plot(sub["date"], sub["realized"],
                 marker=marker, color=color, lw=1.8, ms=7,
                 label=company)

    # Brent som referanse
    brent_q = cdfs.drop_duplicates("quarter").sort_values("date")
    ax4.plot(brent_q["date"], brent_q["brent"],
             color="black", lw=1.2, ls="--", alpha=0.5, label="Brent snitt")

    ax4.axvline(WAR_START, color=COL_WAR, lw=1.5, ls="--", alpha=0.6)
    ax4.axvspan(WAR_START, cdfs["date"].max() + pd.Timedelta(days=30),
                alpha=0.06, color=COL_WAR)

    # Merk Q1-26 for NCS-oppsiden
    ax4.annotate("Q1-26: NCS\n+$20-23 QoQ",
                 (pd.Timestamp("2026-02-15"), 87.0),
                 xytext=(-60, 15), textcoords="offset points",
                 fontsize=8, color=COL_OPEN,
                 arrowprops=dict(arrowstyle="->", color=COL_OPEN, lw=0.8))
    # Merk Kurdistan flat
    ax4.annotate("Kurdistan flat\n(krig-disrupsjon)",
                 (pd.Timestamp("2026-02-15"), 31.0),
                 xytext=(-80, -25), textcoords="offset points",
                 fontsize=8, color=COL_WAR,
                 arrowprops=dict(arrowstyle="->", color=COL_WAR, lw=0.8))

    ax4.set_ylabel("Realisert pris (USD/boe)")
    ax4.set_title("Selskapers realiserte priser per kvartal\nKilde: Offisielle trading-oppdateringer")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 5, 8, 11]))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax4.set_ylim(20, 130)

    # ── Plott 5: Scatter API vs differensialendring (delta) ───────────────
    colors_hormuz = []
    for _, r in hormuz_df.iterrows():
        if "Hormuz" in r["kilde"] or "knapphet" in r["kilde"] or "Dubai" in r["kilde"]:
            colors_hormuz.append("#e74c3c")
        elif "Kurdistan" in r["crude"]:
            colors_hormuz.append("#c0392b")
        else:
            colors_hormuz.append("#2980b9")

    sc = ax5.scatter(
        hormuz_df["api"], hormuz_df["diff_change"],
        s=120, c=colors_hormuz, alpha=0.80,
        edgecolor="black", lw=0.6, zorder=5,
    )
    # Størrelse = sulfur
    sizes = 60 + hormuz_df["sulfur"] * 60
    ax5.scatter(hormuz_df["api"], hormuz_df["diff_change"],
                s=sizes, c=colors_hormuz, alpha=0.35, edgecolor="none")

    for _, r in hormuz_df.iterrows():
        label = r["crude"].replace(" (Til EU)", "").replace(" (Til Asia)", "")
        ax5.annotate(label,
                     (r["api"], min(r["diff_change"], 20)),
                     xytext=(5, 3), textcoords="offset points",
                     fontsize=7.5, color="#2c3e50")

    ax5.axhline(0, color="black", lw=0.8)
    ax5.axvline(35, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax5.text(35.2, -12, "Light/heavy grense", fontsize=8, color="gray")
    ax5.set_xlabel("API-grad (høyere = lettere)")
    ax5.set_ylabel("Differensial-endring: krise-peak minus pre-krise (USD/fat)")
    ax5.set_title("Hormuz-effekt: differensialendring vs API-grad\n"
                  "(størrelse = svovelinnhold; blå = ikke-Hormuz; rød = Hormuz-eksponert)")
    ax5.grid(True, alpha=0.25)
    ax5.set_ylim(-90, 25)

    # Manuel legend
    patch_blue = mpatches.Patch(color="#2980b9", label="Ikke-Hormuz (NCS, US, WA)")
    patch_red  = mpatches.Patch(color="#e74c3c", label="Hormuz-eksponert (Gulf/Iraq)")
    ax5.legend(handles=[patch_blue, patch_red], fontsize=8, loc="lower left")

    fig.suptitle(
        "Hormuz-effekten på råolje-differensialer og selskapers realiseringspriser — Q1 2026\n"
        "Geografisk tilgjengelighet trumfer kvalitet under en forsyningskrise",
        fontsize=12, fontweight="bold", y=1.01,
    )

    out_png = OUT_DIR / "11_hormuz_differential_analysis.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nPlott lagret: {out_png}")

    # Lagre tabell
    out_csv = OUT_DIR / "11_hormuz_differentials.csv"
    hormuz_df.to_csv(out_csv, index=False)
    print(f"Differensial-tabell lagret: {out_csv}")

    # Oppsummering
    print("\n" + "=" * 90)
    print("OPPSUMMERING — IMPLIKASJONER FOR Q2 2026")
    print("=" * 90)
    print("""
KEY INSIGHT: Hormuz-krise skapte en GEOGRAFISK PREMIUM som overskygget kvalitetspremier.

VINNERE (Q1-Q2 2026):
  • NCS-operatører (Aker BP, Equinor, Vår Energi):
    - Nordsjø-olje er ikke-Hormuz og fikk +$10 til +$22/fat ekstra vs dated Brent
    - Vår Energi sier "North Sea premium differentials to be realised in Q2" →
      Q1 hadde pricing-lag; fulle premier regnskapsføres i Q2-rapporten
    - Aker BP: $82.2 i Q1 (Brent-snitt $80.7) → reelt sett litt over Brent

TAPER (Q1 2026):
  • DNO Kurdistan:
    - Produksjon STANSET 28. feb til 9. april pga. Iran-krig (6 uker)
    - Salg til lokale kjøpere, ikke via Ceyhan — samme situasjon som 2023-2025!
    - Realisert: $31/fat mens Brent var $80+ (sannsynligvis priset i jan/feb)
    - Ironisk: nøyaktig da NCS-olje fikk rekord-premier, mistet DNO Kurdistan-produksjonen

Q2 2026 UTSIKT (kvartal pågår nå, rapport 7. mai):
  • Aker BP (full rapport 7.mai): forventes sterk — Q1 trading var god start
  • Vår Energi: bekrefter Q2 dividend $300m, Q2-premier > Q1 ifølge dem
  • DNO Kurdistan: Produksjon gjenopptatt 9. april. Eks via Ceyhan igjen? →
    Hvis ja: STOR positiv overraskelse vs. $31/fat i Q1
    Aksjekurs-katalysator: fra $31 → potensielt $100+ per fat i Q2
    Med 39,600 boepd net og $69 forbedring → ~$100m ekstra inntekt PER KVARTAL
""")


if __name__ == "__main__":
    main()
