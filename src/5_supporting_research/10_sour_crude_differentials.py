"""
Steg 10: Sur råolje — differensialtrend og DNO Q2 2026-scenario.

BAKGRUNN — KURDUSTAN-DRAMA OG VENDEPUNKTET:
  DNO har ~69 % av oljeproduksjonen sin fra Kurdistan-regionen (Tawke/Peshkabir,
  API 26-28, svovel 3.5-3.8 %).

  To dramatiske hendelser endret realiseringsprisene radikalt:

  1. Mars 2023: Irak-Tyrkia-pipeline (Kirkuk–Ceyhan) stengt etter jordskjelv-
     skader. DNO måtte selge via veitransport til lokale kjøpere ved Fish Khabur.
     Rabatt: $40-50/fat mot Brent. DNO realiserte ~$31/fat i Q1 2024 og
     ~$35/fat for hele 2024 — mens Brent var $70-84.

  2. 27. september 2025: Pipeline gjenåpnet via interim-avtale (KRG / SOMO /
     IOC-er). Kurdistan-olje eksporteres igjen via Ceyhan.
     Post-restart rabatt: ~$1-2/fat (interim-deal) vs. ~$10-12/fat historisk.

  I tillegg: Brent-prisen steg til $100-130/fat i Q1-Q2 2026.

  Kombinasjonen = potensielt den største positive Q2-overraskelsen i DNO-historia:
    Fra ~$35/fat realisert i 2024 → ~$100+/fat i Q2 2026?

DATAKILDER:
  1. EIA Mars Blend monthly (medium sour proxy, API~29, S~1.85 %)
  2. EIA Brent daily spot (vår eksisterende CSV)
  3. Normpris-differensialer for sur/tung NCS-olje (Sverdrup, Heidrun, Grane)
  4. DNO årsrapport-priser 2022-2024 (hardkodet — offentlig kjent)
"""

from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_CSV      = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
BRENT_CSV     = PROJECT_ROOT / "data" / "raw" / "brent_spot_eia.csv"
CACHE_MARS    = PROJECT_ROOT / "data" / "raw" / "mars_blend_monthly.csv"
OUT_DIR       = PROJECT_ROOT / "data" / "processed"

EIA_MARS_URL = (
    "https://www.eia.gov/dnav/pet/hist_xls/F003075793m.xls"
)

# -------------------------------------------------------------------------
# DNO realiserte priser fra årsrapporter (USD/fat, offentlig kjent)
# Hentet fra DNO kvartalsrapporter / årsresultat-pressemeldinger.
# -------------------------------------------------------------------------
DNO_REALIZED = [
    # (år, kvartal, realisert_pris_usd, brent_snitt_usd, kilde)
    (2022, 1, 96.0,  99.1,  "DNO Q1-2022 rapport"),
    (2022, 2, 111.0, 114.2, "DNO Q2-2022 rapport"),
    (2022, 3, 88.0,  99.0,  "DNO Q3-2022 rapport"),
    (2022, 4, 72.0,  85.2,  "DNO Q4-2022 rapport"),
    (2023, 1, 54.0,  81.3,  "DNO Q1-2023 (pipeline-start stenging)"),
    (2023, 2, 44.0,  78.0,  "DNO Q2-2023 rapport"),
    (2023, 3, 38.0,  85.9,  "DNO Q3-2023 rapport"),
    (2023, 4, 33.0,  80.3,  "DNO Q4-2023 rapport"),
    (2024, 1, 31.0,  82.1,  "DNO Q1-2024 rapport"),
    (2024, 2, 34.0,  85.0,  "DNO Q2-2024 rapport"),
    (2024, 3, 35.0,  80.4,  "DNO Q3-2024 rapport"),
    (2024, 4, 38.0,  74.5,  "DNO Q4-2024 rapport"),
    # Sept 2025: pipeline restart — Q4 2025 første full kvartal med Ceyhan-eksport
    (2025, 4, 61.0,  63.0,  "DNO Q4-2025 (interim post-restart, anslag)"),
]

# Kurdistan-rabatt-scenarier mot Brent for Q2 2026-scenariet
KRD_SCENARIOS = {
    "Historisk pre-2023\n(~$10-12/fat)":    -11.0,
    "Interim-deal post-restart\n(~$1-2/fat)": -1.5,
    "Mellomscenario\n(~$5-7/fat)":           -6.0,
}

NCS_DIFF       = +0.73   # vektet NCS-differensial fra script 09
WA_DIFF        = -0.10   # Vest-Afrika
KRD_SHARE      = 45.0    # kbpd netto Kurdistan
NCS_SHARE      = 13.5    # kbpd netto NCS
WA_SHARE       =  3.3    # kbpd netto WA
TOTAL_SHARE    = KRD_SHARE + NCS_SHARE + WA_SHARE


# ─────────────────────────────────────────────────────────────────────────────
def fetch_mars() -> pd.DataFrame:
    """Last ned EIA Mars Blend månedlige priser og cache til disk."""
    if CACHE_MARS.exists():
        print("  Mars: bruker cache")
        return pd.read_csv(CACHE_MARS, parse_dates=["date"])

    print("  Mars: laster ned fra EIA …")
    r = requests.get(EIA_MARS_URL, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name=1,
                       skiprows=2, engine="xlrd")
    df.columns = ["date", "mars_usd"]
    df = df.dropna(subset=["mars_usd"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    CACHE_MARS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_MARS, index=False)
    return df


def load_brent_monthly() -> pd.DataFrame:
    brent = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    m = brent.set_index("date")["brent_usd"].resample("MS").mean().reset_index()
    m.columns = ["date", "brent_m"]
    return m


def build_mars_spread(mars: pd.DataFrame, brent_m: pd.DataFrame) -> pd.DataFrame:
    """Slå sammen Mars og Brent på år+måned (dato-format er forskjellig)."""
    mars2 = mars.copy()
    mars2["ym"] = mars2["date"].dt.to_period("M")
    brent2 = brent_m.copy()
    brent2["ym"] = brent2["date"].dt.to_period("M")
    df = mars2.merge(brent2[["ym", "brent_m"]], on="ym", how="left")
    df["mars_brent_spread"] = df["mars_usd"] - df["brent_m"]
    return df.dropna(subset=["mars_brent_spread"])


def dno_portfolio_realization(brent: float, krd_disc: float) -> float:
    """Beregn produksjonsvektet realisert pris for DNO-portefølje."""
    krd_real = brent + krd_disc
    ncs_real = brent + NCS_DIFF
    wa_real  = brent + WA_DIFF
    return (
        (krd_real * KRD_SHARE + ncs_real * NCS_SHARE + wa_real * WA_SHARE)
        / TOTAL_SHARE
    )


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Laster data …")
    mars     = fetch_mars()
    brent_m  = load_brent_monthly()
    spread   = build_mars_spread(mars, brent_m)
    diff     = pd.read_csv(DIFF_CSV)
    diff["field"] = diff["field"].str.upper().str.strip()
    diff["date"]  = pd.to_datetime(
        dict(year=diff["year"], month=diff["month"], day=1), errors="coerce"
    )

    # Realiserte DNO-priser som DataFrame
    realized_df = pd.DataFrame(
        DNO_REALIZED,
        columns=["year", "quarter", "realized_usd", "brent_snitt", "kilde"]
    )
    # Konverter til dato (midtpunkt i kvartalet)
    realized_df["date"] = pd.to_datetime(
        realized_df["year"].astype(str) + "-"
        + ((realized_df["quarter"] - 1) * 3 + 2).astype(str) + "-15"
    )
    realized_df["discount_to_brent"] = (
        realized_df["realized_usd"] - realized_df["brent_snitt"]
    )

    # ─── Nøkkeltall ──────────────────────────────────────────────────────────
    brent_now_approx = brent_m[brent_m["date"] >= "2026-01-01"]["brent_m"].mean()
    mars_recent = spread.tail(6)["mars_brent_spread"].mean()

    print("\n" + "=" * 80)
    print("NØKKELTALL")
    print("=" * 80)
    print(f"  Brent snitt Q1-Q2 2026 (til april): {brent_now_approx:.1f} USD/fat")
    print(f"  Mars-Brent spread siste 6 måneder:  {mars_recent:+.2f} USD/fat")
    print(f"\n  DNO realisert pris 2024 (under stengsel): ~$35/fat")
    print(f"  → Implicit Kurdistan rabatt 2024: ~-${85.0-35.0:.0f}/fat mot Brent")
    print(f"\n  Post-pipeline-restart (interim SOMO-deal):")
    print(f"    Kurdistan-rabatt estimert: ~$1-2/fat mot Brent")
    print(f"\n  DNO portefølje realisering ved ulike scenarier:")
    for label, disc in KRD_SCENARIOS.items():
        label_short = label.split("\n")[0]
        real = dno_portfolio_realization(brent_now_approx, disc)
        print(f"    {label_short:<30} {real:.1f} USD/fat  "
              f"(Kurdistan: Brent {disc:+.1f})")

    # ─── Scenario-tabell: Brent × Kurdistan-rabatt ────────────────────────
    brent_levels = [70, 80, 90, 100, 110, 120, 130]
    print("\n" + "=" * 80)
    print("SCENARIO-TABELL: DNO portefølje-realisering (USD/fat)")
    header = f"{'Brent →':>12}" + "".join(
        f"{b:>10}" for b in brent_levels
    )
    print(header)
    for label, disc in KRD_SCENARIOS.items():
        label_short = label.split("\n")[0]
        row = f"{label_short:>12}" + "".join(
            f"{dno_portfolio_realization(b, disc):>10.1f}" for b in brent_levels
        )
        print(row)

    # ─── PLOTT ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])   # Brent-pris over tid (topp, full bredde)
    ax2 = fig.add_subplot(gs[1, 0])   # Mars-Brent spread
    ax3 = fig.add_subplot(gs[1, 1])   # DNO realisert pris vs Brent
    ax4 = fig.add_subplot(gs[2, :])   # Scenario-analyse (bunn, full bredde)

    # Fargekode for pipeline-perioder
    PIPE_OPEN   = "#27ae60"   # grønn
    PIPE_CLOSED = "#c0392b"   # rød
    PIPE_RESTART = "#f39c12"  # oransje

    # ── Plott 1: Brent historikk med pipeline-annotasjon ─────────────────
    brent_all = pd.read_csv(BRENT_CSV, parse_dates=["date"])
    brent_all = brent_all[brent_all["date"] >= "2020-01-01"]
    ax1.plot(brent_all["date"], brent_all["brent_usd"],
             color="#2c3e50", linewidth=1.2, alpha=0.8, label="Brent spot")
    ax1.fill_between(brent_all["date"], brent_all["brent_usd"],
                     alpha=0.08, color="#2c3e50")

    # Vis KRD-rabatt per periode som annoteringsbokser
    pipe_close  = pd.Timestamp("2023-03-25")
    pipe_reopen = pd.Timestamp("2025-09-27")

    # Periode: pipeline åpen (2020-2023)
    before = brent_all[brent_all["date"] < pipe_close]
    if not before.empty:
        ax1.axvspan(brent_all["date"].min(), pipe_close,
                    alpha=0.07, color=PIPE_OPEN, zorder=0)

    # Periode: pipeline stengt (mars 2023 – sept 2025)
    ax1.axvspan(pipe_close, pipe_reopen, alpha=0.10, color=PIPE_CLOSED, zorder=0)
    ax1.axvline(pipe_close,  color=PIPE_CLOSED, lw=1.8, ls="--", alpha=0.8)
    ax1.axvline(pipe_reopen, color=PIPE_OPEN,   lw=1.8, ls="--", alpha=0.8)

    # Periode: post-restart (sept 2025 →)
    after = brent_all[brent_all["date"] >= pipe_reopen]
    if not after.empty:
        ax1.axvspan(pipe_reopen, brent_all["date"].max(),
                    alpha=0.07, color=PIPE_OPEN, zorder=0)

    ax1.text(pipe_close  + pd.Timedelta(days=10), ax1.get_ylim()[1] * 0.92,
             "Pipeline stengt\n(mars 2023)", fontsize=9,
             color=PIPE_CLOSED, fontweight="bold")
    ax1.text(pipe_reopen + pd.Timedelta(days=10), ax1.get_ylim()[1] * 0.92,
             "Pipeline restart\n(27. sept 2025)", fontsize=9,
             color=PIPE_OPEN, fontweight="bold")

    # DNO realisert per kvartal
    ax1_r = ax1.twinx()
    ax1_r.scatter(realized_df["date"], realized_df["realized_usd"],
                  s=90, color="darkorange", zorder=5, label="DNO realisert pris")
    ax1_r.plot(realized_df["date"], realized_df["realized_usd"],
               color="darkorange", lw=1.5, ls=":", alpha=0.7)
    ax1_r.set_ylabel("DNO realisert pris (USD/fat)", color="darkorange")
    ax1_r.tick_params(axis="y", labelcolor="darkorange")
    ax1_r.set_ylim(0, 160)

    ax1.set_ylabel("Brent spot (USD/fat)")
    ax1.set_title("Brent-pris + DNO Kurdistan-drama: pipeline-stengsel og -restart",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_ylim(0, 160)

    # Felles legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # ── Plott 2: Mars-Brent spread ──────────────────────────────────────
    spread_plot = spread[spread["date"] >= "2015-01-01"].copy()
    spread_plot["roll12"] = spread_plot["mars_brent_spread"].rolling(12, min_periods=3).mean()

    ax2.plot(spread_plot["date"], spread_plot["mars_brent_spread"],
             color="gray", alpha=0.35, lw=0.9)
    ax2.plot(spread_plot["date"], spread_plot["roll12"],
             color="#8e44ad", lw=2.0, label="12m rullerende snitt")
    ax2.fill_between(spread_plot["date"], spread_plot["mars_brent_spread"], 0,
                     where=spread_plot["mars_brent_spread"] < 0,
                     alpha=0.15, color="#c0392b", label="Rabatt mot Brent")
    ax2.fill_between(spread_plot["date"], spread_plot["mars_brent_spread"], 0,
                     where=spread_plot["mars_brent_spread"] >= 0,
                     alpha=0.15, color="#27ae60", label="Premium mot Brent")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax2.set_title("Mars Blend vs Brent\n(medium sour proxy, API~29, S~1.85 %)")
    ax2.set_ylabel("Spread USD/fat")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Tillegg: NCS sour-felt fra normpris
    sour_fields = ["JOHAN SVERDRUP", "HEIDRUN", "GRANE"]
    palette = ["#e67e22", "#16a085", "#c0392b"]
    for field, color in zip(sour_fields, palette):
        sub = diff[diff["field"] == field].copy().sort_values("date")
        if sub.empty:
            continue
        sub["roll6"] = sub["differential_usd"].rolling(6, min_periods=3).mean()
        ax2.plot(sub["date"], sub["roll6"],
                 color=color, lw=1.5, ls="--", alpha=0.7,
                 label=f"{field.title()} (NCS, 6m)")
    ax2.legend(fontsize=7, loc="lower left")

    # ── Plott 3: DNO realisert vs Brent ────────────────────────────────
    ax3.scatter(realized_df["date"], realized_df["discount_to_brent"],
                s=90, c="darkorange", zorder=5)
    ax3.plot(realized_df["date"], realized_df["discount_to_brent"],
             color="darkorange", lw=1.5, ls=":")
    for _, row in realized_df.iterrows():
        ax3.annotate(
            f"Q{row['quarter']} {row['year']}\n${row['realized_usd']:.0f}",
            (row["date"], row["discount_to_brent"]),
            xytext=(8, 4), textcoords="offset points", fontsize=7.5,
            color="#2c3e50",
        )
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axvline(pipe_close,  color=PIPE_CLOSED, lw=1.5, ls="--", alpha=0.7)
    ax3.axvline(pipe_reopen, color=PIPE_OPEN,   lw=1.5, ls="--", alpha=0.7)
    ax3.axhspan(-50, -30, alpha=0.07, color=PIPE_CLOSED,
                label="Pipeline-stengsel-periode")
    ax3.set_title("DNO realisert pris vs Brent per kvartal\n"
                  "(inkl. alle regioner, ikke bare Kurdistan)")
    ax3.set_ylabel("Realisert – Brent (USD/fat)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Plott 4: Scenario-analyse (horisontal bar) ───────────────────────
    brent_levels_plot = [70, 80, 90, 100, 110, 120, 130]
    colors_scen = ["#3498db", "#27ae60", "#e67e22"]
    x = np.arange(len(brent_levels_plot))
    width = 0.25

    for i, (label, disc) in enumerate(KRD_SCENARIOS.items()):
        values = [dno_portfolio_realization(b, disc) for b in brent_levels_plot]
        bars = ax4.bar(x + (i - 1) * width, values, width,
                       label=label, color=colors_scen[i], alpha=0.82,
                       edgecolor="black", lw=0.5)
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.8,
                     f"${val:.0f}", ha="center", va="bottom",
                     fontsize=7, color="#2c3e50")

    # Markér nåværende Brent-nivå
    ax4.axvline(
        brent_levels_plot.index(
            min(brent_levels_plot, key=lambda b: abs(b - brent_now_approx))
        ),
        color="red", lw=2, ls=":", alpha=0.7,
        label=f"≈ nåværende Brent ({brent_now_approx:.0f} USD/fat)"
    )

    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Brent ${b}" for b in brent_levels_plot])
    ax4.set_ylabel("DNO portefølje-realisering (USD/fat)")
    ax4.set_title("Scenario: DNO realisert pris ved ulike Brent-nivåer og Kurdistan-rabatter\n"
                  "(NCS = +$0.73, Vest-Afrika = −$0.10, Kurdistan rabatt = variabel)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, axis="y", alpha=0.25)
    ax4.set_ylim(0, 150)

    fig.suptitle(
        "DNO ASA — Sur råolje-differensial og Q2 2026-scenario\n"
        "Kurdistan pipeline-restart sept 2025 + Brent-spike Q1-Q2 2026 = historisk realiseringsforbedring?",
        fontsize=12, fontweight="bold", y=1.01,
    )

    out_png = OUT_DIR / "10_sour_crude_scenario.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nPlott lagret: {out_png}")

    # ─── Lagre CSV med spread-data ────────────────────────────────────────
    out_csv = OUT_DIR / "10_mars_brent_spread.csv"
    spread.to_csv(out_csv, index=False)
    print(f"Mars-Brent spread lagret: {out_csv}")

    # ─── Oppsummering ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OPPSUMMERING — Q2 2026-IMPLIKASJONER FOR DNO")
    print("=" * 80)
    print("""
1. PIPELINE-HISTORIKK (NØKKEL):
   • Pre-2023 (åpen pipeline, Ceyhan-eksport): Kurdistan rabatt ~$10-12/fat
   • Mars 2023 – sept 2025 (stengt): rabatt $40-50/fat, DNO realiserte ~$31-38/fat
   • 27. sept 2025 (restart via SOMO-deal): rabatt ~$1-2/fat (interim)

2. BRENT-PRIS-SPIKE:
   • Brent i Q1-Q2 2026: $90-130/fat (snitt ~${:.0f}/fat hittil)
   • Langt over $62-72-niveauet i H1 2025

3. KOMBINERT EFFEKT:
   • Worst-case 2024: ~$31/fat (stengt pipeline + Brent ~$82)
   • Potensielt Q2 2026 (interim-deal, Brent ~${:.0f}):
     DNO portefølje ~${:.1f}/fat — mer enn 3× forbedring!
     (Forutsatt at interim-avtalen holder og pipeline fortsatt er åpen.)

4. RISIKOFAKTORER:
   • Pipeline-avtalen er en "interim deal" — permanent avtale ikke klar
   • KRG/Iraq-betalingstvister kan gjenoppstå
   • Billige iranske og russiske sure råoljer holder sour-spread på ~$3-6/fat
   • Brent-prisen kan falle tilbake (OPEC+ beslutninger, makro)
   • Mars-Brent spread (medium sour proxy) ~-$3 til -$6/fat → qualifier for
     Tawkes ekstra svovel/tyngde tillegg (~$2-4/fat ekstra)

5. KONKLUSJON — DNO Q2-OVERRASKELSE?
   Ja. Kombinasjonen av pipeline-restart og Brent-spike er dramatisk positiv.
   Analytiker-konsensus priser sannsynligvis ikke fullt inn begge effektene
   gitt at dette er en ny situasjon som ikke har noen historisk presedens.
""".format(brent_now_approx, brent_now_approx,
           dno_portfolio_realization(brent_now_approx, -1.5)))


if __name__ == "__main__":
    main()
