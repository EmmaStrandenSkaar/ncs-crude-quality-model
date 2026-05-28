"""
Analyse: Tidsforsinkelse (lag) mellom oljepris og oljeserviceaksjer på Oslo Børs.

Metode:
  1. Henter ukentlige avkastninger for Brent og norske oljeserviceaksjer
  2. Beregner krysskorrelasjoner ved ulike lag (0-26 uker)
  3. Finner optimal lag per aksje og for en likevektet indeks
  4. Visualiserer resultatene
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- Konfigurasjon ---
BRENT = "BZ=F"
OILSERVICE_TICKERS = {
    "OTL.OL": "Odfjell Technology",
    "AKRBP.OL": "Aker BP",
    "SUBC.OL": "Subsea 7",
    "TGS.OL": "TGS",
    "AKSO.OL": "Aker Solutions",
    "BWO.OL": "BW Offshore",
    "PGS.OL": "PGS",
}

# Lange tidsserier for robust estimat
START = "2018-01-01"
END = "2026-05-11"
MAX_LAG_WEEKS = 26  # test lag opp til 6 mnd

# --- Hent data ---
print("Henter Brent-data...")
brent = yf.Ticker(BRENT).history(start=START, end=END)
brent.index = brent.index.tz_localize(None)
brent_weekly = brent["Close"].resample("W-FRI").last().dropna()
brent_ret = brent_weekly.pct_change().dropna()

stock_returns = {}
for ticker, name in OILSERVICE_TICKERS.items():
    print(f"Henter {name} ({ticker})...")
    try:
        hist = yf.Ticker(ticker).history(start=START, end=END)
        hist.index = hist.index.tz_localize(None)
        weekly = hist["Close"].resample("W-FRI").last().dropna()
        ret = weekly.pct_change().dropna()
        if len(ret) > 100:
            stock_returns[name] = ret
            print(f"  -> {len(ret)} ukentlige observasjoner")
        else:
            print(f"  -> For lite data ({len(ret)} obs), hopper over")
    except Exception as e:
        print(f"  -> Feil: {e}")

# --- Lag-korrelasjon per aksje ---
print("\n--- Krysskorrelasjonsanalyse ---")

results = {}
for name, stock_ret in stock_returns.items():
    # Align on common dates
    common = pd.concat([brent_ret, stock_ret], axis=1, join="inner")
    common.columns = ["brent", "stock"]
    common = common.dropna()

    if len(common) < 52:
        print(f"  {name}: for lite overlappende data")
        continue

    correlations = []
    for lag in range(0, MAX_LAG_WEEKS + 1):
        if lag == 0:
            corr = common["brent"].corr(common["stock"])
        else:
            # Lag = stock reagerer LAG uker ETTER oljeprisendring
            shifted_brent = common["brent"].shift(lag)
            valid = pd.concat([shifted_brent, common["stock"]], axis=1).dropna()
            if len(valid) > 30:
                corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            else:
                corr = np.nan
        correlations.append(corr)

    results[name] = correlations
    best_lag = np.nanargmax(correlations)
    best_corr = correlations[best_lag]
    lag0_corr = correlations[0]
    print(f"  {name:20s}: Lag 0 = {lag0_corr:.3f}, Best lag = {best_lag} uker (r={best_corr:.3f})")

# --- Likevektet oljeserviceindeks ---
print("\n--- Likevektet oljeserviceindeks (eks. Aker BP) ---")
service_only = {k: v for k, v in stock_returns.items() if k != "Aker BP"}
if service_only:
    combined = pd.concat(service_only.values(), axis=1, join="inner")
    combined.columns = list(service_only.keys())
    index_ret = combined.mean(axis=1)

    common_idx = pd.concat([brent_ret, index_ret], axis=1, join="inner").dropna()
    common_idx.columns = ["brent", "index"]

    index_correlations = []
    for lag in range(0, MAX_LAG_WEEKS + 1):
        if lag == 0:
            corr = common_idx["brent"].corr(common_idx["index"])
        else:
            shifted = common_idx["brent"].shift(lag)
            valid = pd.concat([shifted, common_idx["index"]], axis=1).dropna()
            corr = valid.iloc[:, 0].corr(valid.iloc[:, 1]) if len(valid) > 30 else np.nan
        index_correlations.append(corr)

    best_idx_lag = np.nanargmax(index_correlations)
    print(f"  Indeks: Lag 0 = {index_correlations[0]:.3f}, Best lag = {best_idx_lag} uker (r={index_correlations[best_idx_lag]:.3f})")
else:
    index_correlations = None

# --- Også sjekk med månedlige data ---
print("\n--- Månedlig krysskorrelasjonsanalyse ---")
brent_monthly = brent["Close"].resample("ME").last().dropna()
brent_mret = brent_monthly.pct_change().dropna()

monthly_results = {}
for name, stock_ret_w in stock_returns.items():
    # Resample to monthly from weekly
    stock_m = stock_ret_w.resample("ME").apply(lambda x: (1+x).prod()-1).dropna()
    common_m = pd.concat([brent_mret, stock_m], axis=1, join="inner").dropna()
    common_m.columns = ["brent", "stock"]

    if len(common_m) < 24:
        continue

    m_corrs = []
    for lag in range(0, 7):  # 0-6 måneder lag
        if lag == 0:
            corr = common_m["brent"].corr(common_m["stock"])
        else:
            shifted = common_m["brent"].shift(lag)
            valid = pd.concat([shifted, common_m["stock"]], axis=1).dropna()
            corr = valid.iloc[:, 0].corr(valid.iloc[:, 1]) if len(valid) > 12 else np.nan
        m_corrs.append(corr)

    monthly_results[name] = m_corrs
    best_m = np.nanargmax(m_corrs)
    print(f"  {name:20s}: Lag 0 = {m_corrs[0]:.3f}, Best lag = {best_m} mnd (r={m_corrs[best_m]:.3f})")

# --- Visualisering ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Krysskorrelasjonsanalyse: Brent oljepris vs. norske oljeserviceaksjer\n(ukentlige og månedlige avkastninger, 2018-2026)",
             fontsize=14, fontweight="bold", y=0.98)

# Panel 1: Ukentlig korrelasjon per aksje
ax1 = axes[0, 0]
lags_w = list(range(MAX_LAG_WEEKS + 1))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, (name, corrs) in enumerate(results.items()):
    best = np.nanargmax(corrs)
    ax1.plot(lags_w, corrs, label=f"{name} (best: {best}u)", color=colors[i], linewidth=1.5)
    ax1.scatter([best], [corrs[best]], color=colors[i], s=60, zorder=5, edgecolors="black")
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_xlabel("Lag (uker)", fontsize=11)
ax1.set_ylabel("Korrelasjon", fontsize=11)
ax1.set_title("Ukentlig: Korrelasjon per aksje", fontsize=11)
ax1.legend(fontsize=7.5, loc="upper right")
ax1.grid(True, alpha=0.3)

# Panel 2: Likevektet indeks (ukentlig)
ax2 = axes[0, 1]
if index_correlations:
    ax2.bar(lags_w, index_correlations, color="#3498db", alpha=0.7, edgecolor="white")
    best_w = np.nanargmax(index_correlations)
    ax2.bar(best_w, index_correlations[best_w], color="#e74c3c", alpha=0.9, edgecolor="black",
            label=f"Best lag: {best_w} uker (r={index_correlations[best_w]:.3f})")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Lag (uker)", fontsize=11)
    ax2.set_ylabel("Korrelasjon", fontsize=11)
    ax2.set_title("Ukentlig: Likevektet oljeserviceindeks", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

# Panel 3: Månedlig korrelasjon per aksje
ax3 = axes[1, 0]
lags_m = list(range(7))
for i, (name, corrs) in enumerate(monthly_results.items()):
    best = np.nanargmax(corrs)
    ax3.plot(lags_m, corrs, marker="o", label=f"{name} (best: {best}m)", color=colors[i], linewidth=1.5)
ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_xlabel("Lag (mnd)", fontsize=11)
ax3.set_ylabel("Korrelasjon", fontsize=11)
ax3.set_title("Månedlig: Korrelasjon per aksje", fontsize=11)
ax3.legend(fontsize=7.5, loc="upper right")
ax3.grid(True, alpha=0.3)

# Panel 4: Sammendragstabell
ax4 = axes[1, 1]
ax4.axis("off")
table_data = []
for name in results:
    w_corrs = results[name]
    best_w = np.nanargmax(w_corrs)
    m_corrs = monthly_results.get(name, [0])
    best_m = np.nanargmax(m_corrs) if m_corrs else 0
    table_data.append([
        name,
        f"{w_corrs[0]:.3f}",
        f"{best_w} uker",
        f"{w_corrs[best_w]:.3f}",
        f"{best_m} mnd",
        f"{m_corrs[best_m]:.3f}" if m_corrs else "N/A"
    ])

table = ax4.table(cellText=table_data,
                   colLabels=["Selskap", "Lag 0\n(ukentlig)", "Best lag\n(uker)", "Best r\n(uker)",
                              "Best lag\n(mnd)", "Best r\n(mnd)"],
                   loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.1, 1.6)
ax4.set_title("Sammendrag: Optimal lag per selskap", fontsize=11, pad=20)

# Highlight header row
for j in range(6):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.94])
output_path = "/Users/emmastrandenskaar/Documents/Claude/Projects/Oljepris/data/oilservice_lag_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nGraf lagret: {output_path}")

# --- Beregn median/gjennomsnitt lag for bruk i post-earnings justering ---
weekly_best_lags = []
for name, corrs in results.items():
    if name != "Aker BP":  # Aker BP er E&P, ikke ren oljeservice
        best = np.nanargmax(corrs)
        weekly_best_lags.append(best)

if weekly_best_lags:
    median_lag = int(np.median(weekly_best_lags))
    mean_lag = np.mean(weekly_best_lags)
    print(f"\nOljeservice median lag: {median_lag} uker ({mean_lag:.1f} gjennomsnitt)")
    print(f"Anbefalt justering: Bruk Brent-avkastning forskjøvet {median_lag} uker tilbake")
    print(f"  -> dvs. for post-earnings analyse, sammenlign OTL med Brent endring fra {median_lag} uker FØR rapport")
