"""
Analyse v2: Fundamental lag mellom oljepris og oljeservice.

Metode:
  1. Kumulativ Brent-avkastning over N uker -> predikerer OilService avk. de neste M ukene?
  2. Rolling kvartalsavkastning (12 uker) med lag
  3. Prediktiv regresjon: Brent(t-L, t) -> OilService(t, t+12)
  4. Granger-lignende: tester om fortidige Brent-endringer forklarer
     fremtidige oljeservice-avkastninger bedre enn samtidige
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
    "SUBC.OL": "Subsea 7",
    "TGS.OL": "TGS",
    "AKSO.OL": "Aker Solutions",
    "PGS.OL": "PGS",
}
# Aker BP separat som E&P referanse
EP_TICKERS = {"AKRBP.OL": "Aker BP", "EQNR.OL": "Equinor"}

START = "2018-01-01"
END = "2026-05-11"

# --- Hent data ---
print("Henter data...")
brent = yf.Ticker(BRENT).history(start=START, end=END)
brent.index = brent.index.tz_localize(None)
brent_weekly = brent["Close"].resample("W-FRI").last().dropna()

all_tickers = {**OILSERVICE_TICKERS, **EP_TICKERS}
stock_weekly = {}
for ticker, name in all_tickers.items():
    try:
        hist = yf.Ticker(ticker).history(start=START, end=END)
        hist.index = hist.index.tz_localize(None)
        weekly = hist["Close"].resample("W-FRI").last().dropna()
        if len(weekly) > 100:
            stock_weekly[name] = weekly
            print(f"  {name}: {len(weekly)} uker")
    except Exception as e:
        print(f"  {name}: feil - {e}")

# --- Bygg likevektet oljeserviceindeks ---
svc_names = [n for n in OILSERVICE_TICKERS.values() if n in stock_weekly]
svc_prices = pd.concat([stock_weekly[n] for n in svc_names], axis=1, join="inner")
svc_prices.columns = svc_names
# Normaliser til 100 ved start, så likevekt
svc_norm = svc_prices / svc_prices.iloc[0] * 100
svc_index = svc_norm.mean(axis=1)

ep_names = [n for n in EP_TICKERS.values() if n in stock_weekly]
ep_prices = pd.concat([stock_weekly[n] for n in ep_names], axis=1, join="inner")
ep_prices.columns = ep_names
ep_norm = ep_prices / ep_prices.iloc[0] * 100
ep_index = ep_norm.mean(axis=1)

# --- Metode 1: Rolling kvartalsavkastning med lag ---
# Test: Brent 12-ukers avkastning vs. OilService 12-ukers avkastning med lag L
WINDOW = 12  # uker (ca 1 kvartal)
MAX_LAG = 26  # uker

brent_aligned = brent_weekly.reindex(svc_index.index).dropna()
common_idx = svc_index.reindex(brent_aligned.index).dropna().index
brent_aligned = brent_aligned.loc[common_idx]
svc_aligned = svc_index.loc[common_idx]

brent_roll_ret = brent_aligned.pct_change(WINDOW).dropna()
svc_roll_ret = svc_aligned.pct_change(WINDOW).dropna()

# Krysskorrelasjoner for rolling kvartalsavkastninger
print("\n--- Rolling kvartalsavkastning: Brent vs. Oljeservice-indeks ---")
rolling_corrs = []
for lag in range(-MAX_LAG, MAX_LAG + 1):
    if lag > 0:
        # Positiv lag: service LAGGER brent
        shifted_brent = brent_roll_ret.shift(lag)
    elif lag < 0:
        # Negativ lag: service LEDER brent
        shifted_brent = brent_roll_ret.shift(lag)
    else:
        shifted_brent = brent_roll_ret

    valid = pd.concat([shifted_brent, svc_roll_ret], axis=1).dropna()
    if len(valid) > 30:
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
    else:
        corr = np.nan
    rolling_corrs.append(corr)

lag_range = list(range(-MAX_LAG, MAX_LAG + 1))
best_lag_idx = np.nanargmax(rolling_corrs)
best_lag = lag_range[best_lag_idx]
print(f"  Best lag: {best_lag} uker (r={rolling_corrs[best_lag_idx]:.3f})")
print(f"  Lag 0:    r={rolling_corrs[MAX_LAG]:.3f}")

# --- Metode 2: Prediktiv regresjon ---
# Brent(t-L, t) -> OilService(t, t+12)
print("\n--- Prediktiv regresjon: Brent-endring -> Fremtidig oljeservice ---")
FORWARD = 12  # uker fremover for oljeservice

brent_past_rets = {}
for L in [0, 4, 8, 12, 16, 20, 26]:
    brent_past_rets[L] = brent_aligned.pct_change(L).dropna() if L > 0 else pd.Series(0, index=brent_aligned.index)

svc_fwd_ret = svc_aligned.pct_change(FORWARD).shift(-FORWARD).dropna()

pred_r2 = {}
for L in [4, 8, 12, 16, 20, 26]:
    common = pd.concat([brent_past_rets[L], svc_fwd_ret], axis=1, join="inner").dropna()
    common.columns = ["brent_past", "svc_future"]
    if len(common) > 30:
        corr = common["brent_past"].corr(common["svc_future"])
        r2 = corr ** 2
        pred_r2[L] = (corr, r2, len(common))
        print(f"  Brent endring siste {L:2d} uker -> OilService neste {FORWARD} uker: r={corr:.3f}, R²={r2:.3f} (n={len(common)})")

# --- Metode 3: Nivå-lag (rullerende 26-ukers endring) ---
LONG_WINDOW = 26
brent_26w = brent_aligned.pct_change(LONG_WINDOW).dropna()
svc_26w = svc_aligned.pct_change(LONG_WINDOW).dropna()

level_corrs = []
for lag in range(0, MAX_LAG + 1):
    shifted = brent_26w.shift(lag)
    valid = pd.concat([shifted, svc_26w], axis=1).dropna()
    if len(valid) > 30:
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
    else:
        corr = np.nan
    level_corrs.append(corr)

best_level_lag = np.nanargmax(level_corrs)
print(f"\n--- 26-ukers endring lag-korrelasjon ---")
print(f"  Best lag: {best_level_lag} uker (r={level_corrs[best_level_lag]:.3f})")

# --- Metode 4: Per-aksje med rolling kvartal ---
print("\n--- Per aksje: rolling kvartalavkastning (12 uker) ---")
per_stock_lags = {}
for name in svc_names:
    s_aligned = stock_weekly[name].reindex(brent_aligned.index).dropna()
    ci = s_aligned.index.intersection(brent_aligned.index)
    b_a = brent_aligned.loc[ci]
    s_a = s_aligned.loc[ci]

    b_ret = b_a.pct_change(WINDOW).dropna()
    s_ret = s_a.pct_change(WINDOW).dropna()

    corrs_stock = []
    for lag in range(0, MAX_LAG + 1):
        shifted = b_ret.shift(lag)
        valid = pd.concat([shifted, s_ret], axis=1).dropna()
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1]) if len(valid) > 30 else np.nan
        corrs_stock.append(corr)

    best = np.nanargmax(corrs_stock)
    per_stock_lags[name] = (best, corrs_stock[best], corrs_stock)
    print(f"  {name:20s}: Best lag = {best} uker (r={corrs_stock[best]:.3f}), Lag 0 = {corrs_stock[0]:.3f}")

# E&P for sammenligning
for name in ep_names:
    if name not in stock_weekly:
        continue
    s_aligned = stock_weekly[name].reindex(brent_aligned.index).dropna()
    ci = s_aligned.index.intersection(brent_aligned.index)
    b_a = brent_aligned.loc[ci]
    s_a = s_aligned.loc[ci]

    b_ret = b_a.pct_change(WINDOW).dropna()
    s_ret = s_a.pct_change(WINDOW).dropna()

    corrs_stock = []
    for lag in range(0, MAX_LAG + 1):
        shifted = b_ret.shift(lag)
        valid = pd.concat([shifted, s_ret], axis=1).dropna()
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1]) if len(valid) > 30 else np.nan
        corrs_stock.append(corr)

    best = np.nanargmax(corrs_stock)
    per_stock_lags[name] = (best, corrs_stock[best], corrs_stock)
    print(f"  {name:20s}: Best lag = {best} uker (r={corrs_stock[best]:.3f}), Lag 0 = {corrs_stock[0]:.3f}  [E&P]")

# --- Visualisering ---
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle("Fundamental lag: Oljepris vs. oljeserviceaksjer (Oslo Børs)\nKvartalsavkastninger (12 uker rolling), 2018-2026",
             fontsize=14, fontweight="bold", y=0.99)

# Panel 1: Rolling kvartal cross-corr for indeks
ax1 = axes[0, 0]
ax1.bar(lag_range, rolling_corrs, color=["#e74c3c" if l < 0 else "#3498db" for l in lag_range],
        alpha=0.6, edgecolor="white", width=0.8)
ax1.bar(best_lag, rolling_corrs[best_lag_idx], color="#2ecc71", alpha=0.9, edgecolor="black",
        label=f"Best lag: {best_lag} uker (r={rolling_corrs[best_lag_idx]:.3f})")
ax1.axvline(x=0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax1.set_xlabel("Lag (uker, positiv = service lagger)", fontsize=10)
ax1.set_ylabel("Korrelasjon", fontsize=10)
ax1.set_title("Krysskorrelasjoner: Brent 12u vs. Oljeservice-indeks 12u", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Per-aksje lag-korrelasjoner
ax2 = axes[0, 1]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
lags = list(range(MAX_LAG + 1))
for i, (name, (best, best_r, corrs)) in enumerate(per_stock_lags.items()):
    is_ep = name in ep_names
    ls = "--" if is_ep else "-"
    lw = 1.0 if is_ep else 1.8
    label = f"{name} ({best}u, r={best_r:.2f})" + (" [E&P]" if is_ep else "")
    ax2.plot(lags, corrs, label=label, color=colors[i], linewidth=lw, linestyle=ls, alpha=0.85)
    ax2.scatter([best], [best_r], color=colors[i], s=50, zorder=5, edgecolors="black")
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_xlabel("Lag (uker)", fontsize=10)
ax2.set_ylabel("Korrelasjon", fontsize=10)
ax2.set_title("Per aksje: Rolling 12u avkastning korrelasjon", fontsize=11)
ax2.legend(fontsize=7, loc="upper right")
ax2.grid(True, alpha=0.3)

# Panel 3: 26-ukers nivåendring lag
ax3 = axes[1, 0]
lags_26 = list(range(MAX_LAG + 1))
ax3.bar(lags_26, level_corrs, color="#3498db", alpha=0.6, edgecolor="white")
ax3.bar(best_level_lag, level_corrs[best_level_lag], color="#e74c3c", alpha=0.9, edgecolor="black",
        label=f"Best lag: {best_level_lag} uker (r={level_corrs[best_level_lag]:.3f})")
ax3.set_xlabel("Lag (uker)", fontsize=10)
ax3.set_ylabel("Korrelasjon", fontsize=10)
ax3.set_title("26-ukers endring: Brent vs. Oljeservice-indeks", fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Prediktiv regresjon R²
ax4 = axes[1, 1]
if pred_r2:
    lookbacks = sorted(pred_r2.keys())
    corrs_pred = [pred_r2[L][0] for L in lookbacks]
    r2s = [pred_r2[L][1] for L in lookbacks]

    ax4_twin = ax4.twinx()
    bars = ax4.bar(lookbacks, corrs_pred, width=2.5, color="#3498db", alpha=0.7, label="Korrelasjon (r)")
    ax4_twin.plot(lookbacks, r2s, "o-", color="#e74c3c", linewidth=2, label="R²")

    ax4.set_xlabel("Brent lookback-periode (uker)", fontsize=10)
    ax4.set_ylabel("Korrelasjon (r)", fontsize=10, color="#3498db")
    ax4_twin.set_ylabel("R²", fontsize=10, color="#e74c3c")
    ax4.set_title(f"Prediktiv: Brent fortid -> Oljeservice neste {FORWARD}u", fontsize=11)
    ax4.legend(loc="upper left", fontsize=9)
    ax4_twin.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
output_path = "/Users/emmastrandenskaar/Documents/Claude/Projects/Oljepris/data/oilservice_lag_v2.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nGraf lagret: {output_path}")

# --- Oppsummering ---
svc_best_lags = [per_stock_lags[n][0] for n in svc_names if n in per_stock_lags]
if svc_best_lags:
    median_svc = int(np.median(svc_best_lags))
    mean_svc = np.mean(svc_best_lags)
    print(f"\n{'='*60}")
    print(f"OPPSUMMERING")
    print(f"{'='*60}")
    print(f"Oljeservice (rolling 12u avkastning):")
    print(f"  Median optimal lag: {median_svc} uker")
    print(f"  Gjennomsnitt:       {mean_svc:.1f} uker")
    for n in svc_names:
        if n in per_stock_lags:
            print(f"    {n:20s}: {per_stock_lags[n][0]} uker")
    print(f"\nE&P selskaper:")
    for n in ep_names:
        if n in per_stock_lags:
            print(f"    {n:20s}: {per_stock_lags[n][0]} uker")
    print(f"\nAnbefaling for OTL post-earnings justering:")
    otl_lag = per_stock_lags.get("Odfjell Technology", (0, 0, []))[0]
    print(f"  OTL-spesifikk lag: {otl_lag} uker")
    print(f"  Sektormedian lag:  {median_svc} uker")
    print(f"  -> Juster Brent-benchmark med {max(otl_lag, median_svc)} ukers lag")
