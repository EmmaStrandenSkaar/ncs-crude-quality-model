#!/usr/bin/env python3
"""
Derivatives Lead-Lag Analysis: Aker BP (AKRBP.OL)

Analyzes whether derivatives market signals lead stock price movements.
Uses available proxies for institutional options sentiment:
  - OVX (CBOE Crude Oil Volatility Index) — options-implied oil volatility
  - ^VIX — broad market options sentiment
  - Brent crude futures (BZ=F) — futures market signal
  - AKRBP options chain (if available via yfinance)

Methods:
  1. Cross-correlation with time lags
  2. Granger causality tests
  3. Regime analysis (high vs low volatility periods)
  4. Rolling predictive power
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

# ── 1. Fetch data ──────────────────────────────────────────────────────────

print("=" * 70)
print("DERIVATIVES LEAD-LAG ANALYSIS: AKER BP")
print("=" * 70)

end = datetime.now()
start = end - timedelta(days=5*365)

tickers = {
    'AKRBP.OL': 'Aker BP',
    '^OVX': 'Oil VIX (OVX)',
    '^VIX': 'VIX',
    'BZ=F': 'Brent Futures',
}

print("\nHenter data...")
raw = {}
for ticker, name in tickers.items():
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if len(df) > 100:
            raw[ticker] = df
            print(f"  ✓ {name}: {len(df)} datapunkter ({df.index[0].date()} → {df.index[-1].date()})")
        else:
            print(f"  ✗ {name}: for lite data ({len(df)} rader)")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# Try fetching AKRBP options chain (current snapshot)
print("\nSjekker opsjonsdata for Aker BP...")
akrbp = yf.Ticker('AKRBP.OL')
try:
    exp_dates = akrbp.options
    if exp_dates:
        print(f"  ✓ Opsjoner tilgjengelig med {len(exp_dates)} forfallsdatoer")
        print(f"    Forfallsdatoer: {', '.join(exp_dates[:5])}{'...' if len(exp_dates) > 5 else ''}")
        all_opts = []
        for exp in exp_dates:
            chain = akrbp.option_chain(exp)
            calls = chain.calls.copy()
            calls['type'] = 'call'
            calls['expiry'] = exp
            puts = chain.puts.copy()
            puts['type'] = 'put'
            puts['expiry'] = exp
            all_opts.append(pd.concat([calls, puts]))
        opts_df = pd.concat(all_opts, ignore_index=True)
        total_call_vol = opts_df[opts_df['type'] == 'call']['volume'].sum()
        total_put_vol = opts_df[opts_df['type'] == 'put']['volume'].sum()
        total_call_oi = opts_df[opts_df['type'] == 'call']['openInterest'].sum()
        total_put_oi = opts_df[opts_df['type'] == 'put']['openInterest'].sum()
        print(f"    Nåværende put/call-ratio (volum): {total_put_vol / max(total_call_vol, 1):.2f}")
        print(f"    Nåværende put/call-ratio (OI):    {total_put_oi / max(total_call_oi, 1):.2f}")
        print(f"    Total åpen interesse: {total_call_oi + total_put_oi:,.0f} kontrakter")
    else:
        print("  ✗ Ingen opsjonsdata tilgjengelig via yfinance")
        opts_df = None
except Exception as e:
    print(f"  ✗ Ingen opsjonsdata: {e}")
    opts_df = None

# ── 2. Build aligned dataset ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("BYGGER DATASETT")
print("=" * 70)

# Handle multi-level columns from yfinance
def get_close(df, ticker=None):
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            return df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
        return df.iloc[:, 0]
    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 0]

def get_volume(df, ticker=None):
    if isinstance(df.columns, pd.MultiIndex):
        if 'Volume' in df.columns.get_level_values(0):
            return df['Volume'].iloc[:, 0] if df['Volume'].ndim > 1 else df['Volume']
    if 'Volume' in df.columns:
        return df['Volume']
    return None

panel = pd.DataFrame()
panel['akrbp_close'] = get_close(raw['AKRBP.OL'])
panel['akrbp_volume'] = get_volume(raw['AKRBP.OL'])
panel['akrbp_ret'] = panel['akrbp_close'].pct_change()

if '^OVX' in raw:
    panel['ovx'] = get_close(raw['^OVX'])
    panel['ovx_chg'] = panel['ovx'].pct_change()

if '^VIX' in raw:
    panel['vix'] = get_close(raw['^VIX'])
    panel['vix_chg'] = panel['vix'].pct_change()

if 'BZ=F' in raw:
    panel['brent'] = get_close(raw['BZ=F'])
    panel['brent_ret'] = panel['brent'].pct_change()
    panel['brent_volume'] = get_volume(raw['BZ=F'])

panel = panel.dropna(subset=['akrbp_close'])
panel = panel.ffill().dropna()

print(f"\nSamlet datasett: {len(panel)} handelsdager")
print(f"Periode: {panel.index[0].date()} → {panel.index[-1].date()}")

# ── 3. Cross-correlation analysis ─────────────────────────────────────────

print("\n" + "=" * 70)
print("KRYSSKORRELASJONSANALYSE (Lead-Lag)")
print("=" * 70)

max_lag = 10  # trading days

def cross_corr_analysis(series_x, series_y, name_x, name_y, max_lag=10):
    """Compute cross-correlations at various lags. Positive lag = x leads y."""
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = series_x.shift(lag).corr(series_y)
        elif lag < 0:
            corr = series_x.corr(series_y.shift(-lag))
        else:
            corr = series_x.corr(series_y)
        results.append({'lag': lag, 'correlation': corr})
    return pd.DataFrame(results)

pairs = []
if 'ovx_chg' in panel.columns:
    pairs.append(('ovx_chg', 'akrbp_ret', 'OVX endring', 'AKRBP avkastning'))
if 'vix_chg' in panel.columns:
    pairs.append(('vix_chg', 'akrbp_ret', 'VIX endring', 'AKRBP avkastning'))
if 'brent_ret' in panel.columns:
    pairs.append(('brent_ret', 'akrbp_ret', 'Brent avkastning', 'AKRBP avkastning'))

fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 5 * len(pairs)))
if len(pairs) == 1:
    axes = [axes]

for idx, (col_x, col_y, name_x, name_y) in enumerate(pairs):
    cc = cross_corr_analysis(panel[col_x], panel[col_y], name_x, name_y, max_lag)
    peak_lag = cc.loc[cc['correlation'].abs().idxmax(), 'lag']
    peak_corr = cc.loc[cc['correlation'].abs().idxmax(), 'correlation']

    print(f"\n  {name_x} → {name_y}:")
    print(f"    Sterkeste korrelasjon: {peak_corr:.4f} ved lag={int(peak_lag)} dager")
    if peak_lag > 0:
        print(f"    → {name_x} LEDER {name_y} med {int(peak_lag)} dag(er)")
    elif peak_lag < 0:
        print(f"    → {name_y} LEDER {name_x} med {int(-peak_lag)} dag(er)")
    else:
        print(f"    → Samtidig korrelasjon (ingen lead)")

    for lag_show in [-5, -3, -1, 0, 1, 3, 5]:
        row = cc[cc['lag'] == lag_show]
        if not row.empty:
            direction = f"{name_x} leder" if lag_show > 0 else (f"{name_y} leder" if lag_show < 0 else "samtidig")
            print(f"      lag={lag_show:+d}: r={row['correlation'].values[0]:.4f}  ({direction})")

    ax = axes[idx]
    colors = ['#e74c3c' if l < 0 else '#2ecc71' if l > 0 else '#3498db' for l in cc['lag']]
    ax.bar(cc['lag'], cc['correlation'], color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')

    n = len(panel)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(y=ci, color='gray', linewidth=0.8, linestyle=':', label=f'95% KI (±{ci:.4f})')
    ax.axhline(y=-ci, color='gray', linewidth=0.8, linestyle=':')

    ax.set_xlabel('Lag (dager) — Positiv = venstre variabel leder')
    ax.set_ylabel('Korrelasjon')
    ax.set_title(f'Kryss-korrelasjon: {name_x} → {name_y}')
    ax.legend()
    ax.set_xticks(range(-max_lag, max_lag + 1))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'derivatives_crosscorr.png'))
plt.close()
print(f"\n  → Lagret: data/derivatives_crosscorr.png")

# ── 4. Granger causality tests ────────────────────────────────────────────

print("\n" + "=" * 70)
print("GRANGER KAUSALITETSTEST")
print("=" * 70)
print("(Tester om derivat-signaler hjelper å forutsi aksjekursen)")

def run_granger(data, col_cause, col_effect, name, max_lag=5):
    test_df = data[[col_effect, col_cause]].dropna()

    adf_cause = adfuller(test_df[col_cause])
    adf_effect = adfuller(test_df[col_effect])
    print(f"\n  Stasjonæritetstest (ADF):")
    print(f"    {col_cause}: p={adf_cause[1]:.4f} {'✓ stasjonær' if adf_cause[1] < 0.05 else '✗ ikke-stasjonær'}")
    print(f"    {col_effect}: p={adf_effect[1]:.4f} {'✓ stasjonær' if adf_effect[1] < 0.05 else '✗ ikke-stasjonær'}")

    print(f"\n  Granger-test: '{name}' → AKRBP avkastning")
    print(f"  {'Lag':<6} {'F-stat':>10} {'p-verdi':>10} {'Signifikant?':>14}")
    print(f"  {'-'*42}")

    try:
        results = grangercausalitytests(test_df[[col_effect, col_cause]], maxlag=max_lag, verbose=False)
        significant_lags = []
        for lag in range(1, max_lag + 1):
            f_stat = results[lag][0]['ssr_ftest'][0]
            p_val = results[lag][0]['ssr_ftest'][1]
            sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
            print(f"  {lag:<6} {f_stat:>10.3f} {p_val:>10.4f} {sig:>14}")
            if p_val < 0.05:
                significant_lags.append(lag)
        if significant_lags:
            print(f"  → Signifikant ved lag {significant_lags}: derivat-signalet HAR prediktiv kraft")
        else:
            print(f"  → Ikke signifikant: derivat-signalet har IKKE prediktiv kraft for AKRBP")
        return results
    except Exception as e:
        print(f"  Feil: {e}")
        return None

granger_pairs = []
if 'ovx_chg' in panel.columns:
    granger_pairs.append(('ovx_chg', 'akrbp_ret', 'OVX (oljeopsjon-volatilitet)'))
if 'vix_chg' in panel.columns:
    granger_pairs.append(('vix_chg', 'akrbp_ret', 'VIX (markedsopsjon-volatilitet)'))
if 'brent_ret' in panel.columns:
    granger_pairs.append(('brent_ret', 'akrbp_ret', 'Brent-futures avkastning'))

for col_cause, col_effect, name in granger_pairs:
    run_granger(panel, col_cause, col_effect, name, max_lag=5)

# Also test reverse: does AKRBP lead derivatives?
print("\n" + "-" * 70)
print("OMVENDT TEST: Leder AKRBP derivatmarkedene?")
for col_effect, col_cause, name in granger_pairs:
    run_granger(panel, 'akrbp_ret', col_effect, f'AKRBP → {name}', max_lag=5)

# ── 5. Volatility regime analysis ─────────────────────────────────────────

print("\n" + "=" * 70)
print("REGIME-ANALYSE: Er prediktiv kraft sterkere i volatile perioder?")
print("=" * 70)

if 'ovx' in panel.columns:
    panel['ovx_regime'] = pd.qcut(panel['ovx'], q=3, labels=['Lav vol', 'Medium vol', 'Høy vol'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, regime in enumerate(['Lav vol', 'Medium vol', 'Høy vol']):
        subset = panel[panel['ovx_regime'] == regime]

        if 'brent_ret' in subset.columns and len(subset) > 50:
            lags = range(-5, 6)
            corrs = []
            for lag in lags:
                if lag != 0:
                    shifted = subset['brent_ret'].shift(lag)
                    corrs.append(shifted.corr(subset['akrbp_ret']))
                else:
                    corrs.append(subset['brent_ret'].corr(subset['akrbp_ret']))

            colors = ['#e74c3c' if l < 0 else '#2ecc71' if l > 0 else '#3498db' for l in lags]
            axes[i].bar(list(lags), corrs, color=colors, alpha=0.7, edgecolor='white')
            axes[i].axhline(y=0, color='black', linewidth=0.5)
            axes[i].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
            ci = 1.96 / np.sqrt(len(subset))
            axes[i].axhline(y=ci, color='gray', linewidth=0.8, linestyle=':')
            axes[i].axhline(y=-ci, color='gray', linewidth=0.8, linestyle=':')
            axes[i].set_title(f'{regime} (OVX)\nn={len(subset)}')
            axes[i].set_xlabel('Lag (dager)')
            axes[i].set_ylabel('Korrelasjon')

            peak_idx = np.argmax(np.abs(corrs))
            peak_lag = list(lags)[peak_idx]
            print(f"\n  {regime}: Sterkeste korrelasjon r={corrs[peak_idx]:.4f} ved lag={peak_lag}")

    plt.suptitle('Brent → AKRBP Lead-Lag etter OVX-regime', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'derivatives_regime_analysis.png'))
    plt.close()
    print(f"\n  → Lagret: data/derivatives_regime_analysis.png")

# ── 6. Rolling predictive regression ──────────────────────────────────────

print("\n" + "=" * 70)
print("RULLENDE PREDIKTIV REGRESJON")
print("=" * 70)
print("(Bruker derivat-signaler fra i går til å predikere AKRBP i dag)")

window = 120  # ~6 months rolling

predictors = []
pred_names = []
if 'ovx_chg' in panel.columns:
    predictors.append('ovx_chg')
    pred_names.append('OVX')
if 'brent_ret' in panel.columns:
    predictors.append('brent_ret')
    pred_names.append('Brent')
if 'vix_chg' in panel.columns:
    predictors.append('vix_chg')
    pred_names.append('VIX')

# Create lagged predictors (yesterday's derivative signal → today's stock return)
for col in predictors:
    panel[f'{col}_lag1'] = panel[col].shift(1)

lagged_preds = [f'{col}_lag1' for col in predictors]
panel_reg = panel.dropna(subset=lagged_preds + ['akrbp_ret'])

rolling_r2 = []
rolling_dates = []
rolling_coeffs = {name: [] for name in pred_names}

for i in range(window, len(panel_reg)):
    sub = panel_reg.iloc[i - window:i]
    y = sub['akrbp_ret']
    X = add_constant(sub[lagged_preds])
    try:
        model = OLS(y, X).fit()
        rolling_r2.append(model.rsquared)
        rolling_dates.append(sub.index[-1])
        for j, name in enumerate(pred_names):
            rolling_coeffs[name].append(model.params.iloc[j + 1])
    except:
        continue

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(rolling_dates, rolling_r2, color='#2c3e50', linewidth=1.2)
ax1.fill_between(rolling_dates, rolling_r2, alpha=0.3, color='#3498db')
ax1.set_ylabel('R² (forklaringskraft)')
ax1.set_title(f'Rullende R² — Derivat-signaler (t-1) → AKRBP avkastning (t)\n(vindu={window} dager)')
ax1.axhline(y=np.mean(rolling_r2), color='red', linestyle='--', label=f'Gjennomsnitt R²={np.mean(rolling_r2):.4f}')
ax1.legend()

colors_coeff = ['#e74c3c', '#2ecc71', '#9b59b6']
for i, (name, coeffs) in enumerate(rolling_coeffs.items()):
    ax2.plot(rolling_dates[:len(coeffs)], coeffs, label=name, color=colors_coeff[i % len(colors_coeff)], linewidth=1.0, alpha=0.8)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel('Koeffisient (β)')
ax2.set_xlabel('Dato')
ax2.set_title('Rullende regresjonskoeffisienter for laggede derivat-signaler')
ax2.legend()

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'derivatives_rolling_regression.png'))
plt.close()
print(f"\n  Gjennomsnittlig R²: {np.mean(rolling_r2):.4f}")
print(f"  Maks R²: {np.max(rolling_r2):.4f}")
print(f"  → Lagret: data/derivatives_rolling_regression.png")

# ── 7. Full-sample predictive regression ──────────────────────────────────

print("\n" + "=" * 70)
print("FULL-SAMPLE PREDIKTIV REGRESJON")
print("=" * 70)

y = panel_reg['akrbp_ret']
X = add_constant(panel_reg[lagged_preds])
model = OLS(y, X).fit()
print(model.summary().tables[0])
print(model.summary().tables[1])

# ── 8. Cumulative signal strategy backtest ────────────────────────────────

print("\n" + "=" * 70)
print("ENKEL BACKTEST: Derivat-signal handelsstrategio")
print("=" * 70)

if 'ovx_chg' in panel.columns and 'brent_ret' in panel.columns:
    bt = panel_reg.copy()

    # Signal: Buy when OVX dropped yesterday (less fear) AND Brent rose yesterday
    bt['signal'] = 0
    bt.loc[(bt['ovx_chg_lag1'] < 0) & (bt['brent_ret_lag1'] > 0), 'signal'] = 1   # bullish
    bt.loc[(bt['ovx_chg_lag1'] > 0) & (bt['brent_ret_lag1'] < 0), 'signal'] = -1  # bearish

    bt['strategy_ret'] = bt['signal'] * bt['akrbp_ret']
    bt['cum_strategy'] = (1 + bt['strategy_ret']).cumprod()
    bt['cum_buyhold'] = (1 + bt['akrbp_ret']).cumprod()

    total_strat = bt['cum_strategy'].iloc[-1] - 1
    total_bh = bt['cum_buyhold'].iloc[-1] - 1
    days_in = (bt['signal'] != 0).sum()
    pct_in = days_in / len(bt) * 100

    win_rate = (bt.loc[bt['signal'] != 0, 'strategy_ret'] > 0).mean()
    sharpe_strat = bt['strategy_ret'].mean() / bt['strategy_ret'].std() * np.sqrt(252) if bt['strategy_ret'].std() > 0 else 0
    sharpe_bh = bt['akrbp_ret'].mean() / bt['akrbp_ret'].std() * np.sqrt(252) if bt['akrbp_ret'].std() > 0 else 0

    print(f"\n  Strategi: Kjøp når OVX falt og Brent steg dagen før")
    print(f"           Selg/short når OVX steg og Brent falt dagen før")
    print(f"\n  Dager med posisjon: {days_in} / {len(bt)} ({pct_in:.1f}%)")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"\n  Kumulativ avkastning:")
    print(f"    Strategi:   {total_strat:>8.1%}")
    print(f"    Buy & Hold: {total_bh:>8.1%}")
    print(f"\n  Sharpe-ratio (annualisert):")
    print(f"    Strategi:   {sharpe_strat:>8.3f}")
    print(f"    Buy & Hold: {sharpe_bh:>8.3f}")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(bt.index, bt['cum_strategy'], label=f'Derivat-signal strategi ({total_strat:.1%})', color='#2ecc71', linewidth=1.5)
    ax.plot(bt.index, bt['cum_buyhold'], label=f'Buy & Hold AKRBP ({total_bh:.1%})', color='#3498db', linewidth=1.5)
    ax.axhline(y=1, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Kumulativ avkastning (1 = startverdi)')
    ax.set_xlabel('Dato')
    ax.set_title('Backtest: Derivat-signal strategi vs. Buy & Hold (Aker BP)')
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'derivatives_backtest.png'))
    plt.close()
    print(f"\n  → Lagret: data/derivatives_backtest.png")

# ── 9. Summary ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("OPPSUMMERING")
print("=" * 70)
print("""
Denne analysen tester om derivatmarkeds-signaler (opsjonsvolatilitet,
futures) kan predikere Aker BP's aksjekurs.

Metoder brukt:
  1. Kryss-korrelasjon med tidsforskyvning (±10 dager)
  2. Granger kausalitetstest (opptil 5 lags)
  3. Regime-analyse (lav/medium/høy volatilitet)
  4. Rullende prediktiv regresjon (120-dagers vindu)
  5. Enkel handelsstrategi-backtest

Viktige begrensninger:
  - Bruker OVX/VIX som proxy for opsjonssentiment (ikke AKRBP-spesifikke opsjoner)
  - Brent-futures er det nærmeste vi kommer institusjonell derivat-handel
  - Historisk opsjonsdata for norske aksjer krever typisk Bloomberg/Refinitiv
  - Backtesten inkluderer ikke transaksjonskostnader eller slippage
  - In-sample backtest → reell out-of-sample test ville gitt mer robuste resultater
""")
