"""
OTL post-earnings drift med lagget Brent-justering.

Forbedringer vs. script 12:
  - Brent-avkastning starter L handelsdager FØR rapportdatoen (for å fange lag)
  - Viser både lag=0 og lag=3 uker (15 handelsdager) justering
  - Ekstra panel: scatter av Brent-endring (lagget) vs. OTL alpha
"""
from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

TICKER = "OTL.OL"
COMPANY = "Odfjell Technology"
BRENT_TICKER = "BZ=F"
DAYS_AFTER = 30
LAG_DAYS = 15  # 3 uker x 5 handelsdager (sektormedian fra lag-analyse)

# EBITDA og revenue fra kvartalsrapportene (NOK millioner)
fundamentals = {
    "2024-05-16": {"quarter": "Q1 2024"},
    "2024-08-22": {"quarter": "Q2 2024"},
    "2024-11-07": {"quarter": "Q3 2024"},
    "2025-02-14": {"quarter": "Q4 2024", "rev": 1450, "ebitda": 191},
    "2025-05-16": {"quarter": "Q1 2025", "rev": 1373, "ebitda": 193},
    "2025-08-21": {"quarter": "Q2 2025", "rev": 1373, "ebitda": 193},
    "2025-11-07": {"quarter": "Q3 2025", "rev": 1434, "ebitda": 202},
    "2026-02-26": {"quarter": "Q4 2025", "rev": 1396, "ebitda": 213},
}

stock = yf.Ticker(TICKER)
earnings_df = stock.earnings_dates
past_earnings = earnings_df[earnings_df["Reported EPS"].notna()].copy()
past_earnings = past_earnings.sort_index(ascending=False)
report_dates = sorted(past_earnings.head(6).index.tolist())

earliest = min(report_dates) - timedelta(days=LAG_DAYS + 10)
latest = max(report_dates) + timedelta(days=DAYS_AFTER + 10)
hist = stock.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
hist.index = hist.index.tz_localize(None)

brent = yf.Ticker(BRENT_TICKER)
brent_hist = brent.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
brent_hist.index = brent_hist.index.tz_localize(None)

# --- Beregning ---
fig, axes = plt.subplots(3, 1, figsize=(16, 15),
                         gridspec_kw={"height_ratios": [3, 1.5, 1.3], "hspace": 0.4})
ax_price, ax_bar, ax_eps = axes
fig.suptitle(f"{COMPANY} ({TICKER}) — Meravkastning vs. Brent (lagget {LAG_DAYS} handelsdager)\n"
             f"Totalavkastning {DAYS_AFTER} handelsdager etter kvartalsrapport",
             fontsize=13, fontweight="bold", y=0.99)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
reaction_data = []

for i, report_date in enumerate(report_dates):
    rd_naive = report_date.tz_localize(None)
    date_key = report_date.strftime("%Y-%m-%d")
    fund = fundamentals.get(date_key, {})
    quarter = fund.get("quarter", date_key)

    # OTL totalavkastning
    mask = hist.index >= rd_naive
    post_data = hist.loc[mask].head(DAYS_AFTER + 1)
    if len(post_data) < 5:
        continue

    base_price = post_data["Close"].iloc[0]
    cum_div = post_data["Dividends"].cumsum()
    total_return = ((post_data["Close"] + cum_div - base_price) / base_price) * 100

    # --- Brent UTEN lag (som før) ---
    brent_mask = brent_hist.index >= rd_naive
    brent_post = brent_hist.loc[brent_mask].head(DAYS_AFTER + 1)
    if len(brent_post) >= 2:
        brent_base = brent_post["Close"].iloc[0]
        brent_return_nolag = ((brent_post["Close"] / brent_base) - 1) * 100
    else:
        brent_return_nolag = pd.Series([0] * len(total_return))

    # --- Brent MED lag: starter LAG_DAYS handelsdager FØR rapport ---
    # Finn Brent-pris LAG_DAYS handelsdager før rapportdatoen
    brent_before = brent_hist.loc[brent_hist.index < rd_naive]
    if len(brent_before) >= LAG_DAYS:
        brent_lag_start_date = brent_before.index[-LAG_DAYS]
        brent_lag_base = brent_hist.loc[brent_lag_start_date, "Close"]

        # Brent endring fra (rapport - LAG) til (rapport + X) for hver dag
        brent_lag_mask = brent_hist.index >= brent_lag_start_date
        brent_lag_window = brent_hist.loc[brent_lag_mask].head(LAG_DAYS + DAYS_AFTER + 1)
        brent_return_lagged_full = ((brent_lag_window["Close"] / brent_lag_base) - 1) * 100

        # Vi trenger bare den delen som dekker post-rapport perioden
        # Skift indeks: trading day 0 = rapportdag
        brent_at_report = brent_return_lagged_full.loc[brent_return_lagged_full.index >= rd_naive]
        brent_return_lagged = brent_at_report.head(DAYS_AFTER + 1)
    else:
        brent_return_lagged = brent_return_nolag.copy()

    # --- Meravkastning ---
    min_len_nolag = min(len(total_return), len(brent_return_nolag))
    min_len_lag = min(len(total_return), len(brent_return_lagged))
    trading_days = list(range(min_len_nolag))

    excess_nolag = total_return.values[:min_len_nolag] - brent_return_nolag.values[:min_len_nolag]
    excess_lagged = total_return.values[:min_len_lag] - brent_return_lagged.values[:min_len_lag]

    # Sluttavkastning
    otl_30d = total_return.iloc[min_len_nolag - 1]
    brent_30d_nolag = brent_return_nolag.iloc[min_len_nolag - 1]
    brent_30d_lagged = brent_return_lagged.iloc[min_len_lag - 1] if min_len_lag > 0 else 0
    alpha_nolag = otl_30d - brent_30d_nolag
    alpha_lagged = otl_30d - brent_30d_lagged

    # Brent endring over lag-perioden (før rapport)
    if len(brent_before) >= LAG_DAYS:
        brent_pre_change = ((brent_hist.loc[brent_before.index[-1], "Close"] / brent_lag_base) - 1) * 100
    else:
        brent_pre_change = 0

    reaction_data.append((quarter, otl_30d, brent_30d_nolag, alpha_nolag,
                          brent_30d_lagged, alpha_lagged, brent_pre_change))

    # --- Plot ---
    label = f"{quarter} — Alpha(lag): {alpha_lagged:+.1f}% | Alpha(0): {alpha_nolag:+.1f}%"

    # Lagget meravkastning (hoved)
    ax_price.plot(list(range(min_len_lag)), excess_lagged, color=colors[i],
                  linewidth=2, alpha=0.85, label=label)
    # Ulagget meravkastning (dimmet referanse)
    ax_price.plot(trading_days, excess_nolag, color=colors[i],
                  linewidth=1, alpha=0.25, linestyle="--")

# --- Panel 1: Meravkastning ---
ax_price.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax_price.set_xlabel(f"Handelsdager etter rapport", fontsize=11)
ax_price.set_ylabel("Meravkastning vs. Brent (%)", fontsize=11)
ax_price.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
ax_price.grid(True, alpha=0.3)
ax_price.set_xlim(0, DAYS_AFTER)
ax_price.text(0.98, 0.02,
              f"Heltrukken = lagget Brent ({LAG_DAYS}d), stiplet = ikke lagget",
              transform=ax_price.transAxes, fontsize=8, ha="right", va="bottom",
              style="italic", alpha=0.7)

# --- Panel 2: Sammenligning lagget vs. ulagget alpha ---
if reaction_data:
    quarters = [r[0] for r in reaction_data]
    otl_30 = [r[1] for r in reaction_data]
    alpha_0 = [r[3] for r in reaction_data]
    alpha_lag = [r[5] for r in reaction_data]
    brent_pre = [r[6] for r in reaction_data]

    x_pos = np.arange(len(quarters))
    w = 0.2

    ax_bar.bar(x_pos - w*1.5, otl_30, w, color="#3498db", alpha=0.7,
               label="OTL totalavk. 30d", edgecolor="white")
    alpha0_colors = ["#2ecc71" if a > 0 else "#e74c3c" for a in alpha_0]
    ax_bar.bar(x_pos - w*0.5, alpha_0, w, color=alpha0_colors, alpha=0.5,
               label="Alpha (lag=0)", edgecolor="white")
    alphaL_colors = ["#27ae60" if a > 0 else "#c0392b" for a in alpha_lag]
    ax_bar.bar(x_pos + w*0.5, alpha_lag, w, color=alphaL_colors, alpha=0.8,
               label=f"Alpha (lag={LAG_DAYS}d)", edgecolor="white")
    ax_bar.bar(x_pos + w*1.5, brent_pre, w, color="#95a5a6", alpha=0.5,
               label=f"Brent pre-rapport ({LAG_DAYS}d)", edgecolor="white")

    # Annotations
    for j in range(len(quarters)):
        ax_bar.annotate(f"{alpha_lag[j]:+.1f}%",
                        xy=(x_pos[j] + w*0.5, alpha_lag[j]),
                        xytext=(0, 5 if alpha_lag[j] > 0 else -12),
                        textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#27ae60" if alpha_lag[j] > 0 else "#c0392b")

    ax_bar.axhline(y=0, color="black", linewidth=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(quarters, rotation=45, ha="right", fontsize=9)
    ax_bar.set_ylabel("Avkastning / Alpha (%)", fontsize=10)
    ax_bar.legend(loc="best", fontsize=8)
    ax_bar.grid(True, alpha=0.3)
    ax_bar.set_title(f"OTL meravkastning: Lagget ({LAG_DAYS}d) vs. ulagget Brent-justering", fontsize=11)

# --- Panel 3: EPS surprise ---
eps_data = []
for rd in report_dates:
    date_key = rd.strftime("%Y-%m-%d")
    fund = fundamentals.get(date_key, {})
    quarter = fund.get("quarter", date_key)
    eps_est = past_earnings.loc[rd, "EPS Estimate"]
    eps_act = past_earnings.loc[rd, "Reported EPS"]
    surprise = past_earnings.loc[rd, "Surprise(%)"]
    if pd.notna(eps_est) and pd.notna(eps_act):
        eps_data.append((quarter, eps_est, eps_act, surprise))

if eps_data:
    quarters_eps = [e[0] for e in eps_data]
    estimates = [e[1] for e in eps_data]
    actuals = [e[2] for e in eps_data]
    surprises = [e[3] for e in eps_data]

    x_pos = np.arange(len(quarters_eps))
    w = 0.35

    ax_eps.bar(x_pos - w/2, actuals, w, color="#3498db", alpha=0.8, label="Faktisk EPS (NOK)")
    ax_eps.bar(x_pos + w/2, estimates, w, color="#95a5a6", alpha=0.6, label="Estimat EPS (NOK)")

    for j, s in enumerate(surprises):
        y_pos = max(actuals[j], estimates[j]) + 0.05
        ax_eps.annotate(f"{s:+.1f}%",
                       xy=(x_pos[j], y_pos),
                       ha="center", fontsize=9, fontweight="bold",
                       color="#2ecc71" if s > 0 else "#e74c3c")

    ax_eps.set_xticks(x_pos)
    ax_eps.set_xticklabels(quarters_eps, rotation=45, ha="right", fontsize=9)
    ax_eps.set_ylabel("EPS (NOK)", fontsize=10)
    ax_eps.legend(loc="best", fontsize=9)
    ax_eps.grid(True, alpha=0.3, axis="y")
    ax_eps.set_title("EPS: Faktisk vs. estimat (kilde: Yahoo Finance — ta med forbehold)", fontsize=11)

plt.subplots_adjust(top=0.92, bottom=0.06)
output_path = str(Path(__file__).resolve().parents[2] / "data" / "otl_post_earnings_lagged.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Graf lagret: {output_path}")

# Print oppsummering
print(f"\n{'='*70}")
print(f"OPPSUMMERING: OTL meravkastning med {LAG_DAYS} handelsdagers Brent-lag")
print(f"{'='*70}")
for r in reaction_data:
    q, otl, b0, a0, bL, aL, bp = r
    print(f"  {q:10s}: OTL={otl:+6.1f}%  Alpha(0)={a0:+6.1f}%  Alpha(lag)={aL:+6.1f}%  Brent pre-rapport={bp:+5.1f}%")
avg_a0 = np.mean([r[3] for r in reaction_data])
avg_aL = np.mean([r[5] for r in reaction_data])
print(f"\n  Gjennomsnitt Alpha(lag=0):  {avg_a0:+.1f}%")
print(f"  Gjennomsnitt Alpha(lag={LAG_DAYS}d): {avg_aL:+.1f}%")
