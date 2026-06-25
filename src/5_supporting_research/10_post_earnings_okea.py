from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

TICKER = "OKEA.OL"
COMPANY = "OKEA"
DAYS_AFTER = 50

stock = yf.Ticker(TICKER)
earnings_df = stock.earnings_dates
past_earnings = earnings_df[earnings_df["Reported EPS"].notna()].copy()
past_earnings = past_earnings.sort_index(ascending=False)
report_dates = sorted(past_earnings.head(6).index.tolist())

earliest = min(report_dates) - timedelta(days=5)
latest = max(report_dates) + timedelta(days=DAYS_AFTER + 10)
hist = stock.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
hist.index = hist.index.tz_localize(None)

fig, axes = plt.subplots(2, 1, figsize=(15, 11),
                         gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.35})
ax_price, ax_reaction = axes
fig.suptitle(f"{COMPANY} ({TICKER}) — Totalavkastning 50 handelsdager etter kvartalsrapport\n(ingen utbytter utbetalt i perioden)",
             fontsize=13, fontweight="bold", y=0.99)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
reaction_data = []

for i, report_date in enumerate(report_dates):
    rd_naive = report_date.tz_localize(None)
    date_key = report_date.strftime("%Y-%m-%d")

    mask = hist.index >= rd_naive
    post_data = hist.loc[mask].head(DAYS_AFTER + 1)

    if len(post_data) < 10:
        print(f"  Ikke nok data etter {date_key}, hopper over.")
        continue

    base_price = post_data["Close"].iloc[0]
    cum_div = post_data["Dividends"].cumsum()
    total_return = ((post_data["Close"] + cum_div - base_price) / base_price) * 100
    trading_days = list(range(len(total_return)))

    pre_mask = hist.index < rd_naive
    if pre_mask.any():
        prev_close = hist.loc[pre_mask, "Close"].iloc[-1]
        day1_reaction = ((post_data["Close"].iloc[0] / prev_close) - 1) * 100
    else:
        day1_reaction = 0

    reaction_data.append((date_key, day1_reaction, total_return.iloc[-1]))

    label = f"{date_key} — dag 1: {day1_reaction:+.1f}%"
    ax_price.plot(trading_days, total_return.values, label=label, color=colors[i],
                  linewidth=2, alpha=0.85)

ax_price.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax_price.set_xlabel("Handelsdager etter rapport", fontsize=11)
ax_price.set_ylabel("Totalavkastning (%)", fontsize=11)
ax_price.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax_price.grid(True, alpha=0.3)
ax_price.set_xlim(0, DAYS_AFTER)

if reaction_data:
    labels = [r[0] for r in reaction_data]
    day1 = [r[1] for r in reaction_data]
    drift_50d = [r[2] for r in reaction_data]

    x_pos = np.arange(len(labels))
    w = 0.35
    bar_colors_d1 = ["#2ecc71" if d > 0 else "#e74c3c" for d in day1]
    bar_colors_50 = ["#27ae60" if d > 0 else "#c0392b" for d in drift_50d]

    ax_reaction.bar(x_pos - w/2, day1, w, color=bar_colors_d1, alpha=0.7,
                    label="Dag 1 reaksjon (%)", edgecolor="white")
    ax_reaction.bar(x_pos + w/2, drift_50d, w, color=bar_colors_50, alpha=0.5,
                    label="Drift etter 50d (%)", edgecolor="white")
    ax_reaction.axhline(y=0, color="black", linewidth=0.5)
    ax_reaction.set_xticks(x_pos)
    ax_reaction.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_reaction.set_ylabel("Kursendring (%)", fontsize=10)
    ax_reaction.legend(loc="best", fontsize=9)
    ax_reaction.grid(True, alpha=0.3)
    ax_reaction.set_title("Markedets umiddelbare reaksjon vs. 50-dagers drift", fontsize=11)

plt.subplots_adjust(top=0.92, bottom=0.08)
output_path = str(Path(__file__).resolve().parents[2] / "data" / "okea_post_earnings.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nGraf lagret: {output_path}")
plt.show()
