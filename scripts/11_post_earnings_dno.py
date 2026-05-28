import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

TICKER = "DNO.OL"
COMPANY = "DNO"
DAYS_AFTER = 50

# Konsensus fra Bloomberg / Investing.com (manuelt verifisert)
# Tall i millioner USD
consensus_data = {
    "2024-11-07": {  # Q3 2024
        "quarter": "Q3 2024",
        "rev_actual": 170.5, "rev_cons": None,
        "ebitda_actual": None, "ebitda_cons": None,
    },
    "2025-02-06": {  # Q4 2024
        "quarter": "Q4 2024",
        "rev_actual": 176.6, "rev_cons": 183.5,
        "ebitda_actual": 71.4, "ebitda_cons": 103.9,
    },
    "2025-05-15": {  # Q1 2025
        "quarter": "Q1 2025",
        "rev_actual": 188, "rev_cons": None,
        "ebitda_actual": None, "ebitda_cons": None,
    },
    "2025-08-21": {  # Q2 2025
        "quarter": "Q2 2025",
        "rev_actual": 258, "rev_cons": None,
        "ebitda_actual": None, "ebitda_cons": None,
    },
    "2025-11-06": {  # Q3 2025
        "quarter": "Q3 2025",
        "rev_actual": 547, "rev_cons": None,
        "ebitda_actual": None, "ebitda_cons": None,
    },
    "2026-02-05": {  # Q4 2025
        "quarter": "Q4 2025",
        "rev_actual": 482, "rev_cons": 567,
        "ebitda_actual": 254, "ebitda_cons": 358,
    },
}

stock = yf.Ticker(TICKER)
earnings_df = stock.earnings_dates
past_earnings = earnings_df[earnings_df["Reported EPS"].notna()].copy()
past_earnings = past_earnings.sort_index(ascending=False)
report_dates = sorted(past_earnings.head(6).index.tolist())

earliest = min(report_dates) - timedelta(days=5)
latest = max(report_dates) + timedelta(days=DAYS_AFTER + 10)
hist = stock.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
hist.index = hist.index.tz_localize(None)

fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                         gridspec_kw={"height_ratios": [3, 1.3, 1.3], "hspace": 0.4})
ax_price, ax_reaction, ax_cons = axes
fig.suptitle(f"{COMPANY} ({TICKER}) — Totalavkastning 50 handelsdager etter kvartalsrapport\n(justert for utbytte)",
             fontsize=13, fontweight="bold", y=0.99)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
reaction_data = []

for i, report_date in enumerate(report_dates):
    rd_naive = report_date.tz_localize(None)
    date_key = report_date.strftime("%Y-%m-%d")
    cons = consensus_data.get(date_key, {})
    quarter = cons.get("quarter", date_key)

    mask = hist.index >= rd_naive
    post_data = hist.loc[mask].head(DAYS_AFTER + 1)

    if len(post_data) < 10:
        continue

    base_price = post_data["Close"].iloc[0]
    cum_div = post_data["Dividends"].cumsum()
    total_return = ((post_data["Close"] + cum_div - base_price) / base_price) * 100
    trading_days = list(range(len(total_return)))

    div_sum = cum_div.iloc[-1] if len(cum_div) > 0 else 0

    pre_mask = hist.index < rd_naive
    if pre_mask.any():
        prev_close = hist.loc[pre_mask, "Close"].iloc[-1]
        day1_reaction = ((post_data["Close"].iloc[0] / prev_close) - 1) * 100
    else:
        day1_reaction = 0

    reaction_data.append((quarter, day1_reaction, total_return.iloc[-1]))

    div_label = f" (div: {div_sum:.2f} NOK)" if div_sum > 0 else ""
    label = f"{quarter} — dag 1: {day1_reaction:+.1f}%{div_label}"
    ax_price.plot(trading_days, total_return.values, label=label, color=colors[i],
                  linewidth=2, alpha=0.85)

# --- Panel 1: Kursutvikling ---
ax_price.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax_price.set_xlabel("Handelsdager etter rapport", fontsize=11)
ax_price.set_ylabel("Totalavkastning (%)", fontsize=11)
ax_price.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax_price.grid(True, alpha=0.3)
ax_price.set_xlim(0, DAYS_AFTER)

# --- Panel 2: Markedets reaksjon dag 1 vs drift etter 50 dager ---
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

# --- Panel 3: EBITDA konsensus vs faktisk ---
ebitda_quarters = []
ebitda_actual = []
ebitda_cons = []

for rd in report_dates:
    date_key = rd.strftime("%Y-%m-%d")
    cons = consensus_data.get(date_key, {})
    ea = cons.get("ebitda_actual")
    ec = cons.get("ebitda_cons")
    q = cons.get("quarter", date_key)
    if ea is not None and ec is not None:
        ebitda_quarters.append(q)
        ebitda_actual.append(ea)
        ebitda_cons.append(ec)

if ebitda_quarters:
    x_pos = np.arange(len(ebitda_quarters))
    w = 0.35
    ax_cons.bar(x_pos - w/2, ebitda_actual, w, color="#3498db", alpha=0.8,
                label="Faktisk EBITDA (mUSD)")
    ax_cons.bar(x_pos + w/2, ebitda_cons, w, color="#95a5a6", alpha=0.6,
                label="Konsensus (mUSD)")
    for j, ec in enumerate(ebitda_cons):
        diff_pct = ((ebitda_actual[j] - ec) / ec) * 100
        ax_cons.annotate(f"{diff_pct:+.1f}%",
                        xy=(x_pos[j], max(ebitda_actual[j], ec) + 5),
                        ha="center", fontsize=10, fontweight="bold",
                        color="#2ecc71" if diff_pct > 0 else "#e74c3c")

    ax_cons.set_xticks(x_pos)
    ax_cons.set_xticklabels(ebitda_quarters, rotation=45, ha="right", fontsize=9)
    ax_cons.set_ylabel("EBITDA (mUSD)", fontsize=10)
    ax_cons.legend(loc="best", fontsize=9)
    ax_cons.grid(True, alpha=0.3, axis="y")
    ax_cons.set_title("EBITDA: Faktisk vs. konsensusestimat (kilde: Bloomberg / Investing.com)", fontsize=11)

plt.subplots_adjust(top=0.94, bottom=0.06)
output_path = "/Users/emmastrandenskaar/Documents/Claude/Projects/Oljepris/data/dno_post_earnings.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Graf lagret: {output_path}")
