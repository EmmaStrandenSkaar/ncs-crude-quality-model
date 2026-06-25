from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

TICKER = "AKRBP.OL"
COMPANY = "Aker BP"
DAYS_AFTER = 50

# Konsensus fra Investing.com / Visible Alpha / RBC (manuelt verifisert)
# Format: rapportdato -> {kvartal, revenue_actual, revenue_cons, ebitda_actual, ebitda_cons}
# Alle tall i milliarder USD
consensus_data = {
    "2025-05-06": {  # Q1 2025
        "quarter": "Q1 2025",
        "rev_actual": 3.20, "rev_cons": 3.09,
        "ebitda_actual": 2.80, "ebitda_cons": 2.72,
        "net_income_actual": 316, "net_income_cons": 470,  # millioner USD
    },
    "2025-07-15": {  # Q2 2025
        "quarter": "Q2 2025",
        "rev_actual": 2.58, "rev_cons": 2.62,
        "ebitda_actual": 2.22, "ebitda_cons": 2.27,
        "net_income_actual": -324, "net_income_cons": None,
    },
    "2025-10-22": {  # Q3 2025
        "quarter": "Q3 2025",
        "rev_actual": 2.60, "rev_cons": 2.52,
        "ebitda_actual": 2.26, "ebitda_cons": None,
        "net_income_actual": 285, "net_income_cons": None,
    },
    "2026-02-10": {  # Q4 2025
        "quarter": "Q4 2025",
        "rev_actual": 2.56, "rev_cons": None,
        "ebitda_actual": 2.07, "ebitda_cons": None,
        "net_income_actual": -145, "net_income_cons": None,
    },
    "2024-10-29": {  # Q3 2024
        "quarter": "Q3 2024",
        "rev_actual": None, "rev_cons": None,
        "ebitda_actual": None, "ebitda_cons": None,
        "net_income_actual": None, "net_income_cons": None,
    },
    "2025-02-11": {  # Q4 2024
        "quarter": "Q4 2024",
        "rev_actual": 3.10, "rev_cons": None,
        "ebitda_actual": 2.70, "ebitda_cons": None,
        "net_income_actual": 562, "net_income_cons": None,
    },
}

stock = yf.Ticker(TICKER)
earnings_df = stock.earnings_dates
past_earnings = earnings_df[earnings_df["Reported EPS"].notna()].copy()
past_earnings = past_earnings.sort_index(ascending=False)
report_dates = past_earnings.head(6).index.tolist()
report_dates = sorted(report_dates)

earliest = min(report_dates) - timedelta(days=5)
latest = max(report_dates) + timedelta(days=DAYS_AFTER + 10)
hist = stock.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
hist.index = hist.index.tz_localize(None)

fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                         gridspec_kw={"height_ratios": [3, 1.3, 1.3], "hspace": 0.4})
ax_price, ax_reaction, ax_rev = axes
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
        print(f"  Ikke nok data etter {date_key}, hopper over.")
        continue

    base_price = post_data["Close"].iloc[0]

    # Totalavkastning: kursendring + kumulativt utbytte
    cum_div = post_data["Dividends"].cumsum()
    total_return = ((post_data["Close"] + cum_div - base_price) / base_price) * 100
    price_only = ((post_data["Close"] / base_price) - 1) * 100
    trading_days = list(range(len(total_return)))

    div_sum = cum_div.iloc[-1] if len(cum_div) > 0 else 0

    pre_mask = hist.index < rd_naive
    if pre_mask.any():
        prev_close = hist.loc[pre_mask, "Close"].iloc[-1]
        day1_reaction = ((post_data["Close"].iloc[0] / prev_close) - 1) * 100
    else:
        day1_reaction = 0

    reaction_data.append((quarter, day1_reaction, total_return.iloc[-1]))

    div_label = f" (div: {div_sum:.1f} NOK)" if div_sum > 0 else ""
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
    quarters = [r[0] for r in reaction_data]
    day1 = [r[1] for r in reaction_data]
    drift_50d = [r[2] for r in reaction_data]

    x_pos = np.arange(len(quarters))
    w = 0.35
    bar_colors_d1 = ["#2ecc71" if d > 0 else "#e74c3c" for d in day1]
    bar_colors_50 = ["#27ae60" if d > 0 else "#c0392b" for d in drift_50d]

    ax_reaction.bar(x_pos - w/2, day1, w, color=bar_colors_d1, alpha=0.7,
                    label="Dag 1 reaksjon (%)", edgecolor="white")
    ax_reaction.bar(x_pos + w/2, drift_50d, w, color=bar_colors_50, alpha=0.5,
                    label="Drift etter 50d (%)", edgecolor="white")
    ax_reaction.axhline(y=0, color="black", linewidth=0.5)
    ax_reaction.set_xticks(x_pos)
    ax_reaction.set_xticklabels(quarters, rotation=45, ha="right", fontsize=9)
    ax_reaction.set_ylabel("Kursendring (%)", fontsize=10)
    ax_reaction.legend(loc="best", fontsize=9)
    ax_reaction.grid(True, alpha=0.3)
    ax_reaction.set_title("Markedets umiddelbare reaksjon vs. 50-dagers drift", fontsize=11)

# --- Panel 3: Revenue konsensus vs faktisk ---
rev_quarters = []
rev_actual = []
rev_cons = []

for rd in report_dates:
    date_key = rd.strftime("%Y-%m-%d")
    cons = consensus_data.get(date_key, {})
    ra = cons.get("rev_actual")
    rc = cons.get("rev_cons")
    q = cons.get("quarter", date_key)
    if ra is not None:
        rev_quarters.append(q)
        rev_actual.append(ra)
        rev_cons.append(rc)

if rev_quarters:
    x_pos = np.arange(len(rev_quarters))
    w = 0.35
    bars_actual = ax_rev.bar(x_pos - w/2, rev_actual, w, color="#3498db", alpha=0.8,
                              label="Faktisk omsetning (mrd USD)")
    for j, rc in enumerate(rev_cons):
        if rc is not None:
            ax_rev.bar(x_pos[j] + w/2, rc, w, color="#95a5a6", alpha=0.6,
                       label="Konsensus (mrd USD)" if j == 0 else "")
            diff_pct = ((rev_actual[j] - rc) / rc) * 100
            ax_rev.annotate(f"{diff_pct:+.1f}%",
                           xy=(x_pos[j], max(rev_actual[j], rc) + 0.05),
                           ha="center", fontsize=9, fontweight="bold",
                           color="#2ecc71" if diff_pct > 0 else "#e74c3c")

    ax_rev.set_xticks(x_pos)
    ax_rev.set_xticklabels(rev_quarters, rotation=45, ha="right", fontsize=9)
    ax_rev.set_ylabel("Omsetning (mrd USD)", fontsize=10)
    ax_rev.legend(loc="best", fontsize=9)
    ax_rev.grid(True, alpha=0.3, axis="y")
    ax_rev.set_title("Omsetning: Faktisk vs. konsensusestimat (kilde: Visible Alpha / RBC)", fontsize=11)

plt.subplots_adjust(top=0.95, bottom=0.06)
output_path = str(Path(__file__).resolve().parents[2] / "data" / "aker_bp_post_earnings.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nGraf lagret: {output_path}")
plt.show()
