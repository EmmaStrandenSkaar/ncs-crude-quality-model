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

earliest = min(report_dates) - timedelta(days=5)
latest = max(report_dates) + timedelta(days=DAYS_AFTER + 10)
hist = stock.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
hist.index = hist.index.tz_localize(None)

# Hent Brent-data for samme periode
brent = yf.Ticker(BRENT_TICKER)
brent_hist = brent.history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"))
brent_hist.index = brent_hist.index.tz_localize(None)

fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                         gridspec_kw={"height_ratios": [3, 1.3, 1.3], "hspace": 0.4})
ax_price, ax_reaction, ax_eps = axes
fig.suptitle(f"{COMPANY} ({TICKER}) — Meravkastning vs. Brent, {DAYS_AFTER} handelsdager etter kvartalsrapport\n(justert for utbytte og oljeprisendring)",
             fontsize=13, fontweight="bold", y=0.99)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
reaction_data = []

for i, report_date in enumerate(report_dates):
    rd_naive = report_date.tz_localize(None)
    date_key = report_date.strftime("%Y-%m-%d")
    fund = fundamentals.get(date_key, {})
    quarter = fund.get("quarter", date_key)

    # OTL-data
    mask = hist.index >= rd_naive
    post_data = hist.loc[mask].head(DAYS_AFTER + 1)

    if len(post_data) < 5:
        print(f"  Ikke nok data etter {date_key}, hopper over.")
        continue

    base_price = post_data["Close"].iloc[0]
    cum_div = post_data["Dividends"].cumsum()
    total_return = ((post_data["Close"] + cum_div - base_price) / base_price) * 100

    # Brent-data for same window
    brent_mask = brent_hist.index >= rd_naive
    brent_post = brent_hist.loc[brent_mask].head(DAYS_AFTER + 1)

    if len(brent_post) >= 2:
        brent_base = brent_post["Close"].iloc[0]
        brent_return = ((brent_post["Close"] / brent_base) - 1) * 100
        # Reindex brent til trading days
        brent_days = list(range(len(brent_return)))
    else:
        brent_return = pd.Series([0] * len(total_return))
        brent_days = list(range(len(total_return)))

    # Meravkastning = OTL total return - Brent return (matched by index)
    min_len = min(len(total_return), len(brent_return))
    excess_return = total_return.values[:min_len] - brent_return.values[:min_len]
    trading_days = list(range(min_len))

    div_sum = cum_div.iloc[-1] if len(cum_div) > 0 else 0

    # Dag 1 reaksjon
    pre_mask = hist.index < rd_naive
    if pre_mask.any():
        prev_close = hist.loc[pre_mask, "Close"].iloc[-1]
        day1_reaction = ((post_data["Close"].iloc[0] / prev_close) - 1) * 100
    else:
        day1_reaction = 0

    # Brent dag 1
    brent_pre = brent_hist.loc[brent_hist.index < rd_naive]
    if len(brent_pre) > 0 and len(brent_post) > 0:
        brent_d1 = ((brent_post["Close"].iloc[0] / brent_pre["Close"].iloc[-1]) - 1) * 100
    else:
        brent_d1 = 0

    intraday = ((post_data["Close"].iloc[0] - post_data["Open"].iloc[0]) / post_data["Open"].iloc[0]) * 100

    otl_30d = total_return.iloc[min_len - 1]
    brent_30d = brent_return.iloc[min_len - 1] if min_len <= len(brent_return) else 0
    excess_30d = otl_30d - brent_30d

    reaction_data.append((quarter, day1_reaction, brent_d1, otl_30d, brent_30d, excess_30d))

    label = f"{quarter} — OTL: {otl_30d:+.1f}%, Brent: {brent_30d:+.1f}%, Alpha: {excess_30d:+.1f}%"

    # Plot alle tre linjer
    ax_price.plot(trading_days, total_return.values[:min_len], color=colors[i],
                  linewidth=2, alpha=0.85, label=label)
    ax_price.plot(brent_days[:min_len], brent_return.values[:min_len], color=colors[i],
                  linewidth=1, alpha=0.3, linestyle="--")

# --- Panel 1: Kursutvikling vs Brent ---
ax_price.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax_price.set_xlabel(f"Handelsdager etter rapport", fontsize=11)
ax_price.set_ylabel("Avkastning (%)", fontsize=11)
ax_price.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
ax_price.grid(True, alpha=0.3)
ax_price.set_xlim(0, DAYS_AFTER)
ax_price.text(0.98, 0.02, "Heltrukken = OTL totalavk., stiplet = Brent",
              transform=ax_price.transAxes, fontsize=8, ha="right", va="bottom",
              style="italic", alpha=0.7)

# --- Panel 2: OTL vs Brent 30d + meravkastning ---
if reaction_data:
    quarters = [r[0] for r in reaction_data]
    otl_30 = [r[3] for r in reaction_data]
    brent_30 = [r[4] for r in reaction_data]
    alpha_30 = [r[5] for r in reaction_data]

    x_pos = np.arange(len(quarters))
    w = 0.25

    ax_reaction.bar(x_pos - w, otl_30, w, color="#3498db", alpha=0.7,
                    label="OTL totalavk. 30d (%)", edgecolor="white")
    ax_reaction.bar(x_pos, brent_30, w, color="#95a5a6", alpha=0.7,
                    label="Brent 30d (%)", edgecolor="white")
    alpha_colors = ["#2ecc71" if a > 0 else "#e74c3c" for a in alpha_30]
    ax_reaction.bar(x_pos + w, alpha_30, w, color=alpha_colors, alpha=0.8,
                    label="Meravkastning (alpha) (%)", edgecolor="white")

    ax_reaction.axhline(y=0, color="black", linewidth=0.5)
    ax_reaction.set_xticks(x_pos)
    ax_reaction.set_xticklabels(quarters, rotation=45, ha="right", fontsize=9)
    ax_reaction.set_ylabel("Avkastning (%)", fontsize=10)
    ax_reaction.legend(loc="best", fontsize=8)
    ax_reaction.grid(True, alpha=0.3)
    ax_reaction.set_title("OTL totalavkastning vs. Brent-endring over 30 dager", fontsize=11)

# --- Panel 3: EPS surprise (Yahoo Finance) ---
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
output_path = str(Path(__file__).resolve().parents[2] / "data" / "otl_post_earnings.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Graf lagret: {output_path}")
plt.show()
