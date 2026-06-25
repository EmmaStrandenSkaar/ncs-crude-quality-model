"""
ma_window_monitor.py
Daglig M&A-vindu monitor for Aker BP.

Henter fersk data, beregner vindu-parametre, og returnerer
en strukturert statusrapport.

Brukes av scheduled task — output er ren tekst.
"""

from pathlib import Path
from datetime import datetime
import sys

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"FEIL: Mangler pakke — {e}")
    print("Kjor: pip install yfinance pandas numpy")
    sys.exit(1)

# ── Historiske referansepunkter ─────────────────────────────────────────────
HIST_DEALS = [
    dict(name="BP Norge",      window_days=66,  rally_pct=59, year=2016),
    dict(name="Lundin Energy", window_days=123, rally_pct=41, year=2021),
    dict(name="Total E&P",     window_days=134, rally_pct=48, year=2018),
    dict(name="King Lear",     window_days=173, rally_pct=21, year=2018),
    dict(name="Hess Norge",    window_days=255, rally_pct=27, year=2017),
]
AVG_WINDOW_DAYS = int(np.mean([d["window_days"] for d in HIST_DEALS]))
AVG_RALLY = np.mean([d["rally_pct"] for d in HIST_DEALS])

# Aker BP EV/2P og kandidater
AKRBP_EV2P = 19.7
CANDIDATES = [
    dict(name="OKEA ASA",       ev2p=4.3,  prob="HØY"),
    dict(name="INPEX Valhall",  ev2p=11.4, prob="HØY"),
    dict(name="Harbour NCS",    ev2p=5.8,  prob="MIDDELS"),
    dict(name="Lime Petroleum", ev2p=7.5,  prob="MIDDELS"),
    dict(name="Vår Energi",     ev2p=13.6, prob="LAV"),
    dict(name="DNO NCS",        ev2p=13.2, prob="LAV"),
]


def find_cycle_trough(close, lookback_days=400):
    """Finn siste signifikante bunn."""
    prices = close.values
    dates = close.index

    current_price = prices[-1]

    for lookback in [60, 90, 120, 180, 250, 350]:
        if len(prices) < lookback:
            continue
        window_prices = prices[-lookback:]
        window_dates = dates[-lookback:]
        min_idx = np.argmin(window_prices)
        min_price = window_prices[min_idx]
        min_date = window_dates[min_idx]

        rally = (current_price / min_price - 1)
        if rally > 0.15:
            return min_date, float(min_price)

    min_idx = np.argmin(prices[-250:])
    return dates[-250:][min_idx], float(prices[-250:][min_idx])


def get_usdnok():
    """Hent siste USDNOK."""
    try:
        nok = yf.download("NOK=X", period="5d", progress=False, auto_adjust=True)
        if isinstance(nok.columns, pd.MultiIndex):
            nok = nok.droplevel(level=1, axis=1)
        return float(nok["Close"].iloc[-1])
    except:
        return None


def run_monitor():
    """Kjør daglig sjekk og returner rapport som tekst."""

    # ── Hent data ──────────────────────────────────────────────────────────
    akrbp = yf.download("AKRBP.OL", start="2025-01-01", progress=False, auto_adjust=True)
    if isinstance(akrbp.columns, pd.MultiIndex):
        akrbp = akrbp.droplevel(level=1, axis=1)
    close = akrbp["Close"].dropna()
    close.index = pd.to_datetime(close.index)

    brent = yf.download("BZ=F", period="5d", progress=False, auto_adjust=True)
    if isinstance(brent.columns, pd.MultiIndex):
        brent = brent.droplevel(level=1, axis=1)
    brent_price = float(brent["Close"].iloc[-1]) if len(brent) > 0 else None

    usdnok = get_usdnok()

    # ── Beregn vindu-parametre ─────────────────────────────────────────────
    today = close.index[-1]
    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
    daily_change = ((current_price / prev_price) - 1) * 100

    trough_date, trough_price = find_cycle_trough(close)
    window_days = (today - trough_date).days
    rally_pct = ((current_price / trough_price) - 1) * 100

    # 52-ukers høy/lav
    high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
    low_52w = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
    pct_from_high = ((current_price / high_52w) - 1) * 100

    # ── Vindu-status ───────────────────────────────────────────────────────
    days_to_avg = AVG_WINDOW_DAYS - window_days
    pct_of_avg_window = (window_days / AVG_WINDOW_DAYS) * 100

    if window_days < 60:
        window_phase = "TIDLIG — vinduet har nettopp aapnet"
    elif window_days < AVG_WINDOW_DAYS:
        window_phase = f"AKTIV — {days_to_avg} dager til historisk snitt"
    elif window_days < AVG_WINDOW_DAYS + 60:
        window_phase = "MODEN — rundt historisk snitttidspunkt for deal"
    else:
        window_phase = "SEN — vinduet er eldre enn de fleste historiske deals"

    # Sammenlign med historiske deals
    deals_before = [d for d in HIST_DEALS if d["window_days"] <= window_days]
    deals_after = [d for d in HIST_DEALS if d["window_days"] > window_days]

    # ── Estimert EV/2P (dynamisk) ──────────────────────────────────────────
    # Forenklet: EV = mcap + netto gjeld (~$7bn)
    shares_outstanding = 632  # millioner
    if usdnok:
        mcap_usd = (current_price * shares_outstanding) / usdnok / 1000  # mrd USD
        ev_usd = mcap_usd + 7.0  # netto gjeld ~$7bn
        reserves_2p = 1526  # mmboe
        current_ev2p = ev_usd * 1000 / reserves_2p  # $/boe
    else:
        current_ev2p = AKRBP_EV2P  # fallback

    # ── Bygg rapport ───────────────────────────────────────────────────────
    report = []
    report.append("=" * 60)
    report.append("  AKER BP M&A-VINDU — DAGLIG STATUSRAPPORT")
    report.append(f"  {today.strftime('%A %d. %B %Y')}")
    report.append("=" * 60)

    report.append("")
    report.append("MARKEDSDATA:")
    report.append(f"  AKRBP.OL:  {current_price:.1f} NOK ({daily_change:+.1f}% i dag)")
    report.append(f"  52u høy/lav: {high_52w:.0f} / {low_52w:.0f} NOK ({pct_from_high:+.1f}% fra topp)")
    if brent_price:
        report.append(f"  Brent:     ${brent_price:.1f}/fat")
    if usdnok:
        report.append(f"  USD/NOK:   {usdnok:.2f}")
    report.append(f"  Est. EV/2P: ${current_ev2p:.1f}/boe")

    report.append("")
    report.append("M&A-VINDU STATUS:")
    report.append(f"  Syklusbunn:    {trough_date.strftime('%d. %b %Y')} ({trough_price:.0f} NOK)")
    report.append(f"  Vinduets alder: {window_days} dager")
    report.append(f"  Rally fra bunn: +{rally_pct:.1f}%")
    report.append(f"  Fase:          {window_phase}")
    report.append(f"  vs historisk snitt: {pct_of_avg_window:.0f}% ({AVG_WINDOW_DAYS}d snitt)")

    report.append("")
    report.append("HISTORISK SAMMENLIGNING:")
    report.append(f"  Deals som kom TIDLIGERE enn dag {window_days}:")
    for d in deals_before:
        report.append(f"    - {d['name']} ({d['year']}): dag {d['window_days']}, +{d['rally_pct']}% rally")
    if not deals_before:
        report.append("    (ingen — vi er fortsatt tidlig)")
    report.append(f"  Deals som kom SENERE:")
    for d in deals_after:
        report.append(f"    - {d['name']} ({d['year']}): dag {d['window_days']}, +{d['rally_pct']}% rally")

    report.append("")
    report.append("AKKRESJONSTABELL (maal vs AKRBP):")
    report.append(f"  {'Kandidat':<20} {'EV/2P':>8} {'Rabatt':>8} {'Rabatt%':>8} {'Prob':>8}")
    report.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for c in CANDIDATES:
        rabatt = current_ev2p - c["ev2p"]
        rabatt_pct = (rabatt / current_ev2p) * 100
        report.append(f"  {c['name']:<20} ${c['ev2p']:>6.1f} ${rabatt:>6.1f} {rabatt_pct:>6.0f}%  {c['prob']:>6}")

    # ── Signalvurdering ────────────────────────────────────────────────────
    report.append("")
    report.append("SIGNALER:")

    signals = []
    if rally_pct > 40:
        signals.append("[STERKT] Rally >40% — historisk «sweet spot» for deals")
    elif rally_pct > 25:
        signals.append("[MODERAT] Rally 25-40% — vinduet bygger seg opp")
    else:
        signals.append("[SVAKT] Rally <25% — vinduet er fortsatt ungt")

    if window_days > AVG_WINDOW_DAYS - 30:
        signals.append("[STERKT] Naermer seg historisk snitttidspunkt for deal")
    if window_days > 200:
        signals.append("[ADVARSEL] Vinduet er eldre enn 4 av 5 historiske deals")

    if current_ev2p > 18:
        signals.append(f"[STERKT] EV/2P ${current_ev2p:.1f} — godt over kandidatene")
    elif current_ev2p > 15:
        signals.append(f"[MODERAT] EV/2P ${current_ev2p:.1f} — deals er akkretive")

    if brent_price and brent_price > 90:
        signals.append(f"[POSITIVT] Brent ${brent_price:.0f} — støtter høy aksje + kontantstrøm")
    elif brent_price and brent_price < 65:
        signals.append(f"[NEGATIVT] Brent ${brent_price:.0f} — presser aksje og vindu")

    if pct_from_high > -5:
        signals.append("[STERKT] Aksje nær 52-ukers topp — sterk oppkjøpsvaluta")
    elif pct_from_high < -15:
        signals.append("[SVAKT] Aksje >15% under topp — vinduet kan lukkes")

    for s in signals:
        report.append(f"  {s}")

    # Samlet score
    strong = sum(1 for s in signals if "[STERKT]" in s)
    moderate = sum(1 for s in signals if "[MODERAT]" in s or "[POSITIVT]" in s)
    weak = sum(1 for s in signals if "[SVAKT]" in s or "[NEGATIVT]" in s or "[ADVARSEL]" in s)

    report.append("")
    if strong >= 2 and weak == 0:
        report.append("  SAMLET: 🟢 M&A-vinduet er STERKT AAPENT")
    elif strong >= 1 and weak <= 1:
        report.append("  SAMLET: 🟡 M&A-vinduet er AAPENT")
    else:
        report.append("  SAMLET: 🔴 M&A-vinduet er UNDER PRESS")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    print(run_monitor())
