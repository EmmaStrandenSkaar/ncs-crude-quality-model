"""
Script 57 — Demand → Stocks elastisitet (Stage 1.5)

Modellerer hvordan lagre (inventories) responderer på endringer i produkt-demand.
Dette er nøkkelen til realistiske demand-scenarier:

  Når jet-demand stiger 5% → jet stocks faller med ca. X mbbl
  → Stage 1 oversetter dette til høyere jet crack
  → Stage 2 oversetter til høyere grade-pris for crudes med høy kerosene-yield

EMPIRISK MODELL:
  stocks_t = α + β1·demand_t + β2·lag(stocks_t) + β3·sin_month + β4·cos_month + ε

  β1 = "Demand-elastisitet" — hvor mye stocks endres per kbpd demand-endring

OUTPUT:
  data/processed/57_demand_stocks_elasticity.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DEMAND_CSV   = PROC_DIR / "53_product_demand.csv"
OUT_JSON     = PROC_DIR / "57_demand_stocks_elasticity.json"


PRODUCTS = {
    "motor_gasoline":   ("motor_gasoline_kbpd",  "motor_gasoline_stocks_mbbl"),
    "jet_fuel":         ("jet_fuel_kbpd",        "jet_fuel_stocks_mbbl"),
    "distillate_fuel":  ("distillate_fuel_kbpd", "distillate_stocks_mbbl"),
    "residual_fuel":    ("residual_fuel_kbpd",   "residual_stocks_mbbl"),
}


def fit_stocks_model(df: pd.DataFrame, demand_col: str, stocks_col: str) -> dict:
    """Regress stocks på demand + lag + sesong."""
    df = df.sort_values("date").copy()
    df["sin_month"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["stocks_lag1"] = df[stocks_col].shift(1)

    sub = df.dropna(subset=[demand_col, stocks_col, "stocks_lag1"])
    if len(sub) < 12:
        return {"error": f"For lite data: {len(sub)} obs"}

    X_cols = [demand_col, "stocks_lag1", "sin_month", "cos_month"]
    X = sm.add_constant(sub[X_cols].astype(float))
    y = sub[stocks_col].astype(float)
    m = sm.OLS(y, X).fit(cov_type="HC1")

    # Demand-elastisitet i nivå-termer (mbbl per kbpd)
    demand_coef = float(m.params[demand_col])
    demand_pval = float(m.pvalues[demand_col])

    # Long-run elasticity (i likevekt med stocks_lag1 ≈ stocks_t)
    lag_coef = float(m.params["stocks_lag1"])
    long_run_demand_effect = demand_coef / (1 - lag_coef) if abs(1 - lag_coef) > 0.01 else demand_coef

    # Praktisk: hvis demand stiger 200 kbpd, hvor mye endres stocks?
    short_run_change = demand_coef * 200
    long_run_change  = long_run_demand_effect * 200

    return {
        "n":                  len(sub),
        "r2":                 float(m.rsquared),
        "demand_coef":        round(demand_coef, 6),
        "demand_pvalue":      round(demand_pval, 4),
        "lag_coef":           round(lag_coef, 4),
        "long_run_elasticity":round(long_run_demand_effect, 6),
        "short_run_change_per_200kbpd": round(short_run_change, 2),
        "long_run_change_per_200kbpd":  round(long_run_change, 2),
        "demand_mean":        round(float(sub[demand_col].mean()), 1),
        "stocks_mean":        round(float(sub[stocks_col].mean()), 1),
    }


def main():
    print("=" * 70)
    print("  SCRIPT 57: Demand → Stocks elastisitet")
    print("=" * 70)

    df = pd.read_csv(DEMAND_CSV)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n  Data: {len(df)} mnd, "
          f"{df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}")

    print(f"\n[1] Tren modell per produkt: stocks_t = α + β·demand_t + γ·lag(stocks) + sesong\n")
    results = {}
    print(f"  {'Produkt':<20} {'N':>4} {'R²':>6} "
          f"{'β_demand':>10} {'p':>7} {'lag':>6} "
          f"{'Δstocks per +200kbpd demand':>30}")
    print(f"  {'-'*100}")

    for prod, (demand_col, stocks_col) in PRODUCTS.items():
        if demand_col not in df.columns or stocks_col not in df.columns:
            print(f"  ⚠ {prod}: mangler kolonner")
            continue
        res = fit_stocks_model(df, demand_col, stocks_col)
        if "error" in res:
            print(f"  ⚠ {prod}: {res['error']}")
            continue
        sig = "***" if res["demand_pvalue"] < 0.001 else \
              "**"  if res["demand_pvalue"] < 0.01  else \
              "*"   if res["demand_pvalue"] < 0.05  else \
              "."   if res["demand_pvalue"] < 0.10  else ""
        sr = res["short_run_change_per_200kbpd"]
        lr = res["long_run_change_per_200kbpd"]
        print(f"  {prod:<20} {res['n']:>4} {res['r2']:>6.3f} "
              f"{res['demand_coef']:>+10.5f} {res['demand_pvalue']:>6.3f}{sig:<3} "
              f"{res['lag_coef']:>6.3f} "
              f"SR={sr:>+6.2f}  LR={lr:>+6.2f} mbbl")
        results[prod] = res

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  ✓ Lagret: {OUT_JSON.name}")

    # ── Tolkning ────────────────────────────────────────────────────────────
    print(f"\n[2] TOLKNING")
    print(f"{'─' * 70}")
    print(f"  Negativ β_demand betyr: høyere demand → lavere stocks (forventet)")
    print(f"  Lag-koeffisient nær 1 = stocks justerer seg sakte (sterk persistens)")
    print(f"\n  PRAKTISK BRUK i scenario-analyse:")
    print(f"    Når demand +X kbpd, anslå stocks-endring som:")
    print(f"    Δstocks = β_demand × X  (kortsiktig)")
    print(f"    eller β_demand/(1-lag) × X  (langsiktig likevekt)")

    print(f"\n  Eksempel — jet fuel:")
    if "jet_fuel" in results:
        r = results["jet_fuel"]
        print(f"    +200 kbpd jet demand → {r['short_run_change_per_200kbpd']:+.1f} mbbl stocks (kort sikt)")
        print(f"                            {r['long_run_change_per_200kbpd']:+.1f} mbbl stocks (langsiktig)")
        if r['long_run_change_per_200kbpd'] < 0:
            print(f"    Med stage-1 koeffisient -2.22 USD/bbl per mbbl → "
                  f"jet crack stiger {-2.22 * r['long_run_change_per_200kbpd']:+.2f} USD/bbl")
            print(f"    Crudes med høy kerosene-yield får da premium-løft.")


if __name__ == "__main__":
    main()
