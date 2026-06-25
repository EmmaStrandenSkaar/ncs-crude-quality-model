"""
Script 54 — Stage 1: Demand → Crack-spread modell

Modellerer hvordan produkt-etterspørsel og fundamentaler driver crack-spreads.
Dette er Stage 1 i to-stegs-systemet:

   IEA/EIA demand forecasts → CRACK SPREADS → grade-pris (vår Modell B)
   ─────────────────────────  ─────────────  ────────────────────────
              ↑                  STAGE 1            STAGE 2
              vi har dette       NY MODELL          eksisterende

PER-PRODUKT MODELLER:
  gasoline_crack ~ gasoline_demand + gasoline_stocks + refinery_util
                 + seasonality + market regime dummies
  diesel_crack   ~ distillate_demand + distillate_stocks + refinery_util + ...
  jet_crack      ~ jet_demand + jet_stocks + refinery_util + ...

DEMAND-DATA:
  US product demand (STEO + EIA Weekly). US er ~20% av global demand og
  er sterkest signal for forward-leveranser. Globale STEO-data brukes som
  supplerende cross-check.

KAUSAL TOLKNING:
  Høyere demand → tightere produkt-balanse → høyere crack
  Høyere stocks → looser balanse → lavere crack
  Høyere refinery util → bedre tilbud → kan dempe crack (depends)

OUTPUT:
  data/processed/54_crack_spread_models.json   (3 modeller, en per produkt)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DEMAND_CSV   = PROC_DIR / "53_product_demand.csv"
PANEL_CSV    = PROC_DIR / "regression_panel.csv"
OUT_JSON     = PROC_DIR / "54_crack_spread_models.json"


PRODUCT_MODELS = {
    "gasoline_crack_brent": {
        "demand_col":    "motor_gasoline_kbpd",
        "stocks_col":    "motor_gasoline_stocks_mbbl",
        "label":         "Gasoline crack vs. Brent",
    },
    "diesel_crack_brent": {
        "demand_col":    "distillate_fuel_kbpd",
        "stocks_col":    "distillate_stocks_mbbl",
        "label":         "Diesel/distillate crack vs. Brent",
    },
    "jet_crack_brent": {
        "demand_col":    "jet_fuel_kbpd",
        "stocks_col":    "jet_fuel_stocks_mbbl",
        "label":         "Jet fuel crack vs. Brent",
    },
}

# Felles markedsfeatures som inkluderes i alle modellene
# NB: d_covid og d_iran_sanctions_v2 er essentielt konstante i 2022–2027-perioden
# (STEO-data starter 2022), så de droppes. d_russia_sanctions varierer (slått på feb 2022).
COMMON_FEATURES = [
    "us_refinery_util_pct",
    "brent_price",                # generelt nivå påvirker absolutt crack
    "sin_month", "cos_month",     # sesongmønster (jet ↑ sommer, diesel ↑ vinter)
    "d_russia_sanctions",         # slått på feb 2022 — varierer i sampel
]


def load_combined_panel() -> pd.DataFrame:
    """
    Slå sammen panel-features (crack spreads, refinery util osv.) med
    STEO produkt-demand på månedsnivå.
    """
    # Panel: aggreger til måned (én rad per måned — features er like for alle grades)
    panel = pd.read_csv(PANEL_CSV)
    panel["date"] = pd.to_datetime(panel["date_str"])
    panel["ym"]   = panel["date"].dt.to_period("M").dt.to_timestamp()

    feature_cols = (
        list(PRODUCT_MODELS.keys())     # crack spreads
        + COMMON_FEATURES               # markedsfeatures
    )
    panel_monthly = (panel.groupby("ym")[feature_cols]
                          .mean()
                          .reset_index()
                          .rename(columns={"ym": "date"}))

    # Demand-data
    demand = pd.read_csv(DEMAND_CSV)
    demand["date"] = pd.to_datetime(demand["date"])

    merged = panel_monthly.merge(demand, on="date", how="inner")
    print(f"  Merged panel: {len(merged)} mnd, "
          f"{merged['date'].min().strftime('%Y-%m')} → "
          f"{merged['date'].max().strftime('%Y-%m')}")
    return merged


def train_crack_model(df: pd.DataFrame, crack_col: str, cfg: dict) -> dict:
    """
    Tren OLS for ett crack-spread mot demand + stocks + felles features.
    """
    features = [cfg["demand_col"], cfg["stocks_col"]] + COMMON_FEATURES
    features = [f for f in features if f in df.columns]

    sub = df.dropna(subset=features + [crack_col]).copy()
    if len(sub) < 24:
        return {"error": f"For lite data: {len(sub)} obs"}

    X = sm.add_constant(sub[features].astype(float))
    y = sub[crack_col].astype(float)
    m = sm.OLS(y, X).fit(cov_type="HC1")

    # Cross-validation
    kf = KFold(n_splits=min(5, len(sub) // 6), shuffle=True, random_state=42)
    cv = cross_val_score(LinearRegression(), sub[features].values, y.values,
                          cv=kf, scoring="r2")

    # OOT — bruk siste 25% som test
    n = len(sub)
    cutoff_idx = int(n * 0.75)
    train = sub.iloc[:cutoff_idx]
    test  = sub.iloc[cutoff_idx:]
    r2_oot = np.nan
    if len(test) > 4:
        lr = LinearRegression().fit(train[features].values, train[crack_col].values)
        pred = lr.predict(test[features].values)
        ss_res = ((test[crack_col].values - pred) ** 2).sum()
        ss_tot = ((test[crack_col].values - test[crack_col].mean()) ** 2).sum()
        r2_oot = 1 - ss_res / ss_tot

    rmse = np.sqrt(((y - m.fittedvalues) ** 2).mean())

    return {
        "model":        m,
        "features":     features,
        "n":            len(sub),
        "r2":           m.rsquared,
        "r2_adj":       m.rsquared_adj,
        "r2_cv":        cv.mean(),
        "r2_oot":       r2_oot,
        "rmse":         rmse,
        "y_range":      [y.min(), y.max()],
        "y_mean":       y.mean(),
    }


def print_model_summary(crack_col: str, cfg: dict, res: dict) -> None:
    print(f"\n{'─' * 75}")
    print(f"  {cfg['label']}")
    print(f"  {'─' * 75}")
    print(f"  N={res['n']} mnd | R²={res['r2']:.3f} | adj R²={res['r2_adj']:.3f}"
          f" | CV={res['r2_cv']:.3f} | OOT={res['r2_oot']:.3f}")
    print(f"  RMSE={res['rmse']:.2f} USD/bbl (y-range: "
          f"{res['y_range'][0]:.1f}–{res['y_range'][1]:.1f}, snitt {res['y_mean']:.1f})")
    print(f"\n  {'Feature':<35} {'Coef':>10} {'SE':>8} {'p':>7} {'Sig':>5}")
    print(f"  {'-' * 70}")
    m = res["model"]
    for f in ["const"] + res["features"]:
        coef = m.params.get(f, np.nan)
        se   = m.bse.get(f, np.nan)
        pval = m.pvalues.get(f, np.nan)
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else \
                "*" if pval < 0.05 else "." if pval < 0.10 else ""
        print(f"  {f:<35} {coef:>+10.4f} {se:>8.3f} {pval:>7.3f} {sig:>5}")


def main():
    print("=" * 75)
    print("  SCRIPT 54: Stage 1 — Demand → Crack-spread modell")
    print("=" * 75)

    print("\n[1] Laster og merger panel + demand-data...")
    df = load_combined_panel()

    print("\n[2] Trener én modell per crack spread...")
    results = {}
    for crack_col, cfg in PRODUCT_MODELS.items():
        res = train_crack_model(df, crack_col, cfg)
        if "error" in res:
            print(f"  ⚠ {crack_col}: {res['error']}")
            continue
        results[crack_col] = res
        print_model_summary(crack_col, cfg, res)

    # ── Lagre modell-koeffisienter for stage 2 ──────────────────────────────
    print(f"\n[3] Lagrer modell-koeffisienter til {OUT_JSON.name}...")
    output = {}
    for crack_col, res in results.items():
        m = res["model"]
        output[crack_col] = {
            "label":        PRODUCT_MODELS[crack_col]["label"],
            "features":     res["features"],
            "coefficients": {k: round(v, 6) for k, v in m.params.items()},
            "std_errors":   {k: round(v, 6) for k, v in m.bse.items()},
            "p_values":     {k: round(v, 6) for k, v in m.pvalues.items()},
            "metrics": {
                "n_obs":   res["n"],
                "r2":      round(res["r2"], 4),
                "r2_adj":  round(res["r2_adj"], 4),
                "r2_cv":   round(res["r2_cv"], 4),
                "r2_oot":  round(res["r2_oot"], 4) if not np.isnan(res["r2_oot"]) else None,
                "rmse":    round(res["rmse"], 4),
            },
            "y_mean":       round(res["y_mean"], 2),
            "y_range":      [round(v, 2) for v in res["y_range"]],
        }
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  ✓ Lagret: {OUT_JSON.name}")

    # ── Oppsummerende tolkning ───────────────────────────────────────────────
    print(f"\n[4] TOLKNING — hvilke demand-elastisiteter er statistisk meningsfulle?")
    print(f"{'─' * 75}")
    for crack_col, res in results.items():
        cfg = PRODUCT_MODELS[crack_col]
        m = res["model"]
        demand_coef = m.params.get(cfg["demand_col"], 0)
        demand_p    = m.pvalues.get(cfg["demand_col"], 1)
        stocks_coef = m.params.get(cfg["stocks_col"], 0)
        stocks_p    = m.pvalues.get(cfg["stocks_col"], 1)

        print(f"\n  {cfg['label']}")
        print(f"    Demand-effekt: +1 kbpd → {demand_coef:+.4f} USD/bbl "
              f"(p={demand_p:.3f}{'***' if demand_p<0.001 else '**' if demand_p<0.01 else '*' if demand_p<0.05 else ''})")
        print(f"    Stocks-effekt: +1 mbbl → {stocks_coef:+.4f} USD/bbl "
              f"(p={stocks_p:.3f}{'***' if stocks_p<0.001 else '**' if stocks_p<0.01 else '*' if stocks_p<0.05 else ''})")

        # Praktisk eksempel
        typical_demand_move = 200  # 200 kbpd = realistisk månedlig svingning
        impact = demand_coef * typical_demand_move
        print(f"    → Praktisk: hvis demand stiger {typical_demand_move:,} kbpd, "
              f"crack endres med {impact:+.2f} USD/bbl")


if __name__ == "__main__":
    main()
