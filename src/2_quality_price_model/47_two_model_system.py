"""
Script 47 — To-modell-system: Global vs. Brent-linked

MODELLDESIGN:
  Modell A — Global:       Alle 43 grades. Brukes for globale oljeselskaper og
                           cross-regional M&A-analyse. Inkluderer WTI-linkede
                           grades (kanoniksk, Maya, ecuadorianske etc.).

  Modell B — Brent-linked: 32 grades der prisen settes mot Dated Brent.
                           Ekskluderer alle grades der referanseprisen er WTI
                           (Cushing, Edmonton, Houston). Brukes for NCS-selskaper
                           (AKRBP, Equinor, Vår Energi) og Vest-Afrika/Midtøsten
                           til Europa-salg. Metodisk renere for NCS-analyse.

BEGRUNNELSE FOR UTVALG:
  WTI-linkede grades ekskludert fra Modell B:
    WTI               → Cushing-USGC (referanse er WTI Nymex)
    Bow River Heavy   → AB-USGC (pris satt vs. WTI Edmonton/Hardisty)
    Canadian Light Sour→ AB-USGC (pris satt vs. WTI Edmonton)
    Lloydminster      → AB-USGC (pris satt vs. WTI Hardisty)
    Maya              → Mex-USGC (pris satt vs. WTI + tariff)
    Olmeca            → Mex-USGC (primærmarked USGC)
    Merey             → Ven-USGC (primærmarked USGC)
    Leona             → Ven-USGC (primærmarked USGC)
    Napo              → Ecu-USGC (pris satt vs. WTI)
    Oriente           → Ecu-USGC (pris satt vs. WTI)
    Marlim            → Brazil-Asia (blandet benchmark, usikker Brent-link)

  Beholdt i Modell B (Brent-linked):
    NCS (21 grades):  Alvheim, Asgard, Balder, Draugen, Ekofisk, Gina Krog,
                      Goliat, Grane, Gudrun, Gullfaks, Heidrun, Johan Sverdrup,
                      Jotun, Knarr, Martin Linge, Njord, Norne, Oseberg,
                      Skarv, Statfjord, Troll
    Vest-Afrika (5):  Bonny Light, Cabinda, Forcados, Qua Iboe, Rabi Light
    Nord-Afrika (1):  Saharan Blend
    Midtøsten (5):    Arab Extra Light, Arab Light, Arab Medium, Basrah Light,
                      Dubai Fateh  (europesalg på Dated Brent-basis)

LAGRER:
  34_parsimonious_model.json          — Modell A (Global, uendret)
  34b_brent_model.json               — Modell B (Brent-linked, ny)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Grades som ekskluderes fra Modell B ──────────────────────────────────────
WTI_LINKED = {
    "WTI", "Bow River Heavy", "Canadian Light Sour", "Lloydminster",
    "Maya", "Olmeca", "Merey", "Leona", "Napo", "Oriente", "Marlim",
}

ALWAYS_KEEP = {"api_gravity", "sulfur_pct"}
ALWAYS_DROP = {"vix"}
SIG_THRESHOLD = 0.10

INTERACTION_PARENTS = {
    "api2":                       ["api_gravity"],
    "sulfur_x_brent":             ["sulfur_pct", "brent_price"],
    "vacuum_resid_x_brent":       ["vacuum_resid_pct", "brent_price"],
    "ccr_x_brent":                ["ccr_wt_pct", "brent_price"],
    "landlocked_x_cushing_stocks":["is_landlocked", "cushing_stocks_kbbl_dev_5y_pct"],
    "api_x_contango":             ["api_gravity", "fc_slope_4m"],
    "sulfur_x_refinery_util":     ["sulfur_pct", "us_refinery_util_pct"],
    "landlocked_x_contango":      ["is_landlocked", "fc_slope_4m"],
}


def load_panel(brent_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "regression_panel.csv")
    df["date"] = pd.to_datetime(df["date_str"])

    if brent_only:
        df = df[~df["grade"].isin(WTI_LINKED)].copy()

    region_simple = {
        "North Sea": "NorthSea", "Norwegian Sea": "NorthSea", "Barents Sea": "NorthSea",
        "North America": "NorthAmerica", "Gulf of Mexico": "NorthAmerica",
        "South America": "LatAm", "Middle East": "MiddleEast",
        "West Africa": "WestAfrica", "North Africa": "NorthAfrica",
        "FSU": "FSU", "Asia-Pacific": "AsiaPac", "Various": "NorthAmerica",
    }
    df["region_simple"] = df["region"].map(region_simple).fillna("Other")
    return df


def fit_eval(df: pd.DataFrame, features: list[str]) -> dict | None:
    sub = df.dropna(subset=features + ["differential"]).copy()
    if len(sub) < 100:
        return None
    X = sm.add_constant(sub[features].astype(float))
    y = sub["differential"]
    m = sm.OLS(y, X).fit(cov_type="HC1")

    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(LinearRegression(), sub[features].values, y.values, cv=kf, scoring="r2")

    cutoff = sub["date"].max() - pd.DateOffset(months=24)
    train  = sub[sub["date"] <= cutoff]
    test   = sub[sub["date"] > cutoff]
    r2_oot = np.nan
    if len(train) > 100 and len(test) > 30:
        lr = LinearRegression().fit(train[features].values, train["differential"].values)
        pred   = lr.predict(test[features].values)
        ss_res = ((test["differential"].values - pred) ** 2).sum()
        ss_tot = ((test["differential"].values - test["differential"].mean()) ** 2).sum()
        r2_oot = 1 - ss_res / ss_tot

    rmse = np.sqrt(((y - m.fittedvalues) ** 2).mean())
    return {
        "model": m, "n": len(sub), "k": len(features),
        "r2": m.rsquared, "r2_adj": m.rsquared_adj,
        "r2_cv": cv.mean(), "r2_oot": r2_oot, "rmse": rmse,
        "features": features, "data": sub,
        "grades": sorted(sub["grade"].unique().tolist()),
    }


def stepwise_eliminate(df: pd.DataFrame, initial_features: list[str],
                        label: str, verbose: bool = True) -> dict:
    features = [f for f in initial_features if f not in ALWAYS_DROP]
    it = 0
    while True:
        it += 1
        result = fit_eval(df, features)
        if result is None:
            return None
        m    = result["model"]
        pvals = m.pvalues.drop("const", errors="ignore")

        protected = set(ALWAYS_KEEP)
        for inter, parents in INTERACTION_PARENTS.items():
            if inter in features and inter in pvals.index and pvals[inter] < SIG_THRESHOLD:
                protected.update(parents)

        candidates = pvals[pvals > SIG_THRESHOLD].sort_values(ascending=False)
        candidates = candidates[~candidates.index.isin(protected)]

        if len(candidates) == 0:
            if verbose:
                print(f"  [{label}] Konvergerte etter {it} iter. "
                      f"k={len(features)}, R²={result['r2']:.4f}, "
                      f"CV={result['r2_cv']:.4f}, OOT={result['r2_oot']:.4f}")
            return result

        worst = candidates.index[0]
        features = [f for f in features if f != worst]
        if len(features) < 5:
            return result


def save_model(result: dict, path: Path, model_name: str) -> None:
    m = result["model"]
    obj = {
        "model_name":   model_name,
        "grades":       result["grades"],
        "n_grades":     len(result["grades"]),
        "metrics": {
            "n_obs":    result["n"],
            "k":        result["k"],
            "r2":       round(result["r2"], 4),
            "r2_adj":   round(result["r2_adj"], 4),
            "r2_cv":    round(result["r2_cv"], 4),
            "r2_oot":   round(result["r2_oot"], 4) if not np.isnan(result["r2_oot"]) else None,
            "rmse":     round(result["rmse"], 4),
        },
        "features":     result["features"],
        "coefficients": {k: round(v, 6) for k, v in m.params.items()},
        "std_errors":   {k: round(v, 6) for k, v in m.bse.items()},
        "p_values":     {k: round(v, 6) for k, v in m.pvalues.items()},
    }
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  Lagret: {path.name}")


def build_initial_features(df: pd.DataFrame) -> list[str]:
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg",
                                 drop_first=True, dtype=int)
    df_ext = pd.concat([df, region_dums], axis=1)
    region_cols = list(region_dums.columns)
    # Update df in-place for downstream use
    for c in region_cols:
        df[c] = df_ext[c].values

    initial = (
        ["api_gravity", "sulfur_pct", "api2"]
        + region_cols
        + ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"]
        + ["brent_price", "wti_brent_spread", "vix"]
        + ["gasoline_crack_brent", "diesel_crack_brent",
           "diesel_minus_gasoline_crack", "brent_dubai_spread"]
        + ["d_distance_medium", "d_distance_long",
           "is_landlocked", "is_pipeline_constrained"]
        + ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
           "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
           "d_refinery_tight", "d_refinery_slack"]
        + ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"]
        + ["sin_month", "cos_month", "d_winter"]
        + ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
           "landlocked_x_cushing_stocks", "api_x_contango",
           "sulfur_x_refinery_util", "landlocked_x_contango"]
        + ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
           "d_venezuela_sanctions", "d_us_shale_boom", "d_covid",
           "d_opec_plus_cuts_2023"]
    )
    return [f for f in initial if f in df.columns]


def build_initial_features_brent(df: pd.DataFrame) -> list[str]:
    """
    Brent-spesifikk feature-liste med følgende tilpasninger vs. Global-modellen:

    1. is_landlocked / is_pipeline_constrained: FJERNET — null variasjon i Brent-panelet
       (ingen av de 32 Brent-linkede grades er landlocked eller pipeline-constrained)

    2. d_distance_medium: FJERNET — perfekt kollineær med reg_WestAfrica i Brent-panelet.
       Alle medium-distance grades i Brent-panelet er Vest-Afrikanske. Beholder
       reg_WestAfrica for å gi den rette regionale tolkingen.

    3. landlocked_x_cushing_stocks / landlocked_x_contango: FJERNET — avhenger av
       is_landlocked som er null overalt i Brent-panelet.

    4. wti_brent_spread: FJERNET — ingen WTI-linkede grades i Brent-panelet; WTI/Brent-
       spreaden påvirker ikke prissettingen av Dated Brent-differensialer.

    5. d_venezuela_sanctions: FJERNET — ingen venezuelanske grades i Brent-panelet.

    6. reg_NorthAfrica: Beholdes i initial, men kan droppes av stepwise (kun Saharan Blend,
       46 obs). Stepwise eliminerer den hvis den ikke er signifikant.

    Alle andre features identiske med Global-modellen.
    """
    # Bygg region-dummies — bruk kun regioner med >1 grade for stabilitet
    region_dums = pd.get_dummies(df["region_simple"], prefix="reg",
                                 drop_first=True, dtype=int)

    # Sjekk antall grades per region — dropp enkelt-grade regioner fra initial
    grade_per_region = df.groupby("region_simple")["grade"].nunique()
    single_grade_regions = grade_per_region[grade_per_region == 1].index.tolist()
    if single_grade_regions:
        print(f"  [Brent] Enkelt-grade regioner (holdes i initial, stepwise bestemmer): "
              f"{single_grade_regions}")

    df_ext = pd.concat([df, region_dums], axis=1)
    region_cols = list(region_dums.columns)
    for c in region_cols:
        df[c] = df_ext[c].values

    # Fjern regioner med null variasjon (alle 0 eller alle 1)
    region_cols_valid = []
    for c in region_cols:
        vals = df[c].values
        if vals.sum() > 0 and vals.sum() < len(vals):
            region_cols_valid.append(c)
        else:
            print(f"  [Brent] Fjerner {c} — ingen variasjon (sum={vals.sum()})")

    # Brent-spesifikke eksklusjoner
    BRENT_EXCLUDE = {
        "is_landlocked",            # null variasjon
        "is_pipeline_constrained",  # null variasjon
        "d_distance_medium",        # perfekt kollineær med reg_WestAfrica
        "landlocked_x_cushing_stocks",  # avhenger av is_landlocked
        "landlocked_x_contango",    # avhenger av is_landlocked
        "wti_brent_spread",         # ingen WTI-linkede grades
        "d_venezuela_sanctions",    # ingen venezuelanske grades
    }

    initial = (
        ["api_gravity", "sulfur_pct", "api2"]
        + region_cols_valid
        + ["vacuum_resid_pct", "middle_distillate_pct", "ccr_wt_pct", "log_v_ni"]
        + ["brent_price", "vix"]
        + ["gasoline_crack_brent", "diesel_crack_brent",
           "diesel_minus_gasoline_crack", "brent_dubai_spread"]
        + ["d_distance_long"]
        + ["us_refinery_util_pct", "us_crude_stocks_kbbl_dev_5y_pct",
           "cushing_stocks_kbbl_dev_5y_pct", "us_crude_exports_kbpd",
           "d_refinery_tight", "d_refinery_slack"]
        + ["fc_slope_4m", "d_strong_contango", "d_strong_backwardation"]
        + ["sin_month", "cos_month", "d_winter"]
        + ["sulfur_x_brent", "vacuum_resid_x_brent", "ccr_x_brent",
           "api_x_contango", "sulfur_x_refinery_util"]
        + ["d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
           "d_us_shale_boom", "d_covid", "d_opec_plus_cuts_2023"]
    )
    return [f for f in initial if f in df.columns and f not in BRENT_EXCLUDE]


def sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."


def print_model_summary(result: dict, label: str) -> None:
    m = result["model"]
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  N={result['n']:,} obs | {len(result['grades'])} grades | k={result['k']} features")
    print(f"  R²={result['r2']:.4f} | adj R²={result['r2_adj']:.4f} | "
          f"CV R²={result['r2_cv']:.4f} | OOT R²={result['r2_oot']:.4f} | "
          f"RMSE={result['rmse']:.2f}")
    print(f"{'='*80}")

    coef_df = pd.DataFrame({
        "coef": m.params, "se": m.bse, "t": m.tvalues, "p": m.pvalues,
    }).drop("const", errors="ignore")
    coef_df["sig"] = coef_df["p"].apply(sig)
    for _, row in coef_df.iterrows():
        bar = "█" * min(20, int(abs(row["coef"]) * 2))
        sign = "+" if row["coef"] > 0 else "-"
        print(f"  {_:<35} {sign}{abs(row['coef']):6.4f}  {row['sig']:<4} {bar}")


def compare_models(a: dict, b: dict) -> None:
    print(f"\n{'='*80}")
    print("  SAMMENLIGNING: Modell A (Global) vs. Modell B (Brent-linked)")
    print(f"{'='*80}")
    print(f"  {'Metrikk':<20} {'Modell A':>12} {'Modell B':>12} {'Δ':>8}")
    print(f"  {'-'*52}")
    for metric, key in [("N obs", "n"), ("Grades", None), ("k features", "k"),
                         ("R²", "r2"), ("CV R²", "r2_cv"), ("OOT R²", "r2_oot"),
                         ("RMSE", "rmse")]:
        if key is None:
            va = len(a["grades"]); vb = len(b["grades"])
        else:
            va = a[key]; vb = b[key]
        if isinstance(va, float):
            print(f"  {metric:<20} {va:>12.4f} {vb:>12.4f} {vb-va:>+8.4f}")
        else:
            print(f"  {metric:<20} {va:>12,} {vb:>12,} {vb-va:>+8,}")

    # Koeffisienter som endret seg mest
    coefs_a = dict(a["model"].params)
    coefs_b = dict(b["model"].params)
    shared  = set(coefs_a) & set(coefs_b) - {"const"}
    changes = {f: coefs_b[f] - coefs_a[f] for f in shared}
    top     = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    print(f"\n  Koeffisienter med størst endring (A→B):")
    for f, delta in top:
        print(f"    {f:<35} A={coefs_a[f]:+.4f}  B={coefs_b[f]:+.4f}  Δ={delta:+.4f}")


def main() -> None:
    print("="*80)
    print("  TO-MODELL-SYSTEM: Global (A) vs. Brent-linked (B)")
    print("="*80)

    # ── Modell A: Global ──────────────────────────────────────────────────────
    print("\n[1/2] Trener Modell A — Global (alle grades)...")
    df_a   = load_panel(brent_only=False)
    feats_a = build_initial_features(df_a)
    print(f"  Panel: {len(df_a):,} obs, {df_a['grade'].nunique()} grades, "
          f"{len(feats_a)} kandidat-features")
    result_a = stepwise_eliminate(df_a, feats_a, label="Global")
    save_model(result_a, PROCESSED_DIR / "34_parsimonious_model.json",
               "Parsimonious OLS — Global (43 grades, Brent differential)")

    # ── Modell B: Brent-linked ────────────────────────────────────────────────
    print("\n[2/2] Trener Modell B — Brent-linked (32 grades)...")
    df_b   = load_panel(brent_only=True)
    feats_b = build_initial_features_brent(df_b)
    print(f"  Panel: {len(df_b):,} obs, {df_b['grade'].nunique()} grades, "
          f"{len(feats_b)} kandidat-features")
    print(f"  Ekskluderte grades: {sorted(WTI_LINKED)}")

    # ── Diagnostikk: verifiser at spesifikasjonsfikser er gyldige ────────────
    print(f"\n  [Brent-diagnostikk]")
    for col in ["is_landlocked", "is_pipeline_constrained"]:
        if col in df_b.columns:
            u = df_b[col].unique()
            print(f"    {col}: unique={sorted(u)}  (forventer [0] → fjernet OK)")
    # Sjekk overlap medium / WestAfrica
    if "d_distance_medium" in df_b.columns and "reg_WestAfrica" in df_b.columns:
        overlap = df_b.groupby("distance_band")["reg_WestAfrica"].sum()
        print(f"    d_distance_medium ↔ WestAfrica overlap:\n{overlap.to_string()}")
    # Region-distribusjon
    print(f"    Region-distribusjon: "
          f"{dict(df_b.groupby('region_simple')['grade'].nunique())}")
    result_b = stepwise_eliminate(df_b, feats_b, label="Brent")
    save_model(result_b, PROCESSED_DIR / "34b_brent_model.json",
               "Parsimonious OLS — Brent-linked (32 grades, Dated Brent differential)")

    # ── Sammenligning ─────────────────────────────────────────────────────────
    compare_models(result_a, result_b)

    # ── Residualer per grade (Modell B) ──────────────────────────────────────
    print(f"\n  Residualer per grade — Modell B (Brent-linked):")
    data_b = result_b["data"].copy()
    data_b["pred"] = result_b["model"].fittedvalues
    data_b["resid"] = data_b["differential"] - data_b["pred"]
    g = data_b.groupby("grade").agg(
        n=("differential", "count"),
        mean_actual=("differential", "mean"),
        mean_pred=("pred", "mean"),
        resid=("resid", "mean"),
        rmse=("resid", lambda x: np.sqrt((x**2).mean())),
        api=("api_gravity", "first"),
        sulfur=("sulfur_pct", "first"),
    ).round(2).sort_values("resid", ascending=False)
    print(g[["n","api","sulfur","mean_actual","mean_pred","resid","rmse"]].to_string())

    # ── Plott: sammenligning predikert vs. faktisk (Modell B) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle("Modell B (Brent-linked) — Predikert vs. faktisk differensial",
                 fontsize=13, fontweight="bold")

    for ax, (data, label, color) in zip(axes, [
        (result_a["data"], "Modell A — Global", "#C0392B"),
        (result_b["data"], "Modell B — Brent-linked", "#1A5276"),
    ]):
        pred   = result_a["model"].fittedvalues if "Global" in label else result_b["model"].fittedvalues
        actual = data["differential"]
        ax.scatter(actual, pred, alpha=0.15, s=8, color=color)
        lim = [min(actual.min(), pred.min()) - 1,
               max(actual.max(), pred.max()) + 1]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Faktisk differensial (USD/bbl)", fontsize=10)
        ax.set_ylabel("Predikert differensial (USD/bbl)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold", color=color)
        ax.set_facecolor("#FAFAFA")
        r = "A" if "Global" in label else "B"
        res = result_a if "Global" in label else result_b
        ax.text(0.05, 0.92, f"R²={res['r2']:.3f}  OOT={res['r2_oot']:.3f}  RMSE={res['rmse']:.2f}",
                transform=ax.transAxes, fontsize=9, family="monospace",
                bbox=dict(facecolor="white", edgecolor="#AAAAAA", alpha=0.9))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out_png = PROCESSED_DIR / "47_two_model_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Plott: {out_png}")
    print("\n  Ferdig. Bruk 34_parsimonious_model.json for globale selskaper,")
    print("          34b_brent_model.json for NCS/Brent-linkede selskaper.")


if __name__ == "__main__":
    main()
