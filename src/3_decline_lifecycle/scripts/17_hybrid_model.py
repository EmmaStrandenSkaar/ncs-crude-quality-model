"""
Script 17: Hybrid Model — Best of Both Worlds
═══════════════════════════════════════════════════════════════════════════

Bruker NY API kun når konfidens er høy (operator_direct ELLER dst_robust med n≥10),
ellers beholder gammel API. Dette gir mest fysisk korrekte modell uten å miste
presisjon på felt der vi har lavkvalitets ny data.

Sammenligner 4 modeller:
  V1: Old (alle blend-API)
  V2: New (alle reservoar-API, inkl. lavkonfidens)
  V3: Hybrid (kun høy-konfidens oppdateringer)
  V4: Hybrid + field-specific T (når tilgjengelig)
"""

import json, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
GEO = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir_geo" / "fields.geojson"
WINDOW_MONTHS = 12

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

with open(GEO) as f:
    gj = json.load(f)
op_map = {feat["properties"].get("fldName"): (feat["properties"].get("cmpLongName") or "")
          for feat in gj["features"] if feat["properties"].get("fldName")}
akerbp_fields = {k for k, v in op_map.items() if "Aker BP" in v}

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")
master = pd.read_csv(DATA / "master_fluid_library.csv")

log("═" * 80)
log("HYBRID DECLINE MODEL — Selektiv reservoar-API oppdatering")
log("═" * 80)

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

def loo_cv(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    if len(y) < 10: return np.nan, len(y)
    loo = LeaveOneOut()
    lr = LinearRegression()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        lr.fit(X[tr], y[tr])
        preds[te] = lr.predict(X[te])
    return 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2), len(y)

# ═══════════════════════════════════════════════════════════════
# BUILD HYBRID API
# ═══════════════════════════════════════════════════════════════
df = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
df = df.rename(columns={"api_gravity": "api_old"})

master_subset = master[["field", "api_gravity", "api_quality_tier", "api_confidence",
                        "reservoir_temp_c"]].rename(columns={"api_gravity": "api_master"})
df = df.merge(master_subset, on="field", how="left")

# Hybrid: use new API only for high-confidence cases
HIGH_CONF_TIERS = {"operator_direct", "dst_robust"}
df["api_high_conf"] = df.apply(
    lambda r: r.api_master if (
        pd.notna(r.api_master) and
        r.api_quality_tier in HIGH_CONF_TIERS and
        r.api_confidence == "high"
    ) else r.api_old, axis=1
)

# Also: medium-conf hybrid (operator_direct + dst_robust + operator_medium)
MED_CONF_TIERS = {"operator_direct", "dst_robust", "operator_medium"}
df["api_med_conf"] = df.apply(
    lambda r: r.api_master if (
        pd.notna(r.api_master) and
        r.api_quality_tier in MED_CONF_TIERS and
        r.api_confidence in ["high", "medium"]
    ) else r.api_old, axis=1
)

# Statistics
log(f"\nFelt med høy-konfidens API: {(df.api_old != df.api_high_conf).sum()}/{len(df)}")
log(f"Felt med medium-konfidens API: {(df.api_old != df.api_med_conf).sum()}/{len(df)}")
log(f"Felt med ALL ny API: {(df.api_old != df.api_master.fillna(df.api_old)).sum()}/{len(df)}")

# Compute viscosities
df["visc_v1_old"] = beggs_robinson(df.api_old)

df["api_v2_all"] = df.api_master.fillna(df.api_old)
df["visc_v2_all"] = beggs_robinson(df.api_v2_all)

df["visc_v3_hybrid_high"] = beggs_robinson(df.api_high_conf)
df["visc_v4_hybrid_med"] = beggs_robinson(df.api_med_conf)

# V5: hybrid + field T (only when high confidence has temp)
def hybrid_temp_visc(row):
    api = row.api_med_conf  # Use medium-conf hybrid
    T_c = row.reservoir_temp_c
    use_T = (
        pd.notna(T_c) and T_c > 30 and
        row.api_quality_tier in HIGH_CONF_TIERS and
        row.api_confidence == "high"
    )
    T_F = (T_c * 9/5 + 32) if use_T else 194
    return beggs_robinson(api, T_F)
df["visc_v5_hybrid_T"] = df.apply(hybrid_temp_visc, axis=1)

# Log viscosities
for col in ["visc_v1_old", "visc_v2_all", "visc_v3_hybrid_high", "visc_v4_hybrid_med", "visc_v5_hybrid_T"]:
    df[f"ln_{col}"] = np.log(df[col])

# ═══════════════════════════════════════════════════════════════
# PREMIUM COMPUTATION
# ═══════════════════════════════════════════════════════════════
post = panel[panel.is_post_peak].copy()
post["prod_frac"] = post.oil_pct_peak / 100.0

def compute_premium(D_phys_map):
    out = []
    for field, grp in post.groupby("field"):
        if field not in D_phys_map: continue
        D_phys = D_phys_map[field]
        grp = grp.sort_values("months_since_peak")
        grp = grp[grp.prod_frac > 0.01]
        if len(grp) < 12: continue
        t = grp.months_since_peak.values
        log_prem = np.log(grp.prod_frac.values) - np.log(np.exp(-D_phys / 12 * t))
        mask = t >= t.max() - WINDOW_MONTHS
        if mask.sum() < 6: continue
        out.append({"field": field, "premium": log_prem[mask].mean()})
    return pd.DataFrame(out)

# ═══════════════════════════════════════════════════════════════
# FIT 5 VERSIONS
# ═══════════════════════════════════════════════════════════════
versions = {
    "V1_old":           ("ln_visc_v1_old",          "Original (blend API)"),
    "V2_all_new":       ("ln_visc_v2_all",          "Full reservoir-API replacement"),
    "V3_hybrid_high":   ("ln_visc_v3_hybrid_high",  "Hybrid: only high-confidence updates"),
    "V4_hybrid_med":    ("ln_visc_v4_hybrid_med",   "Hybrid: high + medium confidence"),
    "V5_hybrid_T":      ("ln_visc_v5_hybrid_T",     "Hybrid (med) + field-specific T"),
}

results = {}
for ver_name, (visc_col, label) in versions.items():
    valid = df.dropna(subset=[visc_col, "D_annual"])
    X_p = valid[[visc_col]].values
    y_p = valid["D_annual"].values
    lr_p = LinearRegression().fit(X_p, y_p)
    valid["D_physics"] = lr_p.predict(X_p)
    D_phys_map = valid.set_index("field")["D_physics"].to_dict()
    prem = compute_premium(D_phys_map)
    full = valid.merge(prem, on="field", how="inner")
    full["abs_premium"] = full.premium.abs()
    full["is_akerbp"] = full.field.isin(akerbp_fields)

    feats = [visc_col, "premium", "abs_premium"]
    X = full[feats].values
    y = full["D_annual"].values
    lr = LinearRegression().fit(X, y)
    cv_r2, n = loo_cv(X, y)
    rmse = np.sqrt(np.mean((y - lr.predict(X))**2))
    full["D_pred"] = lr.predict(X)
    full["resid"] = y - full.D_pred
    ak_rmse = np.sqrt((full[full.is_akerbp].resid**2).mean())

    results[ver_name] = {
        "label": label, "model": lr, "phys_model": lr_p,
        "cv_r2": cv_r2, "in_r2": lr.score(X, y), "rmse": rmse,
        "akbp_rmse": ak_rmse, "n": n, "df": full,
        "coef": lr.coef_, "intercept": lr.intercept_,
        "n_changed": (df.api_old != valid[visc_col.replace("ln_visc_", "")].apply(
            lambda v: 0 if pd.isna(v) else v)).sum() if False else 0,
    }

# ═══════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 95)
log("MODELL-SAMMENLIGNING")
log("═" * 95)
log(f"\n{'Versjon':22s} {'Beskrivelse':35s} {'CV R²':>7s} {'RMSE':>7s} {'AkBP RMSE':>11s} {'ln(μ) β':>10s}")
log("─" * 95)

best_cv = max(r["cv_r2"] for r in results.values())
for ver, r in results.items():
    marker = " ★" if r["cv_r2"] == best_cv else "  "
    log(f"  {ver:20s} {r['label']:35s} {r['cv_r2']:6.3f}  {r['rmse']:6.4f}  "
        f"{r['akbp_rmse']:10.4f}  {r['coef'][0]:+9.4f}{marker}")

# Aker BP-fokus
log(f"\n── AKER BP RMSE per versjon ──")
for ver, r in results.items():
    log(f"  {ver:22s}  {r['akbp_rmse']:.4f}")

# Per-field comparison (Aker BP only)
log(f"\n── Aker BP residual per modell-versjon ──")
log(f"\n{'Field':18s} {'D_act':>6s} " + " ".join(f"{v.replace('V', 'V'):>10s}" for v in versions))
log("─" * 95)

all_akbp = {ver: r["df"][r["df"].is_akerbp].set_index("field") for ver, r in results.items()}
common = sorted(set.intersection(*[set(df_v.index) for df_v in all_akbp.values()]))

for f in common:
    line = f"{f:18s} {all_akbp['V1_old'].loc[f, 'D_annual']:6.3f}"
    for ver in versions:
        resid = all_akbp[ver].loc[f, "resid"]
        line += f"  {resid:+9.3f}"
    log(line)

# Save best version's predictions
best_ver = max(results.items(), key=lambda kv: kv[1]["cv_r2"])
log(f"\n── BEST MODELL: {best_ver[0]} ({best_ver[1]['label']}) ──")
log(f"   CV R² = {best_ver[1]['cv_r2']:.3f}")
log(f"   RMSE  = {best_ver[1]['rmse']:.4f}")
log(f"   Aker BP RMSE = {best_ver[1]['akbp_rmse']:.4f}")
log(f"   Koeffisienter:")
log(f"     Intercept:     {best_ver[1]['intercept']:.4f}")
log(f"     ln(viscosity): {best_ver[1]['coef'][0]:+.4f}")
log(f"     premium:       {best_ver[1]['coef'][1]:+.4f}")
log(f"     |premium|:     {best_ver[1]['coef'][2]:+.4f}")

best_ver[1]["df"][["field", "D_annual", "D_pred", "resid", "premium",
                   "abs_premium", "is_akerbp"]].to_csv(DATA / "predictions_best.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
fig.suptitle("Hybrid Modell-sammenligning: Hvor mye reservoar-API skal vi stole på?",
             fontsize=14, fontweight="bold", y=1.0)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

# Panel 1: CV R² comparison
ax = fig.add_subplot(gs[0, 0])
ver_list = list(versions.keys())
cv_vals = [results[v]["cv_r2"] for v in ver_list]
labels = ["V1\nOld blend", "V2\nAll new", "V3\nHigh conf", "V4\nMed conf", "V5\nMed + T"]
colors = ["#9E9E9E", "#1976D2", "#2E7D32", "#E65100", "#7B1FA2"]
bars = ax.bar(range(5), cv_vals, color=colors, alpha=0.85)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("LOO Cross-validated R²")
ax.set_title("CV R² per modell-versjon", fontsize=11, fontweight="bold")
for i, v in enumerate(cv_vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(min(cv_vals) - 0.03, max(cv_vals) * 1.05)

# Panel 2: Aker BP RMSE
ax = fig.add_subplot(gs[0, 1])
rmse_vals = [results[v]["akbp_rmse"] for v in ver_list]
ax.bar(range(5), rmse_vals, color=colors, alpha=0.85)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Aker BP RMSE")
ax.set_title("Aker BP prediksjons-feil", fontsize=11, fontweight="bold")
for i, v in enumerate(rmse_vals):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

# Panel 3: Viscosity coefficient
ax = fig.add_subplot(gs[0, 2])
coef_vals = [results[v]["coef"][0] for v in ver_list]
ax.bar(range(5), coef_vals, color=colors, alpha=0.85)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("β ln(viscosity)")
ax.set_title("Fysikk-koeffisient (fortegn matters!)", fontsize=11, fontweight="bold")
for i, v in enumerate(coef_vals):
    ax.text(i, v + 0.003 * (1 if v > 0 else -1), f"{v:+.3f}", ha="center", fontsize=8, fontweight="bold")

# Panel 4-6: Aker BP per-field residuals
ak_resid_df = pd.DataFrame({
    "field": common,
    **{ver: [all_akbp[ver].loc[f, "resid"] for f in common] for ver in versions}
}).set_index("field")

ax = fig.add_subplot(gs[1, :])
ak_resid_df_sorted = ak_resid_df.sort_values("V1_old")
y_pos = np.arange(len(ak_resid_df_sorted))
width = 0.16
for i, ver in enumerate(ver_list):
    ax.barh(y_pos + (i - 2) * width, ak_resid_df_sorted[ver].values, width,
            color=colors[i], alpha=0.85, label=ver)
ax.set_yticks(y_pos)
ax.set_yticklabels(ak_resid_df_sorted.index, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Residual (D_actual − D_predicted)")
ax.set_title("Aker BP residual per felt — alle 5 versjoner", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_hybrid_comparison.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_hybrid_comparison.png")

with open(RESULTS / "hybrid_comparison.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved: hybrid_comparison.txt")
