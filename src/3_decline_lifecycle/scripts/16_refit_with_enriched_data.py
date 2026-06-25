"""
Script 16: Refit Decline Model with Enriched Fluid Data
═══════════════════════════════════════════════════════════════════════════

Sammenligner gammel modell (blend API) vs ny modell (reservoar API fra master library).

Tester:
  1. Beggs-Robinson med ny API (samme T) → ny viskositet → ny D_physics
  2. Beggs-Robinson med ny API + DST-temperatur (hvor tilgjengelig)
  3. Full refit av decline-modell (visc + premium + |premium|)
  4. Sammenligning av CV R², Aker BP RMSE, individuelle felt-residuals
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
    print(msg)
    lines.append(msg)

# Operator mapping
with open(GEO) as f:
    gj = json.load(f)
op_map = {feat["properties"].get("fldName"): (feat["properties"].get("cmpLongName") or "")
          for feat in gj["features"] if feat["properties"].get("fldName")}
akerbp_fields = {k for k, v in op_map.items() if "Aker BP" in v}

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")
master = pd.read_csv(DATA / "master_fluid_library.csv")

log("═" * 80)
log("REFIT DECLINE MODEL WITH ENRICHED FLUID DATA")
log("═" * 80)

# Beggs-Robinson — supports field-specific temperature
def beggs_robinson(api, T_F=194):
    """Dead-oil viscosity (cP) from API and reservoir temperature."""
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

def loo_cv(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    if len(y) < 10:
        return np.nan, len(y)
    loo = LeaveOneOut()
    lr = LinearRegression()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        lr.fit(X[tr], y[tr])
        preds[te] = lr.predict(X[te])
    cv_r2 = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
    return cv_r2, len(y)

# ═══════════════════════════════════════════════════════════════
# 2. PREPARE THREE VERSIONS
# ═══════════════════════════════════════════════════════════════

# Build base dataset with D_annual
df_base = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
df_base = df_base.rename(columns={"api_gravity": "api_old"})
df_base = df_base[["field", "api_old", "D_annual", "is_direct_assay"]]

# Merge with master (new API)
master_subset = master[["field", "api_gravity", "api_quality_tier", "api_confidence",
                        "reservoir_temp_c"]].copy()
master_subset = master_subset.rename(columns={"api_gravity": "api_new"})
df = df_base.merge(master_subset, on="field", how="left")

# Fields where we got new data
df["api_changed"] = (df.api_new.notna()) & (df.api_new != df.api_old)
log(f"\nFields with API change: {df.api_changed.sum()}/{len(df)}")

# Compute viscosity for three scenarios
df["visc_old"] = beggs_robinson(df.api_old.fillna(df.api_old.median()))
df["api_use"] = df.api_new.fillna(df.api_old)
df["visc_new_fixedT"] = beggs_robinson(df.api_use)

# With field-specific T where available
def field_visc_T(row):
    api = row.api_use
    T_c = row.reservoir_temp_c
    T_F = (T_c * 9/5 + 32) if (pd.notna(T_c) and T_c > 30) else 194
    return beggs_robinson(api, T_F)
df["visc_new_fieldT"] = df.apply(field_visc_T, axis=1)

df["ln_visc_old"] = np.log(df.visc_old)
df["ln_visc_new_fixedT"] = np.log(df.visc_new_fixedT)
df["ln_visc_new_fieldT"] = np.log(df.visc_new_fieldT)

# ═══════════════════════════════════════════════════════════════
# 3. COMPUTE PREMIUM FOR EACH VERSION
# ═══════════════════════════════════════════════════════════════
post = panel[panel.is_post_peak].copy()
post["prod_frac"] = post.oil_pct_peak / 100.0

def compute_premium(D_physics_by_field, post_data, window_months):
    """Compute 12-mnd premium for each field based on D_physics."""
    out = []
    for field, grp in post_data.groupby("field"):
        if field not in D_physics_by_field:
            continue
        D_phys = D_physics_by_field[field]
        grp = grp.sort_values("months_since_peak")
        grp = grp[grp.prod_frac > 0.01]
        if len(grp) < 12:
            continue
        t = grp.months_since_peak.values
        log_prem_all = np.log(grp.prod_frac.values) - np.log(np.exp(-D_phys / 12 * t))
        mask = t >= t.max() - window_months
        if mask.sum() < 6:
            continue
        out.append({"field": field, "premium": log_prem_all[mask].mean()})
    return pd.DataFrame(out)

# For each version: fit physics regression, compute premium, fit full model
versions = {
    "old": "ln_visc_old",
    "new_fixedT": "ln_visc_new_fixedT",
    "new_fieldT": "ln_visc_new_fieldT",
}

results = {}
for ver_name, visc_col in versions.items():
    # Physics regression
    valid = df.dropna(subset=[visc_col, "D_annual"])
    X_p = valid[[visc_col]].values
    y_p = valid["D_annual"].values
    lr_p = LinearRegression().fit(X_p, y_p)
    valid["D_physics"] = lr_p.predict(X_p)

    # Premium
    D_phys_map = valid.set_index("field")["D_physics"].to_dict()
    prem = compute_premium(D_phys_map, post, WINDOW_MONTHS)

    # Full model
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
    ak_mae = full[full.is_akerbp].resid.abs().mean()

    results[ver_name] = {
        "model": lr,
        "phys_model": lr_p,
        "in_sample_r2": lr.score(X, y),
        "cv_r2": cv_r2,
        "n": n,
        "rmse": rmse,
        "akbp_rmse": ak_rmse,
        "akbp_mae": ak_mae,
        "df": full,
        "coef": lr.coef_,
        "intercept": lr.intercept_,
        "phys_coef": lr_p.coef_[0],
        "phys_intercept": lr_p.intercept_,
    }

# ═══════════════════════════════════════════════════════════════
# 4. SAMMENLIGNING
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("SAMMENLIGNING: Gammel modell vs. Ny modell")
log("═" * 80)

log(f"\n{'Versjon':25s} {'CV R²':>8s} {'In-R²':>8s} {'RMSE':>8s} {'Aker BP RMSE':>14s} {'n':>5s}")
log("─" * 80)
for ver in ["old", "new_fixedT", "new_fieldT"]:
    r = results[ver]
    marker = "  ★" if r["cv_r2"] == max(results[v]["cv_r2"] for v in results) else ""
    log(f"  {ver:23s} {r['cv_r2']:7.3f}  {r['in_sample_r2']:7.3f}  {r['rmse']:7.4f}  "
        f"{r['akbp_rmse']:13.4f}  {r['n']:>4d}{marker}")

# Coefficient comparison
log(f"\n── Koeffisient-sammenligning ──")
log(f"  {'Parameter':22s} {'Old':>10s} {'New fixedT':>12s} {'New fieldT':>12s}")
log("─" * 70)
log(f"  {'Intercept':22s} {results['old']['intercept']:10.4f} "
    f"{results['new_fixedT']['intercept']:12.4f} {results['new_fieldT']['intercept']:12.4f}")
log(f"  {'ln(viscosity)':22s} {results['old']['coef'][0]:10.4f} "
    f"{results['new_fixedT']['coef'][0]:12.4f} {results['new_fieldT']['coef'][0]:12.4f}")
log(f"  {'premium (12m)':22s} {results['old']['coef'][1]:10.4f} "
    f"{results['new_fixedT']['coef'][1]:12.4f} {results['new_fieldT']['coef'][1]:12.4f}")
log(f"  {'|premium (12m)|':22s} {results['old']['coef'][2]:10.4f} "
    f"{results['new_fixedT']['coef'][2]:12.4f} {results['new_fieldT']['coef'][2]:12.4f}")

# Aker BP comparison
log(f"\n── Aker BP felt-for-felt: gammel vs ny prediksjon ──")
best_ver = "new_fieldT" if results["new_fieldT"]["cv_r2"] > results["new_fixedT"]["cv_r2"] else "new_fixedT"
old_df = results["old"]["df"]
new_df = results[best_ver]["df"]

old_ak = old_df[old_df.is_akerbp].set_index("field")
new_ak = new_df[new_df.is_akerbp].set_index("field")

common = sorted(set(old_ak.index) & set(new_ak.index))
log(f"\n{'Field':18s} {'D_act':>7s} {'D_old_pred':>10s} {'D_new_pred':>10s} {'Old miss':>9s} {'New miss':>9s}")
log("─" * 80)
for f in common:
    o = old_ak.loc[f]
    n = new_ak.loc[f]
    log(f"{f:18s} {o.D_annual:7.3f} {o.D_pred:10.3f} {n.D_pred:10.3f} "
        f"{o.resid:+8.3f} {n.resid:+8.3f}")

# ═══════════════════════════════════════════════════════════════
# 5. SAVE & VISUALIZE
# ═══════════════════════════════════════════════════════════════
# Save best model predictions
best_df = results[best_ver]["df"]
best_df[["field", "api_old", "api_use", "reservoir_temp_c", "D_annual", "D_pred",
        "resid", "premium", "abs_premium", "api_quality_tier", "api_confidence",
        "is_akerbp"]].to_csv(DATA / "predictions_enriched.csv", index=False)
log(f"\nSaved: {DATA / 'predictions_enriched.csv'}")

# Visualization
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f"Decline Model Refit med Reservoar-API\nGammel (blend) vs Ny (DST + operator + premium)",
             fontsize=15, fontweight="bold", y=1.0)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Panel 1: CV R² comparison
ax = fig.add_subplot(gs[0, 0])
vers = ["old", "new_fixedT", "new_fieldT"]
labels = ["Gammel\n(blend API)", "Ny API\n(fixed T=90°C)", "Ny API\n(field-specific T)"]
cv_vals = [results[v]["cv_r2"] for v in vers]
colors = ["#9E9E9E", "#1565C0", "#2E7D32"]
bars = ax.bar(range(3), cv_vals, color=colors, alpha=0.85)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("LOO Cross-validated R²")
ax.set_title("Modell-ytelse sammenligning", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(cv_vals) * 1.15)
for i, v in enumerate(cv_vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

# Panel 2: Aker BP RMSE comparison
ax = fig.add_subplot(gs[0, 1])
rmse_vals = [results[v]["akbp_rmse"] for v in vers]
bars = ax.bar(range(3), rmse_vals, color=colors, alpha=0.85)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Aker BP RMSE")
ax.set_title("Aker BP-prediksjon: feilreduksjon", fontsize=12, fontweight="bold")
for i, v in enumerate(rmse_vals):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

# Panel 3: API changes scatter
ax = fig.add_subplot(gs[0, 2])
plot_df = df.dropna(subset=["api_new"])
sc = ax.scatter(plot_df.api_old, plot_df.api_new,
               c=plot_df.api_changed.astype(int), s=50, alpha=0.7,
               cmap="RdYlGn", vmin=-0.2, vmax=1.2)
lims = [15, 60]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
# Annotate big changes
big = plot_df[(plot_df.api_new - plot_df.api_old).abs() > 5]
for _, r in big.iterrows():
    ax.annotate(r.field, (r.api_old, r.api_new), fontsize=6, alpha=0.85,
               xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Gammel API (blend)")
ax.set_ylabel("Ny API (reservoar)")
ax.set_title("API-endringer per felt", fontsize=12, fontweight="bold")

# Panel 4-5: Old vs New predicted-actual scatter
for i, ver in enumerate(["old", best_ver]):
    ax = fig.add_subplot(gs[1, i])
    r = results[ver]
    plot = r["df"]
    others = plot[~plot.is_akerbp]
    akbp = plot[plot.is_akerbp]
    ax.scatter(others.D_pred, others.D_annual, c="lightgray", s=35, alpha=0.5, label="Andre NCS")
    ax.scatter(akbp.D_pred, akbp.D_annual, c="#E91E63", s=70, alpha=0.85, label="Aker BP",
              edgecolors="white", lw=0.5)
    for _, row in akbp.iterrows():
        ax.annotate(row.field, (row.D_pred, row.D_annual), fontsize=6, alpha=0.85,
                   xytext=(4, 3), textcoords="offset points")
    lims = [-0.02, max(plot.D_annual.max(), plot.D_pred.max()) + 0.03]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
    label = "GAMMEL modell" if ver == "old" else "NY modell (reservoar-API)"
    ax.set_title(f"{label}\nCV R²={r['cv_r2']:.3f}  Aker BP RMSE={r['akbp_rmse']:.4f}",
                fontsize=11, fontweight="bold")
    ax.set_xlabel("Predikert D_annual")
    ax.set_ylabel("Faktisk D_annual")
    ax.legend(fontsize=8)

# Panel 6: Per-field Aker BP residual change
ax = fig.add_subplot(gs[1, 2])
ak_compare = pd.DataFrame({
    "field": common,
    "old_resid": [old_ak.loc[f].resid for f in common],
    "new_resid": [new_ak.loc[f].resid for f in common],
}).sort_values("old_resid")
y_pos = np.arange(len(ak_compare))
ax.barh(y_pos - 0.18, ak_compare.old_resid, 0.36, color="#9E9E9E", alpha=0.85, label="Gammel")
ax.barh(y_pos + 0.18, ak_compare.new_resid, 0.36, color="#2E7D32", alpha=0.85, label="Ny")
ax.set_yticks(y_pos)
ax.set_yticklabels(ak_compare.field, fontsize=8)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Residual (faktisk − predikert)")
ax.set_title("Aker BP: residual-endring per felt", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_refit_enriched_model.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_refit_enriched_model.png")

with open(RESULTS / "refit_enriched_summary.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved: refit_enriched_summary.txt")
