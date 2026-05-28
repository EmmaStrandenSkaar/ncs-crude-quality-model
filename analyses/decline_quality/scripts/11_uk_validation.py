"""
Script 11: Out-of-sample validation on UK Continental Shelf (UKCS).

Tests whether the momentum + physics model trained on NCS generalizes
to 270 UK fields — a completely independent dataset.

Strategy:
  1. Build UK panel: monthly production, peak normalization, D per window
  2. Compute API gravity from reported oil density (kg/m3 → SG → API)
  3. Apply Beggs-Robinson viscosity
  4. Train model on NCS, predict on UKCS (true out-of-sample)
  5. Also train on pooled NCS+UKCS for maximum power

Data: NSTA PPRS field production (downloaded from ArcGIS FeatureServer)

Outputs:
  - fig_uk_validation.png       — main results
  - uk_validation_results.txt   — tables
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
SAVEKW = dict(bbox_inches="tight")

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

def fit_D_window(grp, t_min, t_max, y_col="oil_pct_peak", min_obs=10):
    d = grp[(grp.months_since_peak >= t_min) & (grp.months_since_peak < t_max)]
    d = d[d[y_col] > 0].dropna(subset=["months_since_peak", y_col])
    if len(d) < min_obs:
        return np.nan
    t = d.months_since_peak.values
    ln_y = np.log(d[y_col].values)
    slope, _, _, _, _ = stats.linregress(t, ln_y)
    return -slope * 12

def loo_cv_r2(X, y):
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
    return 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2), len(y)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD NCS DATA (already processed)
# ═══════════════════════════════════════════════════════════════════════════

ncs_panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
ncs_summary = pd.read_csv(DATA / "field_summary.csv")

ncs_post = ncs_panel[ncs_panel.is_post_peak].copy()
D_12_ncs = ncs_post.groupby("field").apply(lambda g: fit_D_window(g, 0, 12)).rename("D_12")
D_12_24_ncs = ncs_post.groupby("field").apply(lambda g: fit_D_window(g, 12, 24)).rename("D_12_24")

ncs = ncs_summary.dropna(subset=["D_annual", "api_gravity"]).copy()
ncs = ncs.merge(D_12_ncs, on="field").merge(D_12_24_ncs, on="field", how="left")
ncs["ln_viscosity"] = np.log(beggs_robinson(ncs.api_gravity))
ncs["source"] = "NCS"
ncs = ncs.dropna(subset=["D_12", "D_annual", "ln_viscosity"])

log("NCS DATA")
log(f"  Fields: {len(ncs)}, D_annual range: {ncs.D_annual.min():.3f}–{ncs.D_annual.max():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# BUILD UK PANEL
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("BUILDING UK PANEL")
log(f"{'═'*65}")

uk_raw = pd.read_csv(RAW / "nsta_oil_production.csv")
uk_raw = uk_raw[uk_raw.OILPRODM3 > 0].copy()
uk_raw["PERIODYR"] = uk_raw["PERIODYR"].astype(int)
uk_raw["PERIODMNTH"] = uk_raw["PERIODMNTH"].astype(int)

# Compute API gravity from density
uk_raw = uk_raw[uk_raw.OILPRDDENS > 0].copy()
uk_raw["SG"] = uk_raw.OILPRDDENS / 999.012
uk_raw["api_gravity"] = 141.5 / uk_raw.SG - 131.5

# Filter unreasonable densities
uk_raw = uk_raw[(uk_raw.api_gravity >= 10) & (uk_raw.api_gravity <= 55)]

# Convert m3 to comparable units (Mill Sm3 for consistency)
uk_raw["oil_msm3"] = uk_raw.OILPRODM3 / 1e6

# Sort and compute peak + normalization per field
uk_fields = []
for field, grp in uk_raw.groupby("FIELDNAME"):
    grp = grp.sort_values(["PERIODYR", "PERIODMNTH"]).reset_index(drop=True)
    if len(grp) < 24:
        continue

    # Assign month index
    grp["month_idx"] = range(len(grp))

    # Find peak
    peak_idx = grp.oil_msm3.idxmax()
    peak_val = grp.loc[peak_idx, "oil_msm3"]
    peak_month = grp.loc[peak_idx, "month_idx"]

    if peak_val <= 0:
        continue

    grp["peak_oil"] = peak_val
    grp["oil_pct_peak"] = (grp.oil_msm3 / peak_val) * 100
    grp["months_since_peak"] = grp.month_idx - peak_month
    grp["is_post_peak"] = grp.months_since_peak > 0
    grp["field"] = field

    # Water cut
    if "WATPRODVOL" in grp.columns:
        total_fluid = grp.OILPRODM3 + grp.WATPRODVOL.fillna(0)
        grp["water_cut"] = grp.WATPRODVOL.fillna(0) / total_fluid.clip(lower=1)
    else:
        grp["water_cut"] = np.nan

    # GOR
    if "AGASPROKSM" in grp.columns:
        grp["gor"] = grp.AGASPROKSM.fillna(0) / grp.OILPRODM3.clip(lower=1)
    else:
        grp["gor"] = np.nan

    uk_fields.append(grp)

uk_panel = pd.concat(uk_fields, ignore_index=True)
log(f"  UK panel: {len(uk_panel)} records, {uk_panel.field.nunique()} fields")

# ═══════════════════════════════════════════════════════════════════════════
# FIT D_annual AND D_12 FOR UK FIELDS
# ═══════════════════════════════════════════════════════════════════════════

uk_post = uk_panel[uk_panel.is_post_peak].copy()

# Full-life D
uk_D_full = uk_post.groupby("field").apply(
    lambda g: fit_D_window(g, 0, 9999, min_obs=24)
).rename("D_annual")

# D from first 12 months
uk_D_12 = uk_post.groupby("field").apply(
    lambda g: fit_D_window(g, 0, 12)
).rename("D_12")

# D from months 12-24
uk_D_12_24 = uk_post.groupby("field").apply(
    lambda g: fit_D_window(g, 12, 24)
).rename("D_12_24")

# Field-level API gravity (median across all months)
uk_api = uk_panel.groupby("field")["api_gravity"].median().rename("api_gravity")
uk_oil_mean = uk_panel.groupby("field")["oil_msm3"].mean().rename("oil_mean")
uk_n_months = uk_post.groupby("field").size().rename("n_post_peak")

uk_summary = (uk_D_full.to_frame()
    .merge(uk_D_12.to_frame(), on="field")
    .merge(uk_D_12_24.to_frame(), on="field", how="left")
    .merge(uk_api.to_frame(), on="field")
    .merge(uk_oil_mean.to_frame(), on="field")
    .merge(uk_n_months.to_frame(), on="field"))

# Filter: positive D, enough data
uk = uk_summary.dropna(subset=["D_annual", "D_12", "api_gravity"]).copy()
uk = uk[(uk.D_annual > 0) & (uk.D_annual < 1)]
uk = uk[uk.n_post_peak >= 36]
uk["ln_viscosity"] = np.log(beggs_robinson(uk.api_gravity))
uk["source"] = "UKCS"

log(f"  UK fields for modeling: {len(uk)}")
log(f"  D_annual range: {uk.D_annual.min():.3f}–{uk.D_annual.max():.3f}")
log(f"  API gravity range: {uk.api_gravity.min():.1f}–{uk.api_gravity.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Train on NCS, predict on UKCS (true out-of-sample)
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 1: Train on NCS → Predict on UKCS (true out-of-sample)")
log(f"{'═'*65}")

model_specs = {
    "M1: D_12 only":             ["D_12"],
    "M2: D_12 + visc":           ["D_12", "ln_viscosity"],
    "M3: D_12 + visc + size":    ["D_12", "ln_viscosity", "oil_mean"],
}

for name, cols in model_specs.items():
    # Train on NCS
    ncs_sub = ncs.dropna(subset=cols + ["D_annual"])
    X_train = sm.add_constant(ncs_sub[cols])
    y_train = ncs_sub.D_annual
    m = sm.OLS(y_train, X_train).fit()

    # Predict on UK
    uk_sub = uk.dropna(subset=cols + ["D_annual"])
    X_test = sm.add_constant(uk_sub[cols])
    y_test = uk_sub.D_annual
    y_pred = m.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot

    r, p = stats.pearsonr(y_pred, y_test)
    mae = np.mean(np.abs(y_test - y_pred))

    log(f"\n  {name}")
    log(f"    Train: NCS (n={len(ncs_sub)}), Test: UKCS (n={len(uk_sub)})")
    log(f"    OOS R²={oos_r2:.3f}, r={r:+.3f} (p={p:.4f}), MAE={mae:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Models trained & tested within UKCS (LOO-CV)
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 2: Within-UKCS models (LOO-CV)")
log(f"{'═'*65}")

for name, cols in model_specs.items():
    uk_sub = uk.dropna(subset=cols + ["D_annual"])
    X = sm.add_constant(uk_sub[cols])
    m = sm.OLS(uk_sub.D_annual, X).fit(cov_type="HC1")
    cv_r2, n = loo_cv_r2(uk_sub[cols].values, uk_sub.D_annual.values)

    log(f"\n  {name} (n={n})")
    log(f"    R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, CV R²={cv_r2:.3f}")
    for v in cols:
        sig = "***" if m.pvalues[v] < 0.01 else "**" if m.pvalues[v] < 0.05 else "*" if m.pvalues[v] < 0.1 else ""
        log(f"      {v:<20s} β={m.params[v]:>10.5f} (p={m.pvalues[v]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Pooled NCS + UKCS
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 3: Pooled NCS + UKCS")
log(f"{'═'*65}")

common_cols = ["field", "D_annual", "D_12", "D_12_24", "ln_viscosity", "oil_mean",
               "api_gravity", "source"]
ncs_pool = ncs[["field", "D_annual", "D_12", "D_12_24", "ln_viscosity", "oil_mean",
                 "api_gravity", "source"]].copy()
uk_pool = uk.reset_index()[["field", "D_annual", "D_12", "D_12_24", "ln_viscosity",
                             "oil_mean", "api_gravity", "source"]].copy()
pooled = pd.concat([ncs_pool, uk_pool], ignore_index=True)
pooled = pooled.dropna(subset=["D_12", "D_annual", "ln_viscosity"])

log(f"  Pooled: {len(pooled)} fields (NCS={len(ncs_pool.dropna(subset=['D_12']))}, "
    f"UKCS={len(uk_pool.dropna(subset=['D_12']))})")

for name, cols in model_specs.items():
    sub = pooled.dropna(subset=cols + ["D_annual"])
    X = sm.add_constant(sub[cols])
    m = sm.OLS(sub.D_annual, X).fit(cov_type="HC1")
    cv_r2, n = loo_cv_r2(sub[cols].values, sub.D_annual.values)

    log(f"\n  {name} (n={n})")
    log(f"    R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, CV R²={cv_r2:.3f}")
    for v in cols:
        sig = "***" if m.pvalues[v] < 0.01 else "**" if m.pvalues[v] < 0.05 else "*" if m.pvalues[v] < 0.1 else ""
        log(f"      {v:<20s} β={m.params[v]:>10.5f} (p={m.pvalues[v]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1: NCS vs UK data comparison
ax = axes[0, 0]
ax.hist(ncs.D_annual, bins=20, alpha=0.6, color="#1565C0", label=f"NCS (n={len(ncs)})", density=True)
ax.hist(uk.D_annual, bins=30, alpha=0.6, color="#E65100", label=f"UKCS (n={len(uk)})", density=True)
ax.legend(fontsize=9)
style_ax(ax, "D_annual (yr⁻¹)", "Density", "Decline Rate Distribution")

# 2: API gravity comparison
ax = axes[0, 1]
ax.hist(ncs.api_gravity, bins=15, alpha=0.6, color="#1565C0", label="NCS", density=True)
ax.hist(uk.api_gravity, bins=25, alpha=0.6, color="#E65100", label="UKCS", density=True)
ax.legend(fontsize=9)
style_ax(ax, "API Gravity (°)", "Density", "Oil Quality Distribution")

# 3: Train NCS → Predict UKCS (best model: D_12 + visc)
ax = axes[0, 2]
best_cols = ["D_12", "ln_viscosity"]
ncs_sub = ncs.dropna(subset=best_cols + ["D_annual"])
uk_sub = uk.dropna(subset=best_cols + ["D_annual"])

X_train = sm.add_constant(ncs_sub[best_cols])
m_train = sm.OLS(ncs_sub.D_annual, X_train).fit()
X_test = sm.add_constant(uk_sub[best_cols])
y_pred_uk = m_train.predict(X_test)

ss_res = np.sum((uk_sub.D_annual - y_pred_uk) ** 2)
ss_tot = np.sum((uk_sub.D_annual - uk_sub.D_annual.mean()) ** 2)
oos_r2 = 1 - ss_res / ss_tot

ax.scatter(y_pred_uk, uk_sub.D_annual, c="#E65100", s=20, alpha=0.4, edgecolors="white", linewidths=0.3)
lims = [0, max(y_pred_uk.max(), uk_sub.D_annual.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(fontsize=8)
style_ax(ax, "Predicted D (yr⁻¹) [NCS model]", "Actual D (yr⁻¹) [UKCS]",
         f"Train NCS → Predict UKCS\n(OOS R² = {oos_r2:.3f})")

# 4: Pooled model — CV predictions
ax = axes[1, 0]
sub = pooled.dropna(subset=best_cols + ["D_annual"])
X_sk = sub[best_cols].values
y_sk = sub.D_annual.values
loo = LeaveOneOut()
lr = LinearRegression()
cv_preds = np.zeros(len(y_sk))
for tr, te in loo.split(X_sk):
    lr.fit(X_sk[tr], y_sk[tr])
    cv_preds[te] = lr.predict(X_sk[te])

cv_r2_pooled = 1 - np.sum((y_sk - cv_preds)**2) / np.sum((y_sk - y_sk.mean())**2)

ncs_mask = sub.source == "NCS"
ax.scatter(cv_preds[ncs_mask], y_sk[ncs_mask], c="#1565C0", s=30, alpha=0.5,
           edgecolors="white", linewidths=0.3, label=f"NCS (n={ncs_mask.sum()})")
ax.scatter(cv_preds[~ncs_mask], y_sk[~ncs_mask], c="#E65100", s=15, alpha=0.3,
           edgecolors="white", linewidths=0.3, label=f"UKCS (n={(~ncs_mask).sum()})")
lims = [0, max(cv_preds.max(), y_sk.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(fontsize=8)
style_ax(ax, "LOO-CV Predicted D (yr⁻¹)", "Actual D (yr⁻¹)",
         f"Pooled Model (CV R² = {cv_r2_pooled:.3f}, n={len(sub)})")

# 5: D_12 vs D_annual by shelf
ax = axes[1, 1]
ax.scatter(ncs.D_12, ncs.D_annual, c="#1565C0", s=40, alpha=0.6,
           edgecolors="white", linewidths=0.5, label="NCS", zorder=3)
ax.scatter(uk.D_12, uk.D_annual, c="#E65100", s=15, alpha=0.3,
           edgecolors="white", linewidths=0.3, label="UKCS", zorder=2)

r_ncs, _ = stats.pearsonr(ncs.D_12.dropna(), ncs.loc[ncs.D_12.notna(), "D_annual"])
r_uk, _ = stats.pearsonr(uk.D_12.dropna(), uk.loc[uk.D_12.notna(), "D_annual"])
ax.text(0.03, 0.97, f"NCS: r={r_ncs:+.3f}\nUKCS: r={r_uk:+.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.legend(fontsize=8)
style_ax(ax, "D_12 (yr⁻¹, first 12 months)", "D_annual (yr⁻¹, full life)",
         "Momentum Signal: NCS vs UKCS")

# 6: Summary comparison table as text
ax = axes[1, 2]
ax.axis("off")

summary_text = (
    "MODEL COMPARISON SUMMARY\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"{'Sample':<25s} {'n':>5s} {'CV R²':>7s}\n"
    f"{'─'*25} {'─'*5} {'─'*7}\n"
)

# Get key results
for label, dataset, cols in [
    ("NCS only", ncs, best_cols),
    ("UKCS only", uk, best_cols),
    ("Pooled NCS+UKCS", pooled, best_cols),
]:
    sub = dataset.dropna(subset=cols + ["D_annual"])
    cv, n = loo_cv_r2(sub[cols].values, sub.D_annual.values)
    summary_text += f"{label:<25s} {n:>5d} {cv:>+7.3f}\n"

summary_text += f"\n{'─'*25} {'─'*5} {'─'*7}\n"
summary_text += f"{'NCS→UKCS (true OOS)':<25s} {len(uk_sub):>5d} {oos_r2:>+7.3f}\n"
summary_text += "\nModel: D_12 + ln(viscosity)"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

fig.suptitle("Out-of-Sample Validation: NCS Model → UK Continental Shelf (270 fields)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_uk_validation.png", **SAVEKW)
plt.close()
log("\nSaved fig_uk_validation.png")

with open(RESULTS / "uk_validation_results.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved uk_validation_results.txt ({len(lines)} lines)")
