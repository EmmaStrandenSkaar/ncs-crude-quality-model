"""
Script 08: Improved physics model — add GOR, interactions, cross-validation.

Improvements over Script 07:
  1. GOR as proxy for drive mechanism (solution gas vs gas cap vs water drive)
  2. Interaction: viscosity × water cut rate (compounding effect)
  3. Leave-one-out cross-validation for honest R²
  4. Comparison: raw → physics → physics+GOR → full model

Outputs:
  - fig_improved_model.png         — model comparison + CV results
  - fig_gor_analysis.png           — GOR relationships
  - fig_model_diagnostics.png      — residuals, leverage, normality
  - improved_model_results.txt     — full regression tables
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
SAVEKW = dict(bbox_inches="tight")

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")
physics = pd.read_csv(RESULTS / "physics_model_results.csv")

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

# ── Prepare features ────────────────────────────────────────────────────────

T_F = 194
MU_WATER = 0.35

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

df = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
df["viscosity_cp"] = beggs_robinson(df.api_gravity)
df["ln_viscosity"] = np.log(df.viscosity_cp)
df["mobility_ratio"] = df.viscosity_cp / MU_WATER
df["ln_mobility"] = np.log(df.mobility_ratio)

# GOR features
df["ln_gor"] = np.log(df.gor_mean.clip(lower=0.001))
df["gor_class"] = pd.cut(df.gor_mean, bins=[0, 0.1, 0.3, 10],
                          labels=["Low GOR", "Medium GOR", "High GOR"])

# Water cut features from physics results
df = df.merge(physics[["field", "wc_k", "wc_t50", "wc_r2"]], on="field", how="left")

# Interaction terms
df["visc_x_wck"] = df.ln_viscosity * df.wc_k
df["visc_x_gor"] = df.ln_viscosity * df.ln_gor
df["wck_x_gor"] = df.wc_k * df.ln_gor

# Water cut at maturity (current state)
df["wc_current"] = df.water_cut_mean

# Complete cases for modeling
model_df = df.dropna(subset=["D_annual", "ln_viscosity", "wc_k", "ln_gor",
                              "oil_mean", "visc_x_wck"]).copy()

log("IMPROVED PHYSICS MODEL")
log(f"Fields: {len(model_df)}")
log(f"D_annual range: {model_df.D_annual.min():.4f} – {model_df.D_annual.max():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# GOR exploration
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("GOR ANALYSIS")
log(f"{'═'*65}")
log(f"GOR range: {model_df.gor_mean.min():.4f} – {model_df.gor_mean.max():.4f}")
log(f"GOR distribution: {model_df.gor_class.value_counts().to_dict()}")

for col, label in [("gor_mean", "GOR"), ("ln_gor", "ln(GOR)")]:
    r, p = stats.pearsonr(model_df[col], model_df.D_annual)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  {label:15s} vs D: r={r:+.3f} (p={p:.3f}) {sig}")

# GOR vs viscosity
r, p = stats.pearsonr(model_df.ln_gor, model_df.ln_viscosity)
log(f"  ln(GOR) vs ln(μ): r={r:+.3f} (p={p:.3f}) — {'correlated' if p < 0.05 else 'independent'}")

# ═══════════════════════════════════════════════════════════════════════════
# Model ladder — progressively adding physics features
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("MODEL LADDER: progressively adding physics")
log(f"{'═'*65}")

y = model_df.D_annual

specs = {
    "M0: Size only":
        ["oil_mean"],
    "M1: Raw quality":
        ["api_gravity", "sulfur_pct", "oil_mean"],
    "M2: Viscosity (Darcy)":
        ["ln_viscosity", "oil_mean"],
    "M3: + Water cut (B-L)":
        ["ln_viscosity", "wc_k", "oil_mean"],
    "M4: + GOR":
        ["ln_viscosity", "wc_k", "ln_gor", "oil_mean"],
    "M5: + Interaction μ×k":
        ["ln_viscosity", "wc_k", "ln_gor", "visc_x_wck", "oil_mean"],
    "M6: Full physics":
        ["ln_viscosity", "wc_k", "ln_gor", "visc_x_wck", "wc_current", "oil_mean"],
}

results = {}
for name, xcols in specs.items():
    X = sm.add_constant(model_df[xcols])
    m = sm.OLS(y, X).fit(cov_type="HC1")
    results[name] = {"model": m, "xcols": xcols}

    # LOO-CV R² (manual — collect all predictions, compute R² once)
    X_sk = model_df[xcols].values
    y_sk = y.values
    loo = LeaveOneOut()
    lr = LinearRegression()
    cv_preds = np.zeros(len(y_sk))
    for train_idx, test_idx in loo.split(X_sk):
        lr.fit(X_sk[train_idx], y_sk[train_idx])
        cv_preds[test_idx] = lr.predict(X_sk[test_idx])
    ss_res = np.sum((y_sk - cv_preds) ** 2)
    ss_tot = np.sum((y_sk - y_sk.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot
    results[name]["cv_r2"] = cv_r2

    n_vars = len(xcols)
    log(f"\n  {name}")
    log(f"    R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, LOO-CV R²={cv_r2:.3f}, "
        f"AIC={m.aic:.1f}, k={n_vars}")
    for var in xcols:
        sig = "***" if m.pvalues[var] < 0.01 else "**" if m.pvalues[var] < 0.05 else "*" if m.pvalues[var] < 0.1 else ""
        log(f"      {var:<20s} β={m.params[var]:>10.5f} (p={m.pvalues[var]:.3f}) {sig}")

# Best model by CV
best_name = max(results, key=lambda k: results[k]["cv_r2"])
best = results[best_name]
log(f"\n  ★ Best by LOO-CV: {best_name} (CV R²={best['cv_r2']:.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# Standardized β for best model
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log(f"STANDARDIZED β — {best_name}")
log(f"{'═'*65}")

best_cols = best["xcols"]
df_std = model_df.copy()
for col in best_cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
df_std["D_z"] = (df_std.D_annual - df_std.D_annual.mean()) / df_std.D_annual.std()

X_std = sm.add_constant(df_std[best_cols])
m_std = sm.OLS(df_std.D_z, X_std).fit(cov_type="HC1")

for var in best_cols:
    sig = "***" if m_std.pvalues[var] < 0.01 else "**" if m_std.pvalues[var] < 0.05 else "*" if m_std.pvalues[var] < 0.1 else ""
    log(f"  {var:<20s} β={m_std.params[var]:>+7.3f} (p={m_std.pvalues[var]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Model ladder + CV comparison
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1a: R² ladder
ax = axes[0]
names = list(results.keys())
r2 = [results[n]["model"].rsquared for n in names]
adj_r2 = [results[n]["model"].rsquared_adj for n in names]
cv_r2 = [results[n]["cv_r2"] for n in names]

x = np.arange(len(names))
w = 0.25
ax.bar(x - w, r2, w, color="#1565C0", alpha=0.8, label="R²", edgecolor="white")
ax.bar(x, adj_r2, w, color="#42A5F5", alpha=0.8, label="Adj R²", edgecolor="white")
ax.bar(x + w, cv_r2, w, color="#E65100", alpha=0.8, label="LOO-CV R²", edgecolor="white")

for i, (r, ar, cr) in enumerate(zip(r2, adj_r2, cv_r2)):
    ax.text(i - w, r + 0.005, f"{r:.3f}", ha="center", fontsize=6, rotation=45)
    ax.text(i + w, max(cr, 0) + 0.005, f"{cr:.3f}", ha="center", fontsize=6, rotation=45, color="#E65100")

short_names = [n.split(": ")[1] if ": " in n else n for n in names]
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=7, rotation=25, ha="right")
ax.set_ylabel("R²")
ax.set_title("Model Ladder: Progressive Physics Features")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, axis="y")

# 1b: Predicted vs actual for best model
ax = axes[1]
best_m = best["model"]
y_pred = best_m.predict(sm.add_constant(model_df[best_cols]))

ax.scatter(y_pred, y, c="#1565C0", s=50, alpha=0.6, edgecolors="white")
lims = [0, max(y_pred.max(), y.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect")
ax.set_xlabel("Predicted D (yr⁻¹)")
ax.set_ylabel("Actual D (yr⁻¹)")
ax.set_title(f"Best Model: Predicted vs Actual\n{best_name} (CV R²={best['cv_r2']:.3f})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlim(lims)
ax.set_ylim(lims)

for _, row in model_df.nlargest(3, "D_annual").iterrows():
    idx = model_df.index.get_loc(row.name)
    ax.annotate(row.field, (y_pred.iloc[idx], row.D_annual),
                fontsize=7, alpha=0.6, xytext=(5, 5), textcoords="offset points")

# 1c: Standardized β for best model
ax = axes[2]
std_coefs = [m_std.params[v] for v in best_cols]
std_ci_lo = [m_std.conf_int().loc[v, 0] for v in best_cols]
std_ci_hi = [m_std.conf_int().loc[v, 1] for v in best_cols]
std_pvals = [m_std.pvalues[v] for v in best_cols]

label_map = {
    "ln_viscosity": "ln(Viscosity)\n[Darcy's Law]",
    "wc_k": "Water Cut Rate\n[Buckley-Leverett]",
    "ln_gor": "ln(GOR)\n[Drive mechanism]",
    "visc_x_wck": "Viscosity × WC Rate\n[Interaction]",
    "wc_current": "Current Water Cut\n[Maturity]",
    "oil_mean": "Field Size\n[Infrastructure]",
}
ylabels = [label_map.get(v, v) for v in best_cols]
colors = ["#C62828" if p < 0.05 else "#FF8F00" if p < 0.1 else "#9E9E9E" for p in std_pvals]

y_pos = range(len(best_cols))
ax.barh(y_pos, std_coefs, color=colors, alpha=0.7, edgecolor="white", height=0.6, zorder=2)
for i, (lo, hi) in enumerate(zip(std_ci_lo, std_ci_hi)):
    ax.plot([lo, hi], [i, i], color="#37474F", linewidth=2, zorder=3)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_xlabel("Standardized β")
ax.set_title("Relative Importance (z-scored)")
ax.grid(True, alpha=0.2, axis="x")

for i, (c, p) in enumerate(zip(std_coefs, std_pvals)):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.text(c + (0.02 if c >= 0 else -0.02), i, f"{c:+.3f}{sig}",
            va="center", ha="left" if c >= 0 else "right", fontsize=8)

fig.suptitle("Improved Physics Model — Progressive Feature Addition with Cross-Validation",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_improved_model.png", **SAVEKW)
plt.close()
log("\nSaved fig_improved_model.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: GOR analysis
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 2a: GOR vs D
ax = axes[0]
for cls, color in [("Low GOR", "#2196F3"), ("Medium GOR", "#FF9800"), ("High GOR", "#4CAF50")]:
    mask = model_df.gor_class == cls
    d = model_df[mask]
    if len(d) > 0:
        ax.scatter(d.gor_mean, d.D_annual, c=color, s=50, alpha=0.6,
                   edgecolors="white", label=f"{cls} (n={len(d)})")

r, p = stats.pearsonr(model_df.gor_mean, model_df.D_annual)
sig = "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Gas-Oil Ratio (GOR)")
ax.set_ylabel("Annual Decline Rate D (yr⁻¹)")
ax.set_title("GOR vs. Decline Rate")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 2b: GOR vs viscosity (are they independent?)
ax = axes[1]
ax.scatter(model_df.viscosity_cp, model_df.gor_mean, c="#1565C0", s=50, alpha=0.6, edgecolors="white")
r, p = stats.pearsonr(model_df.ln_viscosity, model_df.ln_gor)
sig = "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Oil Viscosity (cp)")
ax.set_ylabel("GOR")
ax.set_title("Viscosity vs. GOR\n(independent = good for model)")
ax.set_xscale("log")
ax.grid(True, alpha=0.2)

# 2c: D by GOR class (box plot)
ax = axes[2]
gor_groups = {}
for cls in ["Low GOR", "Medium GOR", "High GOR"]:
    vals = model_df.loc[model_df.gor_class == cls, "D_annual"].dropna()
    if len(vals) > 1:
        gor_groups[cls] = vals

if gor_groups:
    bp = ax.boxplot(gor_groups.values(), labels=gor_groups.keys(), patch_artist=True, widths=0.5)
    box_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for patch, color in zip(bp["boxes"], box_colors[:len(gor_groups)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

ax.set_ylabel("Annual Decline Rate D (yr⁻¹)")
ax.set_title("Decline Rate by GOR Class")
ax.grid(True, alpha=0.2)

fig.suptitle("Gas-Oil Ratio (GOR) — Drive Mechanism Analysis",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_gor_analysis.png", **SAVEKW)
plt.close()
log("Saved fig_gor_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Model diagnostics
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

best_m = best["model"]
resid = best_m.resid
fitted = best_m.fittedvalues

# 3a: Residuals vs fitted
ax = axes[0, 0]
ax.scatter(fitted, resid, c="#1565C0", s=40, alpha=0.6, edgecolors="white")
ax.axhline(0, color="red", linewidth=1)
lowess = sm.nonparametric.lowess(resid, fitted, frac=0.5)
ax.plot(lowess[:, 0], lowess[:, 1], color="#C62828", linewidth=2, label="LOWESS")
ax.set_xlabel("Fitted D (yr⁻¹)")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs. Fitted")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 3b: Q-Q plot
ax = axes[0, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
ax.scatter(osm, osr, c="#1565C0", s=30, alpha=0.6, edgecolors="white")
ax.plot(osm, intercept + slope * np.array(osm), "r-", linewidth=1.5)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title(f"Normal Q-Q Plot (Shapiro p={stats.shapiro(resid)[1]:.3f})")
ax.grid(True, alpha=0.2)

# 3c: Residuals by field (ordered)
ax = axes[1, 0]
resid_df = pd.DataFrame({"field": model_df.field.values, "resid": resid.values})
resid_df = resid_df.sort_values("resid")
colors = ["#C62828" if r < -0.05 else "#2E7D32" if r > 0.05 else "#9E9E9E" for r in resid_df.resid]
ax.barh(range(len(resid_df)), resid_df.resid, color=colors, alpha=0.7, edgecolor="white", height=0.7)
ax.set_yticks(range(len(resid_df)))
ax.set_yticklabels(resid_df.field, fontsize=5)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Residual (actual D − predicted D)")
ax.set_title("Per-Field Residuals")
ax.grid(True, alpha=0.2, axis="x")

# 3d: LOO-CV predictions vs actual
ax = axes[1, 1]
loo = LeaveOneOut()
lr = LinearRegression()
X_sk = model_df[best_cols].values
y_sk = y.values
cv_preds = np.zeros_like(y_sk, dtype=float)
for train_idx, test_idx in loo.split(X_sk):
    lr.fit(X_sk[train_idx], y_sk[train_idx])
    cv_preds[test_idx] = lr.predict(X_sk[test_idx])

ax.scatter(cv_preds, y_sk, c="#E65100", s=50, alpha=0.6, edgecolors="white")
lims = [0, max(cv_preds.max(), y_sk.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
cv_r2_final = 1 - np.sum((y_sk - cv_preds)**2) / np.sum((y_sk - y_sk.mean())**2)
ax.set_xlabel("LOO-CV Predicted D (yr⁻¹)")
ax.set_ylabel("Actual D (yr⁻¹)")
ax.set_title(f"Leave-One-Out Cross-Validation\n(CV R² = {cv_r2_final:.3f})")
ax.grid(True, alpha=0.2)
ax.set_xlim(lims)
ax.set_ylim(lims)

fig.suptitle(f"Model Diagnostics — {best_name}",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_model_diagnostics.png", **SAVEKW)
plt.close()
log("Saved fig_model_diagnostics.png")

# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("MODEL COMPARISON SUMMARY")
log(f"{'═'*65}")
log(f"  {'Model':<30s} {'R²':>6s} {'AdjR²':>6s} {'CV-R²':>6s} {'AIC':>7s} {'k':>3s}")
log(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*3}")
for name, res in results.items():
    m = res["model"]
    cv = res["cv_r2"]
    k = len(res["xcols"])
    marker = " ★" if name == best_name else ""
    log(f"  {name:<30s} {m.rsquared:>6.3f} {m.rsquared_adj:>6.3f} {cv:>6.3f} {m.aic:>7.1f} {k:>3d}{marker}")

log(f"\n  ★ = best by LOO cross-validation")
log(f"  CV R² penalizes overfitting — it's the most honest metric")

with open(RESULTS / "improved_model_results.txt", "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved improved_model_results.txt ({len(lines)} lines)")
