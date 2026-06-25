"""
Script 03: Regression analysis — oil quality → exponential decline rate D.

Uses peak-normalized D (annual) as dependent variable.
Cross-sectional: one D per field, regressed on quality features.

Outputs (in results/):
  - regression_results.txt   — full regression tables
  - coefficient_plot.png     — visual comparison of quality coefficients
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

summary = pd.read_csv(DATA / "field_summary.csv")
df = summary.dropna(subset=["D_annual", "api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct"]).copy()

QUALITY = ["api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct"]
SAVEKW = dict(bbox_inches="tight")

output_lines = []

def log(msg=""):
    print(msg)
    output_lines.append(msg)


def run_ols(y, X, title, label=None):
    """Run OLS with HC1 robust SE. Print results. Return model."""
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC1")

    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}")
    log(f"  N={model.nobs:.0f}, R²={model.rsquared:.4f}, Adj-R²={model.rsquared_adj:.4f}")
    log(f"  F={model.fvalue:.2f}, p(F)={model.f_pvalue:.4f}")
    log(f"{'─'*70}")
    log(f"  {'Variable':<28} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8} {'sig':>5}")
    log(f"{'─'*70}")

    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        log(f"  {var:<28} {coef:>10.6f} {se:>10.6f} {t:>8.3f} {p:>8.4f} {sig:>5}")

    log(f"{'─'*70}")
    return model


# ═══════════════════════════════════════════════════════════════════════════

log("DECLINE-QUALITY REGRESSION ANALYSIS (Peak-Normalized D)")
log(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
log(f"Fields with valid D: {len(df)}")
log(f"Dependent variable: D_annual (exponential decline constant, yr⁻¹)")
log(f"D > 0 = declining production. Higher D = faster decline.")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL A: Quality features only
# ═══════════════════════════════════════════════════════════════════════════

y = df["D_annual"]
X_a = df[QUALITY]
model_a = run_ols(y, X_a, "Model A: Quality features only")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL B: Quality + field controls
# ═══════════════════════════════════════════════════════════════════════════

X_b = df[QUALITY + ["field_age_mean", "oil_mean"]]
model_b = run_ols(y, X_b, "Model B: Quality + field age + avg production size")

# VIF
log("\n  VIF check (Model B):")
X_vif = sm.add_constant(X_b)
for i, col in enumerate(X_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    flag = " ⚠" if vif > 5 else ""
    log(f"    {col:<28} VIF={vif:.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL C: Quality + controls + area
# ═══════════════════════════════════════════════════════════════════════════

df["is_north_sea"] = (df.main_area == "North sea").astype(int)
X_c = df[QUALITY + ["field_age_mean", "oil_mean", "is_north_sea"]]
model_c = run_ols(y, X_c, "Model C: Quality + controls + area dummy")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL D: Parsimonious — only significant variables
# ═══════════════════════════════════════════════════════════════════════════

X_d = df[["api_gravity", "sulfur_pct", "oil_mean"]]
model_d = run_ols(y, X_d, "Model D: Parsimonious (API + sulfur + field size)")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL E: Standardized β for relative importance
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'='*70}")
log("  Standardized β (Model B specification)")
log(f"{'='*70}")

df_std = df.copy()
std_cols = QUALITY + ["field_age_mean", "oil_mean"]
for col in std_cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
df_std["D_annual_z"] = (df_std.D_annual - df_std.D_annual.mean()) / df_std.D_annual.std()

y_std = df_std["D_annual_z"]
X_std = df_std[std_cols]
model_std = run_ols(y_std, X_std, "Standardized β (all variables z-scored)")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL F: Log(D) for robustness (handles skewed D)
# ═══════════════════════════════════════════════════════════════════════════

df_log = df[df.D_annual > 0].copy()
df_log["ln_D"] = np.log(df_log.D_annual)

y_log = df_log["ln_D"]
X_log = df_log[QUALITY + ["field_age_mean", "oil_mean"]]
model_log = run_ols(y_log, X_log, "Model F: ln(D) — robustness (semi-elasticity)")

# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

X_bp = sm.add_constant(df[QUALITY + ["field_age_mean", "oil_mean"]])
bp_stat, bp_p, _, _ = het_breuschpagan(model_b.resid, X_bp)
log(f"\n  Breusch-Pagan (Model B): stat={bp_stat:.2f}, p={bp_p:.4f}")
log(f"  → {'Heteroskedasticity detected' if bp_p < 0.05 else 'No significant heteroskedasticity'}")

# Shapiro-Wilk on residuals
sw_stat, sw_p = stats.shapiro(model_b.resid)
log(f"  Shapiro-Wilk (residuals): W={sw_stat:.3f}, p={sw_p:.4f}")
log(f"  → {'Non-normal residuals' if sw_p < 0.05 else 'Residuals approximately normal'}")

# ═══════════════════════════════════════════════════════════════════════════
# Economic magnitude
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'='*70}")
log("  Economic interpretation (Model B)")
log(f"{'='*70}")

api_coef = model_b.params["api_gravity"]
sulfur_coef = model_b.params["sulfur_pct"]
api_iqr = df.api_gravity.quantile(0.75) - df.api_gravity.quantile(0.25)
sulfur_iqr = df.sulfur_pct.quantile(0.75) - df.sulfur_pct.quantile(0.25)

log(f"  API gravity (IQR = {api_iqr:.1f}°):")
log(f"    Moving from Q1→Q3 API changes D by {api_coef * api_iqr:+.4f}/yr")
log(f"    = {abs(api_coef * api_iqr / df.D_annual.median()) * 100:.1f}% of median D")

log(f"  Sulfur (IQR = {sulfur_iqr:.3f}%):")
log(f"    Moving from Q1→Q3 sulfur changes D by {sulfur_coef * sulfur_iqr:+.4f}/yr")
log(f"    = {abs(sulfur_coef * sulfur_iqr / df.D_annual.median()) * 100:.1f}% of median D")

med_D = df.D_annual.median()
log(f"\n  Median D = {med_D:.4f}/yr → half-life = {np.log(2)/med_D:.0f} months")
log(f"  10° higher API → D changes by {api_coef*10:+.4f}/yr → half-life changes by "
    f"{np.log(2)/(med_D+api_coef*10) - np.log(2)/med_D:+.0f} months")

# ═══════════════════════════════════════════════════════════════════════════
# Coefficient plot
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Model B coefficients with 95% CI
ax = axes[0]
vars_plot = QUALITY + ["field_age_mean", "oil_mean"]
coefs = [model_b.params.get(v, 0) for v in vars_plot]
ci_lo = [model_b.conf_int().loc[v, 0] for v in vars_plot]
ci_hi = [model_b.conf_int().loc[v, 1] for v in vars_plot]
pvals = [model_b.pvalues.get(v, 1) for v in vars_plot]
colors = ["#C62828" if p < 0.05 else "#FF8F00" if p < 0.1 else "#9E9E9E" for p in pvals]

y_pos = range(len(vars_plot))
ax.barh(y_pos, coefs, color=colors, alpha=0.7, edgecolor="white", height=0.6, zorder=2)
for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
    ax.plot([lo, hi], [i, i], color="#37474F", linewidth=2, zorder=3, solid_capstyle="round")

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_pos)
labels = ["API Gravity", "Sulfur %", "Pour Point", "Vacuum Resid %", "Field Age", "Avg Production"]
ax.set_yticklabels(labels)
ax.set_xlabel("Coefficient (effect on D, yr⁻¹)")
ax.set_title("Model B: Quality + Controls")

for i, (c, p) in enumerate(zip(coefs, pvals)):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.text(c + (0.001 if c >= 0 else -0.001), i, f"{c:.5f}{sig}",
            va="center", ha="left" if c >= 0 else "right", fontsize=7.5)

# Right: standardized β
ax = axes[1]
std_vars = QUALITY + ["field_age_mean", "oil_mean"]
coefs_std = [model_std.params.get(v, 0) for v in std_vars]
ci_lo_std = [model_std.conf_int().loc[v, 0] for v in std_vars]
ci_hi_std = [model_std.conf_int().loc[v, 1] for v in std_vars]
pvals_std = [model_std.pvalues.get(v, 1) for v in std_vars]
colors_std = ["#C62828" if p < 0.05 else "#FF8F00" if p < 0.1 else "#9E9E9E" for p in pvals_std]

y_pos = range(len(std_vars))
ax.barh(y_pos, coefs_std, color=colors_std, alpha=0.7, edgecolor="white", height=0.6, zorder=2)
for i, (lo, hi) in enumerate(zip(ci_lo_std, ci_hi_std)):
    ax.plot([lo, hi], [i, i], color="#37474F", linewidth=2, zorder=3, solid_capstyle="round")

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Standardized β (relative importance)")
ax.set_title("Standardized Coefficients")

for i, (c, p) in enumerate(zip(coefs_std, pvals_std)):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.text(c + (0.02 if c >= 0 else -0.02), i, f"{c:.3f}{sig}",
            va="center", ha="left" if c >= 0 else "right", fontsize=8)

fig.suptitle("Quality → Decline Rate D: Regression Coefficients (N=51 NCS fields)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "coefficient_plot.png", **SAVEKW)
plt.close()
log("\nSaved coefficient_plot.png")

# ═══════════════════════════════════════════════════════════════════════════
# Model comparison table
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'='*70}")
log("  Model comparison")
log(f"{'='*70}")
log(f"  {'Model':<35} {'N':>4} {'R²':>7} {'Adj-R²':>7} {'AIC':>8} {'BIC':>8}")
log(f"{'─'*70}")
for name, m in [("A: Quality only", model_a), ("B: + controls", model_b),
                ("C: + area", model_c), ("D: Parsimonious", model_d),
                ("F: ln(D) robustness", model_log)]:
    log(f"  {name:<35} {m.nobs:>4.0f} {m.rsquared:>7.4f} {m.rsquared_adj:>7.4f} {m.aic:>8.1f} {m.bic:>8.1f}")

# Save
with open(RESULTS / "regression_results.txt", "w") as f:
    f.write("\n".join(output_lines))

print(f"\nSaved regression_results.txt ({len(output_lines)} lines)")
