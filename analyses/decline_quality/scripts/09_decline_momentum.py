"""
Script 09: Decline momentum — can early decline predict late decline?

Splits post-peak into early (first 24 months) and late (months 25+).
Tests whether D_early + physics (viscosity) predicts D_late better
than either alone. Also measures decline acceleration.

Outputs:
  - fig_momentum.png              — main results (4 panels)
  - fig_momentum_diagnostics.png  — per-field residuals, CV
  - momentum_results.txt          — tables
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
SAVEKW = dict(bbox_inches="tight")

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

# ── Beggs-Robinson viscosity ────────────────────────────────────────────────

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

# ── Fit exponential D on a window ───────────────────────────────────────────

def fit_D(grp, t_col="months_since_peak", y_col="oil_pct_peak", min_obs=12):
    """Fit ln(y) = a - D*t, return D_monthly, D_annual, r2, n_obs."""
    d = grp[[t_col, y_col]].dropna()
    d = d[d[y_col] > 0]
    if len(d) < min_obs:
        return pd.Series({"D_monthly": np.nan, "D_annual": np.nan,
                           "D_r2": np.nan, "n_obs": len(d)})
    t = d[t_col].values
    ln_y = np.log(d[y_col].values)
    slope, intercept, r, p, se = stats.linregress(t, ln_y)
    D_monthly = -slope
    D_annual = D_monthly * 12
    return pd.Series({"D_monthly": D_monthly, "D_annual": D_annual,
                       "D_r2": r**2, "n_obs": len(d)})

# ═══════════════════════════════════════════════════════════════════════════
# Split post-peak into early and late windows
# ═══════════════════════════════════════════════════════════════════════════

EARLY_CUTOFF = 24  # months
LATE_START = 24
MIN_LATE_MONTHS = 36  # need enough late data for reliable D_late

post = panel[panel.is_post_peak].copy()

early = post[post.months_since_peak <= EARLY_CUTOFF]
late = post[post.months_since_peak > LATE_START]

log("DECLINE MOMENTUM ANALYSIS")
log(f"Early window: months 0–{EARLY_CUTOFF} post-peak")
log(f"Late window: months {LATE_START}+ post-peak")
log(f"Min late observations: {MIN_LATE_MONTHS}")

# Fit D on each window
D_early = early.groupby("field").apply(fit_D, min_obs=12).reset_index()
D_early.columns = ["field", "D_early_mo", "D_early", "D_early_r2", "n_early"]

D_late = late.groupby("field").apply(fit_D, min_obs=MIN_LATE_MONTHS).reset_index()
D_late.columns = ["field", "D_late_mo", "D_late", "D_late_r2", "n_late"]

# Merge
df = D_early.merge(D_late, on="field")
df = df.merge(summary[["field", "api_gravity", "sulfur_pct", "oil_mean",
                         "main_area", "D_annual", "gor_mean", "water_cut_mean"]],
              on="field")

df = df.dropna(subset=["D_early", "D_late", "api_gravity"])
df["viscosity_cp"] = beggs_robinson(df.api_gravity)
df["ln_viscosity"] = np.log(df.viscosity_cp)

# Decline acceleration: is late decline faster or slower than early?
df["D_accel"] = df.D_late - df.D_early
df["D_ratio"] = df.D_late / df.D_early.clip(lower=0.001)

log(f"\nFields with both windows: {len(df)}")
log(f"D_early range: {df.D_early.min():.4f} – {df.D_early.max():.4f}")
log(f"D_late range:  {df.D_late.min():.4f} – {df.D_late.max():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Correlation analysis
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("CORRELATION ANALYSIS")
log(f"{'═'*65}")

pairs = [
    ("D_early", "D_late", "D_early → D_late (momentum)"),
    ("D_early", "D_annual", "D_early → D_full (persistence)"),
    ("ln_viscosity", "D_late", "Viscosity → D_late"),
    ("oil_mean", "D_late", "Field size → D_late"),
    ("D_accel", "ln_viscosity", "Acceleration vs viscosity"),
]

for x_col, y_col, label in pairs:
    r, p = stats.pearsonr(df[x_col], df[y_col])
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  {label:40s} r={r:+.3f} (p={p:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# Model ladder — predicting D_late
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("MODEL LADDER: predicting D_late (months 25+ post-peak)")
log(f"{'═'*65}")

y = df.D_late

specs = {
    "M0: Size only":
        ["oil_mean"],
    "M1: Physics (viscosity)":
        ["ln_viscosity", "oil_mean"],
    "M2: Momentum only":
        ["D_early"],
    "M3: Momentum + size":
        ["D_early", "oil_mean"],
    "M4: Momentum + physics":
        ["D_early", "ln_viscosity", "oil_mean"],
}

results = {}
for name, xcols in specs.items():
    sub = df.dropna(subset=xcols + ["D_late"])
    X = sm.add_constant(sub[xcols])
    y_sub = sub.D_late
    m = sm.OLS(y_sub, X).fit(cov_type="HC1")

    # LOO-CV
    X_sk = sub[xcols].values
    y_sk = y_sub.values
    loo = LeaveOneOut()
    lr = LinearRegression()
    cv_preds = np.zeros(len(y_sk))
    for train_idx, test_idx in loo.split(X_sk):
        lr.fit(X_sk[train_idx], y_sk[train_idx])
        cv_preds[test_idx] = lr.predict(X_sk[test_idx])
    ss_res = np.sum((y_sk - cv_preds) ** 2)
    ss_tot = np.sum((y_sk - y_sk.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot

    results[name] = {"model": m, "xcols": xcols, "cv_r2": cv_r2,
                      "cv_preds": cv_preds, "y_actual": y_sk, "n": len(sub)}

    log(f"\n  {name} (n={len(sub)})")
    log(f"    R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, "
        f"LOO-CV R²={cv_r2:.3f}, AIC={m.aic:.1f}")
    for var in xcols:
        sig = "***" if m.pvalues[var] < 0.01 else "**" if m.pvalues[var] < 0.05 else "*" if m.pvalues[var] < 0.1 else ""
        log(f"      {var:<20s} β={m.params[var]:>10.5f} (p={m.pvalues[var]:.3f}) {sig}")

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
sub = df.dropna(subset=best_cols + ["D_late"]).copy()
df_std = sub.copy()
for col in best_cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
df_std["D_z"] = (df_std.D_late - df_std.D_late.mean()) / df_std.D_late.std()

X_std = sm.add_constant(df_std[best_cols])
m_std = sm.OLS(df_std.D_z, X_std).fit(cov_type="HC1")

for var in best_cols:
    sig = "***" if m_std.pvalues[var] < 0.01 else "**" if m_std.pvalues[var] < 0.05 else "*" if m_std.pvalues[var] < 0.1 else ""
    log(f"  {var:<20s} β={m_std.params[var]:>+7.3f} (p={m_std.pvalues[var]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# Decline acceleration analysis
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("DECLINE ACCELERATION")
log(f"{'═'*65}")

accel = df.D_accel
log(f"  Mean acceleration: {accel.mean():+.4f} yr⁻¹ "
    f"({'decelerating' if accel.mean() < 0 else 'accelerating'})")
log(f"  Median: {accel.median():+.4f}")
log(f"  Fields accelerating: {(accel > 0).sum()}/{len(accel)} "
    f"({(accel > 0).mean()*100:.0f}%)")
log(f"  Fields decelerating: {(accel < 0).sum()}/{len(accel)} "
    f"({(accel < 0).mean()*100:.0f}%)")

t_stat, p_val = stats.ttest_1samp(accel, 0)
log(f"  t-test (H0: no acceleration): t={t_stat:.2f}, p={p_val:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# PRACTICAL MODEL: predict D_annual from first 24 months + physics
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("PRACTICAL MODEL: predict full-life D_annual from early observations")
log(f"{'═'*65}")

y_annual = df.D_annual

specs_annual = {
    "A0: Size only":
        ["oil_mean"],
    "A1: Physics":
        ["ln_viscosity", "oil_mean"],
    "A2: Momentum only":
        ["D_early"],
    "A3: Momentum + size":
        ["D_early", "oil_mean"],
    "A4: Momentum + physics":
        ["D_early", "ln_viscosity", "oil_mean"],
}

results_annual = {}
for name, xcols in specs_annual.items():
    sub = df.dropna(subset=xcols + ["D_annual"])
    X = sm.add_constant(sub[xcols])
    y_sub = sub.D_annual
    m = sm.OLS(y_sub, X).fit(cov_type="HC1")

    X_sk = sub[xcols].values
    y_sk = y_sub.values
    loo = LeaveOneOut()
    lr = LinearRegression()
    cv_preds = np.zeros(len(y_sk))
    for train_idx, test_idx in loo.split(X_sk):
        lr.fit(X_sk[train_idx], y_sk[train_idx])
        cv_preds[test_idx] = lr.predict(X_sk[test_idx])
    ss_res = np.sum((y_sk - cv_preds) ** 2)
    ss_tot = np.sum((y_sk - y_sk.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot

    results_annual[name] = {"model": m, "xcols": xcols, "cv_r2": cv_r2,
                             "cv_preds": cv_preds, "y_actual": y_sk, "n": len(sub)}

    log(f"\n  {name} (n={len(sub)})")
    log(f"    R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, "
        f"LOO-CV R²={cv_r2:.3f}, AIC={m.aic:.1f}")
    for var in xcols:
        sig = "***" if m.pvalues[var] < 0.01 else "**" if m.pvalues[var] < 0.05 else "*" if m.pvalues[var] < 0.1 else ""
        log(f"      {var:<20s} β={m.params[var]:>10.5f} (p={m.pvalues[var]:.3f}) {sig}")

best_annual_name = max(results_annual, key=lambda k: results_annual[k]["cv_r2"])
best_annual = results_annual[best_annual_name]
log(f"\n  ★ Best by LOO-CV: {best_annual_name} (CV R²={best_annual['cv_r2']:.3f})")

# Standardized β for best annual model
log(f"\n  Standardized β:")
ba_cols = best_annual["xcols"]
sub = df.dropna(subset=ba_cols + ["D_annual"]).copy()
df_std2 = sub.copy()
for col in ba_cols:
    df_std2[col] = (df_std2[col] - df_std2[col].mean()) / df_std2[col].std()
df_std2["D_z"] = (df_std2.D_annual - df_std2.D_annual.mean()) / df_std2.D_annual.std()
X_std2 = sm.add_constant(df_std2[ba_cols])
m_std2 = sm.OLS(df_std2.D_z, X_std2).fit(cov_type="HC1")
for var in ba_cols:
    sig = "***" if m_std2.pvalues[var] < 0.01 else "**" if m_std2.pvalues[var] < 0.05 else "*" if m_std2.pvalues[var] < 0.1 else ""
    log(f"    {var:<20s} β={m_std2.params[var]:>+7.3f} (p={m_std2.pvalues[var]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main momentum results (4 panels)
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {"fit": "#455A64", "north": "#1565C0", "norwegian": "#EF6C00",
           "barents": "#2E7D32", "accel": "#C62828", "decel": "#2E7D32"}

area_map = {
    "North sea": ("North Sea", COLORS["north"], "o"),
    "Norwegian sea": ("Norwegian Sea", COLORS["norwegian"], "s"),
    "Barents sea": ("Barents Sea", COLORS["barents"], "D"),
}

def style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: D_early vs D_late — the momentum signal
ax = axes[0, 0]
for area, (label, color, marker) in area_map.items():
    mask = df.main_area == area
    d = df[mask]
    if len(d) > 0:
        ax.scatter(d.D_early, d.D_late, c=color, marker=marker, s=50,
                   alpha=0.7, edgecolors="white", linewidths=0.5, label=label)

r, p = stats.pearsonr(df.D_early, df.D_late)
x_fit = np.linspace(df.D_early.min(), df.D_early.max(), 100)
slope, intercept, _, _, _ = stats.linregress(df.D_early, df.D_late)
ax.plot(x_fit, intercept + slope * x_fit, color=COLORS["fit"], linewidth=2)

sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})\nslope = {slope:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDBDBD", alpha=0.9))

ax.plot([0, 0.5], [0, 0.5], "k:", alpha=0.3, linewidth=1, label="1:1 line")
ax.legend(fontsize=7, loc="lower right")
style_ax(ax, "D_early (yr⁻¹, months 0–24)", "D_late (yr⁻¹, months 25+)",
         "Decline Momentum: Early → Late")

for _, row in pd.concat([df.nlargest(2, "D_late"), df.nsmallest(2, "D_late")]).drop_duplicates().iterrows():
    ax.annotate(row.field, (row.D_early, row.D_late),
                fontsize=6, alpha=0.5, xytext=(5, 3), textcoords="offset points")

# Panel 2: Model comparison — D_annual (practical)
ax = axes[0, 1]
names_a = list(results_annual.keys())
r2_a = [results_annual[n]["model"].rsquared for n in names_a]
adj_r2_a = [results_annual[n]["model"].rsquared_adj for n in names_a]
cv_r2_a = [results_annual[n]["cv_r2"] for n in names_a]

x = np.arange(len(names_a))
w = 0.25
ax.bar(x - w, r2_a, w, color="#1565C0", alpha=0.8, label="R²", edgecolor="white")
ax.bar(x, adj_r2_a, w, color="#42A5F5", alpha=0.8, label="Adj R²", edgecolor="white")
ax.bar(x + w, cv_r2_a, w, color="#E65100", alpha=0.8, label="LOO-CV R²", edgecolor="white")

for i, cr in enumerate(cv_r2_a):
    ax.text(i + w, max(cr, 0) + 0.01, f"{cr:.3f}", ha="center", fontsize=7,
            color="#E65100", fontweight="bold")

short = [n.split(": ")[1] if ": " in n else n for n in names_a]
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=7, rotation=25, ha="right")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(fontsize=7)
style_ax(ax, "", "R²", "Predict D_annual from First 24 Months")

# Panel 3: Decline acceleration histogram
ax = axes[1, 0]
accel_colors = [COLORS["accel"] if a > 0 else COLORS["decel"] for a in df.D_accel.sort_values()]
ax.barh(range(len(df)), df.D_accel.sort_values().values, color=accel_colors,
        alpha=0.7, edgecolor="white", height=0.7)

sorted_df = df.sort_values("D_accel")
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df.field.values, fontsize=5)
ax.axvline(0, color="black", linewidth=0.8)
style_ax(ax, "D_late − D_early (yr⁻¹)", "",
         "Decline Acceleration by Field")
ax.text(0.97, 0.97, f"Accelerating: {(df.D_accel > 0).sum()}\nDecelerating: {(df.D_accel < 0).sum()}",
        transform=ax.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Panel 4: Predicted vs actual for best annual model (CV predictions)
ax = axes[1, 1]
cv_preds_a = best_annual["cv_preds"]
y_actual_a = best_annual["y_actual"]
ax.scatter(cv_preds_a, y_actual_a, c="#E65100", s=50, alpha=0.6, edgecolors="white")
lims = [0, max(cv_preds_a.max(), y_actual_a.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(fontsize=8)
style_ax(ax, "LOO-CV Predicted D_annual (yr⁻¹)", "Actual D_annual (yr⁻¹)",
         f"CV: {best_annual_name}\n(CV R² = {best_annual['cv_r2']:.3f})")

fig.suptitle("Decline Momentum — Can Early Decline Predict Late Decline?",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_momentum.png", **SAVEKW)
plt.close()
log("\nSaved fig_momentum.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Diagnostics for best model
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

best_m = best["model"]
resid = best_m.resid
fitted = best_m.fittedvalues

# Residuals vs fitted
ax = axes[0]
ax.scatter(fitted, resid, c="#1565C0", s=40, alpha=0.6, edgecolors="white")
ax.axhline(0, color="red", linewidth=1)
lowess = sm.nonparametric.lowess(resid, fitted, frac=0.5)
ax.plot(lowess[:, 0], lowess[:, 1], color="#C62828", linewidth=2, label="LOWESS")
ax.legend(fontsize=8)
style_ax(ax, "Fitted D_late", "Residual", "Residuals vs. Fitted")

# Q-Q
ax = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
ax.scatter(osm, osr, c="#1565C0", s=30, alpha=0.6, edgecolors="white")
ax.plot(osm, intercept + slope * np.array(osm), "r-", linewidth=1.5)
shapiro_p = stats.shapiro(resid)[1]
style_ax(ax, "Theoretical Quantiles", "Sample Quantiles",
         f"Q-Q Plot (Shapiro p={shapiro_p:.3f})")

# Viscosity adds value beyond momentum?
ax = axes[2]
sub = df.dropna(subset=["D_early", "ln_viscosity", "D_late"])
X_mom = sm.add_constant(sub[["D_early"]])
resid_visc = sm.OLS(sub.ln_viscosity, X_mom).fit().resid
resid_D = sm.OLS(sub.D_late, X_mom).fit().resid

ax.scatter(resid_visc, resid_D, c="#E65100", s=50, alpha=0.6, edgecolors="white")
r_partial, p_partial = stats.pearsonr(resid_visc, resid_D)
x_fit = np.linspace(resid_visc.min(), resid_visc.max(), 100)
slope, intercept, _, _, _ = stats.linregress(resid_visc, resid_D)
ax.plot(x_fit, intercept + slope * x_fit, color=COLORS["fit"], linewidth=2)

sig = "**" if p_partial < 0.05 else "*" if p_partial < 0.1 else "n.s."
ax.text(0.03, 0.97, f"partial r = {r_partial:+.3f} ({sig})",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
style_ax(ax, "Viscosity (residual, net of D_early)", "D_late (residual, net of D_early)",
         "Does Viscosity Add Beyond Momentum?")

fig.suptitle(f"Diagnostics — {best_name}",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_momentum_diagnostics.png", **SAVEKW)
plt.close()
log("Saved fig_momentum_diagnostics.png")

# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("MODEL COMPARISON SUMMARY")
log(f"{'═'*65}")
log(f"  {'Model':<30s} {'n':>3s} {'R²':>6s} {'AdjR²':>6s} {'CV-R²':>6s} {'AIC':>7s}")
log(f"  {'─'*30} {'─'*3} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")
for name, res in results.items():
    m = res["model"]
    cv = res["cv_r2"]
    marker = " ★" if name == best_name else ""
    log(f"  {name:<30s} {res['n']:>3d} {m.rsquared:>6.3f} {m.rsquared_adj:>6.3f} "
        f"{cv:>6.3f} {m.aic:>7.1f}{marker}")

log(f"\n  ★ = best by LOO cross-validation")

log(f"\n{'═'*65}")
log("PRACTICAL MODEL: predict D_annual from early obs + physics")
log(f"{'═'*65}")
log(f"  {'Model':<30s} {'n':>3s} {'R²':>6s} {'AdjR²':>6s} {'CV-R²':>6s} {'AIC':>7s}")
log(f"  {'─'*30} {'─'*3} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")
for name, res in results_annual.items():
    m = res["model"]
    cv = res["cv_r2"]
    marker = " ★" if name == best_annual_name else ""
    log(f"  {name:<30s} {res['n']:>3d} {m.rsquared:>6.3f} {m.rsquared_adj:>6.3f} "
        f"{cv:>6.3f} {m.aic:>7.1f}{marker}")

log(f"\n  ★ = best by LOO cross-validation")

# Save text
with open(RESULTS / "momentum_results.txt", "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved momentum_results.txt ({len(lines)} lines)")
