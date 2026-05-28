"""
Script 10: Systematic momentum optimization.

Tests every reasonable way to capture decline momentum:
  1. Single-window D: vary window length (12, 18, 24, 36, 48 months)
  2. Multi-period D: split post-peak into 2 or 3 equal periods
  3. Trend features: slope of D over rolling windows (acceleration)
  4. Level + change: D_early plus ΔD (deceleration)
  5. Production-shape features: time to 50% of peak, curvature

Target: D_annual (full-life decline rate)
Evaluation: LOO cross-validation R²

Outputs:
  - fig_momentum_opt.png       — heatmap + best model
  - momentum_opt_results.txt   — full tables
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


def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1


def fit_D_window(grp, t_min, t_max, min_obs=10):
    d = grp[(grp.months_since_peak >= t_min) & (grp.months_since_peak < t_max)].copy()
    d = d[d.oil_pct_peak > 0].dropna(subset=["months_since_peak", "oil_pct_peak"])
    if len(d) < min_obs:
        return np.nan
    t = d.months_since_peak.values
    ln_y = np.log(d.oil_pct_peak.values)
    slope, intercept, r, p, se = stats.linregress(t, ln_y)
    return -slope * 12  # annualized


def loo_cv_r2(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    if len(y) < 10:
        return np.nan, len(y)
    loo = LeaveOneOut()
    lr = LinearRegression()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        lr.fit(X[train_idx], y[train_idx])
        preds[test_idx] = lr.predict(X[test_idx])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, len(y)


post = panel[panel.is_post_peak].copy()
fields = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
fields["ln_viscosity"] = np.log(beggs_robinson(fields.api_gravity))

log("MOMENTUM OPTIMIZATION")
log(f"Fields with D_annual: {len(fields)}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. SINGLE WINDOW: vary early window length
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 1: Single early window — vary length")
log(f"{'═'*65}")

windows = [6, 12, 18, 24, 36, 48, 60]
single_results = {}

for w in windows:
    D_w = post.groupby("field").apply(lambda g: fit_D_window(g, 0, w)).rename("D_window")
    merged = fields.merge(D_w, on="field")
    merged = merged.dropna(subset=["D_window", "D_annual"])

    if len(merged) < 10:
        log(f"  Window 0–{w:>2d}mo: skipped (n={len(merged)} < 10)")
        continue

    # D_window alone
    cv_r2, n = loo_cv_r2(merged[["D_window"]].values, merged.D_annual.values)
    r, p = stats.pearsonr(merged.D_window, merged.D_annual)

    # D_window + size
    cv_r2_size, _ = loo_cv_r2(merged[["D_window", "oil_mean"]].values, merged.D_annual.values)

    # D_window + viscosity + size
    cv_r2_full, _ = loo_cv_r2(
        merged[["D_window", "ln_viscosity", "oil_mean"]].values, merged.D_annual.values)

    single_results[w] = {
        "n": n, "r": r, "p": p,
        "cv_alone": cv_r2, "cv_size": cv_r2_size, "cv_full": cv_r2_full,
    }

    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  Window 0–{w:>2d}mo: r={r:+.3f}{sig:3s}  "
        f"CV(alone)={cv_r2:+.3f}  CV(+size)={cv_r2_size:+.3f}  "
        f"CV(+visc+size)={cv_r2_full:+.3f}  n={n}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TWO-PERIOD SPLIT: D from period 1 and period 2
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 2: Two-period momentum (D_p1, D_p2)")
log(f"{'═'*65}")

splits_2p = [
    (12, 24), (12, 36), (18, 36), (24, 48), (12, 48), (18, 48),
]

two_period_results = {}

for s1, s2 in splits_2p:
    D_p1 = post.groupby("field").apply(lambda g: fit_D_window(g, 0, s1)).rename("D_p1")
    D_p2 = post.groupby("field").apply(lambda g: fit_D_window(g, s1, s2)).rename("D_p2")

    merged = fields.merge(D_p1, on="field").merge(D_p2, on="field")
    merged = merged.dropna(subset=["D_p1", "D_p2", "D_annual"])
    merged["D_delta"] = merged.D_p2 - merged.D_p1
    merged["D_ratio"] = merged.D_p2 / merged.D_p1.clip(lower=0.001)

    # Test various combos
    combos = {
        "D_p1 only":       ["D_p1"],
        "D_p2 only":       ["D_p2"],
        "D_p1 + D_p2":     ["D_p1", "D_p2"],
        "D_p1 + ΔD":       ["D_p1", "D_delta"],
        "D_p1+D_p2+size":  ["D_p1", "D_p2", "oil_mean"],
    }

    label = f"[0–{s1}]+[{s1}–{s2}]"
    best_cv = -999
    best_combo = ""

    for combo_name, cols in combos.items():
        cv_r2, n = loo_cv_r2(merged[cols].values, merged.D_annual.values)
        if cv_r2 > best_cv:
            best_cv = cv_r2
            best_combo = combo_name

    two_period_results[(s1, s2)] = {"n": len(merged), "best_cv": best_cv, "best_combo": best_combo}
    log(f"  {label:20s} n={len(merged):2d}  best: {best_combo:20s} CV R²={best_cv:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. THREE-PERIOD SPLIT
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 3: Three-period momentum (D_p1, D_p2, D_p3)")
log(f"{'═'*65}")

splits_3p = [
    (12, 24, 36), (12, 24, 48), (12, 36, 60),
    (18, 36, 54), (24, 48, 72),
]

three_period_results = {}

for s1, s2, s3 in splits_3p:
    D_p1 = post.groupby("field").apply(lambda g: fit_D_window(g, 0, s1)).rename("D_p1")
    D_p2 = post.groupby("field").apply(lambda g: fit_D_window(g, s1, s2)).rename("D_p2")
    D_p3 = post.groupby("field").apply(lambda g: fit_D_window(g, s2, s3)).rename("D_p3")

    merged = fields.merge(D_p1, on="field").merge(D_p2, on="field").merge(D_p3, on="field")
    merged = merged.dropna(subset=["D_p1", "D_p2", "D_p3", "D_annual"])

    # Trend: slope of D across periods
    mid_points = np.array([s1/2, (s1+s2)/2, (s2+s3)/2])
    def calc_trend(row):
        D_vals = np.array([row.D_p1, row.D_p2, row.D_p3])
        if np.any(np.isnan(D_vals)):
            return np.nan
        slope, _, _, _, _ = stats.linregress(mid_points, D_vals)
        return slope
    merged["D_trend"] = merged.apply(calc_trend, axis=1)

    combos = {
        "D_p1+D_p2+D_p3":       ["D_p1", "D_p2", "D_p3"],
        "D_p1 + trend":          ["D_p1", "D_trend"],
        "all + size":            ["D_p1", "D_p2", "D_p3", "oil_mean"],
        "D_p1+trend+size":       ["D_p1", "D_trend", "oil_mean"],
    }

    label = f"[0–{s1}]+[{s1}–{s2}]+[{s2}–{s3}]"
    best_cv = -999
    best_combo = ""

    for combo_name, cols in combos.items():
        sub = merged.dropna(subset=cols)
        if len(sub) < 15:
            continue
        cv_r2, n = loo_cv_r2(sub[cols].values, sub.D_annual.values)
        if cv_r2 > best_cv:
            best_cv = cv_r2
            best_combo = combo_name

    three_period_results[(s1, s2, s3)] = {"n": len(merged), "best_cv": best_cv, "best_combo": best_combo}
    log(f"  {label:28s} n={len(merged):2d}  best: {best_combo:20s} CV R²={best_cv:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. PRODUCTION SHAPE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 4: Production shape features")
log(f"{'═'*65}")

def shape_features(grp):
    d = grp[grp.is_post_peak].sort_values("months_since_peak")
    if len(d) < 12:
        return pd.Series({"t_50": np.nan, "t_75": np.nan, "prod_12m_avg": np.nan,
                           "curvature": np.nan, "initial_drop": np.nan})

    # Time to reach 50% and 75% of peak
    below_50 = d[d.oil_pct_peak <= 50]
    t_50 = below_50.months_since_peak.min() if len(below_50) > 0 else np.nan

    below_75 = d[d.oil_pct_peak <= 75]
    t_75 = below_75.months_since_peak.min() if len(below_75) > 0 else np.nan

    # Average production in first 12 months (as % of peak) — how fast initial drop
    first_12 = d[d.months_since_peak <= 12]
    prod_12m_avg = first_12.oil_pct_peak.mean() if len(first_12) > 0 else np.nan

    # Initial drop: peak to month 6 average
    first_6 = d[d.months_since_peak <= 6]
    initial_drop = 100 - first_6.oil_pct_peak.mean() if len(first_6) > 0 else np.nan

    # Curvature: fit quadratic to first 36 months of ln(production)
    first_36 = d[d.months_since_peak <= 36]
    first_36 = first_36[first_36.oil_pct_peak > 0]
    if len(first_36) >= 12:
        t = first_36.months_since_peak.values
        ln_y = np.log(first_36.oil_pct_peak.values)
        coeffs = np.polyfit(t, ln_y, 2)
        curvature = coeffs[0]  # quadratic term: >0 means convex (decelerating)
    else:
        curvature = np.nan

    return pd.Series({"t_50": t_50, "t_75": t_75, "prod_12m_avg": prod_12m_avg,
                       "curvature": curvature, "initial_drop": initial_drop})

shapes = panel.groupby("field").apply(shape_features).reset_index()
merged_shape = fields.merge(shapes, on="field")
merged_shape = merged_shape.dropna(subset=["D_annual"])

# Test each shape feature
shape_cols = ["t_50", "t_75", "prod_12m_avg", "curvature", "initial_drop"]
shape_results = {}

for col in shape_cols:
    sub = merged_shape.dropna(subset=[col])
    r, p = stats.pearsonr(sub[col], sub.D_annual)
    cv_r2, n = loo_cv_r2(sub[[col]].values, sub.D_annual.values)
    cv_r2_size, _ = loo_cv_r2(sub[[col, "oil_mean"]].values, sub.D_annual.values)
    shape_results[col] = {"r": r, "p": p, "cv_alone": cv_r2, "cv_size": cv_r2_size, "n": n}

    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  {col:20s} r={r:+.3f}{sig:3s}  CV(alone)={cv_r2:+.3f}  "
        f"CV(+size)={cv_r2_size:+.3f}  n={n}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. COMBINED: comprehensive combo search
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("TEST 5: Best combinations")
log(f"{'═'*65}")

best_window = max(single_results, key=lambda k: single_results[k]["cv_alone"])
log(f"  Best single window: 0–{best_window} months")

D_p1_12 = post.groupby("field").apply(lambda g: fit_D_window(g, 0, 12)).rename("D_12")
D_p2_12_24 = post.groupby("field").apply(lambda g: fit_D_window(g, 12, 24)).rename("D_12_24")
D_p3_24_36 = post.groupby("field").apply(lambda g: fit_D_window(g, 24, 36)).rename("D_24_36")

master = (fields
    .merge(D_p1_12, on="field")
    .merge(shapes, on="field")
    .merge(D_p2_12_24, on="field", how="left")
    .merge(D_p3_24_36, on="field", how="left"))
master = master.dropna(subset=["D_12", "D_annual"])
master["D_delta"] = master.D_12_24 - master.D_12

# Trend across 3 periods
mid_pts = np.array([6, 18, 30])
def calc_trend(row):
    vals = np.array([row.D_12, row.D_12_24, row.D_24_36])
    if np.any(np.isnan(vals)):
        return np.nan
    return stats.linregress(mid_pts, vals)[0]
master["D_trend"] = master.apply(calc_trend, axis=1)

# Also add best 2-period split
best_2p = max(two_period_results, key=lambda k: two_period_results[k]["best_cv"])
s1, s2 = best_2p
D_p1 = post.groupby("field").apply(lambda g: fit_D_window(g, 0, s1)).rename("D_p1")
D_p2 = post.groupby("field").apply(lambda g: fit_D_window(g, s1, s2)).rename("D_p2")
master = master.merge(D_p1, on="field", how="left").merge(D_p2, on="field", how="left")

log(f"  Best 2-period split: [0–{s1}]+[{s1}–{s2}]")
log(f"  Master dataset: {len(master)} fields")

combo_specs = {
    "D_12":                              ["D_12"],
    "D_12 + size":                       ["D_12", "oil_mean"],
    "D_12 + visc":                       ["D_12", "ln_viscosity"],
    "D_12 + visc + size":                ["D_12", "ln_viscosity", "oil_mean"],
    "D_p1 + D_p2":                       ["D_p1", "D_p2"],
    "D_p1 + ΔD":                         ["D_p1", "D_delta"],
    "D_12 + D_12_24":                    ["D_12", "D_12_24"],
    "D_12 + D_12_24 + visc":             ["D_12", "D_12_24", "ln_viscosity"],
    "D_12 + D_12_24 + visc + size":      ["D_12", "D_12_24", "ln_viscosity", "oil_mean"],
    "D_12 + trend":                      ["D_12", "D_trend"],
    "D_12 + trend + visc":               ["D_12", "D_trend", "ln_viscosity"],
    "D_12 + trend + visc + size":        ["D_12", "D_trend", "ln_viscosity", "oil_mean"],
    "initial_drop":                      ["initial_drop"],
    "initial_drop + size":               ["initial_drop", "oil_mean"],
    "curvature":                         ["curvature"],
    "curvature + size":                  ["curvature", "oil_mean"],
    "D_12 + curvature":                  ["D_12", "curvature"],
    "D_12 + curvature + size":           ["D_12", "curvature", "oil_mean"],
    "D_12 + initial_drop":               ["D_12", "initial_drop"],
    "D_12 + init_drop + size":           ["D_12", "initial_drop", "oil_mean"],
    "t_50 only":                         ["t_50"],
    "t_50 + size":                       ["t_50", "oil_mean"],
    "D_12 + D_12_24 + curvature":        ["D_12", "D_12_24", "curvature"],
    "kitchen sink":                      ["D_12", "D_delta", "curvature",
                                          "initial_drop", "oil_mean", "ln_viscosity"],
}

combo_results = {}
log(f"\n  {'Model':<35s} {'n':>3s} {'CV R²':>7s} {'R²':>6s}")
log(f"  {'─'*35} {'─'*3} {'─'*7} {'─'*6}")

for name, cols in combo_specs.items():
    sub = master.dropna(subset=cols + ["D_annual"])
    if len(sub) < 15:
        continue
    cv_r2, n = loo_cv_r2(sub[cols].values, sub.D_annual.values)

    X = sm.add_constant(sub[cols])
    m = sm.OLS(sub.D_annual, X).fit()
    r2 = m.rsquared

    combo_results[name] = {"cv_r2": cv_r2, "r2": r2, "n": n, "cols": cols}
    log(f"  {name:<35s} {n:>3d} {cv_r2:>+7.3f} {r2:>6.3f}")

# Sort by CV R²
log(f"\n  TOP 5 by CV R²:")
sorted_combos = sorted(combo_results.items(), key=lambda x: x[1]["cv_r2"], reverse=True)
for i, (name, res) in enumerate(sorted_combos[:5]):
    log(f"  {i+1}. {name:<35s} CV R²={res['cv_r2']:+.3f} (R²={res['r2']:.3f}, n={res['n']})")

# ═══════════════════════════════════════════════════════════════════════════
# Detailed regression for top model
# ═══════════════════════════════════════════════════════════════════════════

top_name, top_res = sorted_combos[0]
top_cols = top_res["cols"]

log(f"\n{'═'*65}")
log(f"BEST MODEL: {top_name}")
log(f"{'═'*65}")

sub = master.dropna(subset=top_cols + ["D_annual"])
X = sm.add_constant(sub[top_cols])
m = sm.OLS(sub.D_annual, X).fit(cov_type="HC1")

log(f"  R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, AIC={m.aic:.1f}, n={len(sub)}")
for var in top_cols:
    sig = "***" if m.pvalues[var] < 0.01 else "**" if m.pvalues[var] < 0.05 else "*" if m.pvalues[var] < 0.1 else ""
    log(f"    {var:<20s} β={m.params[var]:>10.5f} (p={m.pvalues[var]:.3f}) {sig}")

# Standardized β
df_std = sub.copy()
for col in top_cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
df_std["D_z"] = (df_std.D_annual - df_std.D_annual.mean()) / df_std.D_annual.std()
X_std = sm.add_constant(df_std[top_cols])
m_std = sm.OLS(df_std.D_z, X_std).fit(cov_type="HC1")

log(f"\n  Standardized β:")
for var in top_cols:
    sig = "***" if m_std.pvalues[var] < 0.01 else "**" if m_std.pvalues[var] < 0.05 else "*" if m_std.pvalues[var] < 0.1 else ""
    log(f"    {var:<20s} β={m_std.params[var]:>+7.3f} (p={m_std.pvalues[var]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

def style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

# 1: Single window CV R² by window length
ax = axes[0, 0]
w_list = sorted(single_results.keys())
cv_alone = [single_results[w]["cv_alone"] for w in w_list]
cv_size = [single_results[w]["cv_size"] for w in w_list]
cv_full = [single_results[w]["cv_full"] for w in w_list]

ax.plot(w_list, cv_alone, "o-", color="#1565C0", linewidth=2, markersize=8, label="D_window alone")
ax.plot(w_list, cv_size, "s-", color="#E65100", linewidth=2, markersize=8, label="+ field size")
ax.plot(w_list, cv_full, "D-", color="#2E7D32", linewidth=2, markersize=8, label="+ visc + size")
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

best_w = max(w_list, key=lambda w: single_results[w]["cv_alone"])
ax.axvline(best_w, color="#1565C0", linewidth=1, linestyle="--", alpha=0.5)
ax.text(best_w + 1, max(cv_alone) * 0.9, f"Best: {best_w}mo", fontsize=9, color="#1565C0")

ax.set_xticks(w_list)
ax.legend(fontsize=8)
style_ax(ax, "Early Window Length (months)", "LOO-CV R²",
         "1. Optimal Window Length")

# 2: Two-period results
ax = axes[0, 1]
labels_2p = [f"[0–{s1}]+\n[{s1}–{s2}]" for s1, s2 in splits_2p]
cv_2p = [two_period_results[k]["best_cv"] for k in splits_2p]
colors_2p = ["#E65100" if v == max(cv_2p) else "#78909C" for v in cv_2p]
bars = ax.bar(range(len(labels_2p)), cv_2p, color=colors_2p, alpha=0.8, edgecolor="white")
ax.set_xticks(range(len(labels_2p)))
ax.set_xticklabels(labels_2p, fontsize=7)
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
for i, v in enumerate(cv_2p):
    ax.text(i, max(v, 0) + 0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
style_ax(ax, "", "Best LOO-CV R²", "2. Two-Period Splits")

# 3: Shape features
ax = axes[0, 2]
shape_names = list(shape_results.keys())
cv_shape = [shape_results[k]["cv_alone"] for k in shape_names]
cv_shape_size = [shape_results[k]["cv_size"] for k in shape_names]

x_pos = np.arange(len(shape_names))
w_bar = 0.35
ax.bar(x_pos - w_bar/2, cv_shape, w_bar, color="#1565C0", alpha=0.8, label="Alone", edgecolor="white")
ax.bar(x_pos + w_bar/2, cv_shape_size, w_bar, color="#E65100", alpha=0.8, label="+ size", edgecolor="white")
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.set_xticks(x_pos)
nice_names = {"t_50": "Time to\n50%", "t_75": "Time to\n75%", "prod_12m_avg": "Avg prod\n12mo",
              "curvature": "Curva-\nture", "initial_drop": "Initial\ndrop"}
ax.set_xticklabels([nice_names.get(n, n) for n in shape_names], fontsize=8)
ax.legend(fontsize=8)
style_ax(ax, "", "LOO-CV R²", "3. Production Shape Features")

# 4: Top 10 models ranked
ax = axes[1, 0]
top_10 = sorted_combos[:10]
model_names = [n for n, _ in top_10]
cv_vals = [r["cv_r2"] for _, r in top_10]
colors_top = ["#E65100" if i == 0 else "#1565C0" for i in range(len(top_10))]
ax.barh(range(len(top_10)), cv_vals, color=colors_top, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(model_names, fontsize=8)
ax.axvline(0, color="black", linewidth=0.5)
for i, v in enumerate(cv_vals):
    ax.text(max(v, 0) + 0.003, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")
ax.invert_yaxis()
style_ax(ax, "LOO-CV R²", "", "4. Top 10 Models (all categories)")

# 5: Best model — predicted vs actual
ax = axes[1, 1]
sub = master.dropna(subset=top_cols + ["D_annual"])
X_sk = sub[top_cols].values
y_sk = sub.D_annual.values
loo = LeaveOneOut()
lr = LinearRegression()
cv_preds = np.zeros(len(y_sk))
for train_idx, test_idx in loo.split(X_sk):
    lr.fit(X_sk[train_idx], y_sk[train_idx])
    cv_preds[test_idx] = lr.predict(X_sk[test_idx])

ax.scatter(cv_preds, y_sk, c="#E65100", s=50, alpha=0.6, edgecolors="white")
lims = [0, max(cv_preds.max(), y_sk.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect")
cv_r2_top = 1 - np.sum((y_sk - cv_preds)**2) / np.sum((y_sk - y_sk.mean())**2)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(fontsize=8)
style_ax(ax, "LOO-CV Predicted D_annual (yr⁻¹)", "Actual D_annual (yr⁻¹)",
         f"5. Best: {top_name}\n(CV R² = {cv_r2_top:.3f})")

for _, row in sub.nlargest(3, "D_annual").iterrows():
    idx = sub.index.get_loc(row.name)
    ax.annotate(row.field, (cv_preds[idx], row.D_annual),
                fontsize=7, alpha=0.6, xytext=(5, 3), textcoords="offset points")

# 6: Standardized β for best model
ax = axes[1, 2]
std_coefs = [m_std.params[v] for v in top_cols]
std_pvals = [m_std.pvalues[v] for v in top_cols]
colors_beta = ["#C62828" if p < 0.05 else "#FF8F00" if p < 0.1 else "#9E9E9E" for p in std_pvals]

label_map = {
    "D_12": "D (0–12mo)\n[Momentum]",
    "D_12_24": "D (12–24mo)\n[2nd period]",
    "D_trend": "D trend\n[Acceleration]",
    "oil_mean": "Field Size",
    "ln_viscosity": "ln(Viscosity)\n[Darcy]",
    "D_p1": "D Period 1",
    "D_p2": "D Period 2",
    "D_delta": "ΔD (P2−P1)\n[Acceleration]",
    "curvature": "Curvature\n[Shape]",
    "initial_drop": "Initial Drop\n[Shape]",
    "t_50": "Time to 50%",
    "t_75": "Time to 75%",
}
ylabels = [label_map.get(v, v) for v in top_cols]

ax.barh(range(len(top_cols)), std_coefs, color=colors_beta, alpha=0.7,
        edgecolor="white", height=0.6)
for i, (lo, hi) in enumerate(zip(m_std.conf_int().loc[top_cols, 0], m_std.conf_int().loc[top_cols, 1])):
    ax.plot([lo, hi], [i, i], color="#37474F", linewidth=2)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(range(len(top_cols)))
ax.set_yticklabels(ylabels, fontsize=9)

for i, (c, p) in enumerate(zip(std_coefs, std_pvals)):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.text(c + (0.02 if c >= 0 else -0.02), i, f"{c:+.3f}{sig}",
            va="center", ha="left" if c >= 0 else "right", fontsize=9)

style_ax(ax, "Standardized β", "", "6. Relative Importance")

fig.suptitle("Momentum Optimization — Systematic Search for Best Decline Predictor",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_momentum_opt.png", **SAVEKW)
plt.close()
log("\nSaved fig_momentum_opt.png")

with open(RESULTS / "momentum_opt_results.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved momentum_opt_results.txt ({len(lines)} lines)")
