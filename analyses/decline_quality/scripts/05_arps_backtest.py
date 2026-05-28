"""
Script 05: Arps Decline Curve Backtest.

Arps (1945) decline models:
  Exponential (b=0):  q(t) = qi * exp(-Di * t)
  Hyperbolic  (0<b<1): q(t) = qi * (1 + b*Di*t)^(-1/b)
  Harmonic    (b=1):  q(t) = qi / (1 + Di*t)

Strategy:
  1. For each field, fit all three Arps models on post-peak data up to 2012
  2. Forecast 2013-2026 using fitted parameters
  3. Compare forecast vs. actual production
  4. Analyze: does oil quality predict which model fits best, and the forecast error?

Outputs (in results/):
  - arps_backtest_results.csv  — per-field fit parameters + forecast errors
  - fig_arps_examples.png      — example fits for selected fields
  - fig_arps_errors.png        — forecast error vs. quality features
  - fig_arps_b_parameter.png   — Arps b-parameter vs. quality
  - arps_backtest_summary.txt  — text summary
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit, minimize

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
SAVEKW = dict(bbox_inches="tight")

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")

CUTOFF_YEAR = 2012
MIN_PRE = 24
MIN_POST = 36

# ── Arps decline functions ──────────────────────────────────────────────────

def arps_exp(t, qi, Di):
    return qi * np.exp(-Di * t)

def arps_hyp(t, qi, Di, b):
    return qi * (1 + b * Di * t) ** (-1 / b)

def arps_harm(t, qi, Di):
    return qi / (1 + Di * t)


def fit_arps(t, q, model="hyp"):
    """Fit an Arps model. Returns (params_dict, q_pred, r2)."""
    q0 = q[0]

    try:
        if model == "exp":
            popt, _ = curve_fit(arps_exp, t, q, p0=[q0, 0.01],
                                bounds=([0, 1e-6], [q0 * 3, 1.0]), maxfev=5000)
            q_pred = arps_exp(t, *popt)
            params = {"qi": popt[0], "Di": popt[1], "b": 0.0}

        elif model == "hyp":
            popt, _ = curve_fit(arps_hyp, t, q, p0=[q0, 0.01, 0.5],
                                bounds=([0, 1e-6, 0.01], [q0 * 3, 1.0, 2.0]), maxfev=5000)
            q_pred = arps_hyp(t, *popt)
            params = {"qi": popt[0], "Di": popt[1], "b": popt[2]}

        elif model == "harm":
            popt, _ = curve_fit(arps_harm, t, q, p0=[q0, 0.01],
                                bounds=([0, 1e-6], [q0 * 3, 1.0]), maxfev=5000)
            q_pred = arps_harm(t, *popt)
            params = {"qi": popt[0], "Di": popt[1], "b": 1.0}
        else:
            raise ValueError(f"Unknown model: {model}")

        ss_res = np.sum((q - q_pred) ** 2)
        ss_tot = np.sum((q - q.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return params, q_pred, r2

    except (RuntimeError, ValueError):
        return None, None, np.nan


def forecast_arps(t_future, params, model="hyp"):
    """Forecast using fitted Arps parameters."""
    qi, Di, b = params["qi"], params["Di"], params["b"]
    if model == "exp":
        return arps_exp(t_future, qi, Di)
    elif model == "hyp":
        return arps_hyp(t_future, qi, Di, b)
    elif model == "harm":
        return arps_harm(t_future, qi, Di)


# ── Prepare data ────────────────────────────────────────────────────────────

post_peak = panel[panel.is_post_peak & (panel.oil_pct_peak > 1)].copy()

pre = post_peak[post_peak.year <= CUTOFF_YEAR]
post = post_peak[post_peak.year > CUTOFF_YEAR]

pre_fields = pre.groupby("field").size()
post_fields = post.groupby("field").size()

eligible = set(pre_fields[pre_fields >= MIN_PRE].index) & set(post_fields[post_fields >= MIN_POST].index)
print(f"Eligible fields: {len(eligible)}")

# ── Fit and forecast ────────────────────────────────────────────────────────

results = []

for field in sorted(eligible):
    field_pre = pre[pre.field == field].sort_values("months_since_peak")
    field_post = post[post.field == field].sort_values("months_since_peak")

    t_pre = field_pre.months_since_peak.values.astype(float)
    q_pre = field_pre.oil_pct_peak.values
    t_post = field_post.months_since_peak.values.astype(float)
    q_actual = field_post.oil_pct_peak.values

    row = {"field": field}

    best_model = None
    best_r2_fit = -np.inf

    for model_name in ["exp", "hyp", "harm"]:
        params, q_pred_pre, r2_fit = fit_arps(t_pre, q_pre, model=model_name)

        if params is None:
            row[f"{model_name}_r2_fit"] = np.nan
            row[f"{model_name}_mape_forecast"] = np.nan
            continue

        row[f"{model_name}_qi"] = params["qi"]
        row[f"{model_name}_Di"] = params["Di"]
        row[f"{model_name}_b"] = params["b"]
        row[f"{model_name}_r2_fit"] = r2_fit

        # Forecast
        q_forecast = forecast_arps(t_post, params, model=model_name)
        q_forecast = np.clip(q_forecast, 0, None)

        # Forecast errors
        mae = np.mean(np.abs(q_actual - q_forecast))
        mape = np.mean(np.abs((q_actual - q_forecast) / q_actual.clip(min=1))) * 100
        bias = np.mean(q_forecast - q_actual)
        rmse = np.sqrt(np.mean((q_actual - q_forecast) ** 2))

        row[f"{model_name}_mae_forecast"] = mae
        row[f"{model_name}_mape_forecast"] = mape
        row[f"{model_name}_bias_forecast"] = bias
        row[f"{model_name}_rmse_forecast"] = rmse

        if r2_fit > best_r2_fit:
            best_r2_fit = r2_fit
            best_model = model_name

    row["best_model"] = best_model
    row["best_r2_fit"] = best_r2_fit

    # Use best model's forecast errors as the main metrics
    if best_model:
        row["best_mape"] = row.get(f"{best_model}_mape_forecast", np.nan)
        row["best_rmse"] = row.get(f"{best_model}_rmse_forecast", np.nan)
        row["best_bias"] = row.get(f"{best_model}_bias_forecast", np.nan)
        row["best_Di"] = row.get(f"{best_model}_Di", np.nan)
        row["best_b"] = row.get(f"{best_model}_b", np.nan)

    results.append(row)

res = pd.DataFrame(results)

# Merge quality features
quality_cols = ["field", "api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct",
                "oil_mean", "main_area", "D_annual", "field_age_mean", "grade"]
res = res.merge(summary[quality_cols], on="field", how="left")

res.to_csv(RESULTS / "arps_backtest_results.csv", index=False)
print(f"Results: {len(res)} fields")

# ── Summary stats ───────────────────────────────────────────────────────────

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

log("ARPS DECLINE CURVE BACKTEST")
log(f"Cutoff: fit on post-peak data ≤{CUTOFF_YEAR}, forecast {CUTOFF_YEAR+1}-2026")
log(f"Fields: {len(res)}")
log(f"\nBest-fitting model distribution:")
log(f"  {res.best_model.value_counts().to_dict()}")
log(f"\nIn-sample fit (best model R²): median={res.best_r2_fit.median():.3f}, mean={res.best_r2_fit.mean():.3f}")
log(f"Out-of-sample MAPE: median={res.best_mape.median():.1f}%, mean={res.best_mape.mean():.1f}%")
log(f"Out-of-sample RMSE: median={res.best_rmse.median():.1f} pp, mean={res.best_rmse.mean():.1f} pp")
log(f"Forecast bias: median={res.best_bias.median():.1f} pp (>0 = overforecast)")

# Hyperbolic b-parameter
hyp_b = res[res.best_model == "hyp"]["hyp_b"]
if len(hyp_b) > 0:
    log(f"\nHyperbolic b-parameter: median={hyp_b.median():.3f}, range=[{hyp_b.min():.3f}, {hyp_b.max():.3f}]")
    log(f"  b<0.5 (near-exponential): {(hyp_b < 0.5).sum()}")
    log(f"  b=0.5-1.0 (typical): {((hyp_b >= 0.5) & (hyp_b <= 1.0)).sum()}")
    log(f"  b>1.0 (very slow decline): {(hyp_b > 1.0).sum()}")

# Quality vs forecast error correlations
log(f"\n{'='*60}")
log("Correlations: quality → forecast error (best model MAPE)")
log(f"{'='*60}")
valid = res.dropna(subset=["best_mape"])
for col in ["api_gravity", "sulfur_pct", "vacuum_resid_pct", "oil_mean", "D_annual", "best_b"]:
    v = valid[[col, "best_mape"]].dropna()
    if len(v) < 5:
        continue
    r, p = stats.pearsonr(v[col], v.best_mape)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  {col:25s} r={r:+.3f} (p={p:.3f}) {sig}")

# Quality vs b-parameter
log(f"\n{'='*60}")
log("Correlations: quality → Arps b-parameter (hyperbolic fits)")
log(f"{'='*60}")
hyp_fields = res[res.best_model == "hyp"].dropna(subset=["hyp_b"])
for col in ["api_gravity", "sulfur_pct", "vacuum_resid_pct", "oil_mean"]:
    v = hyp_fields[[col, "hyp_b"]].dropna()
    if len(v) < 5:
        continue
    r, p = stats.pearsonr(v[col], v.hyp_b)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  {col:25s} r={r:+.3f} (p={p:.3f}) {sig}")

# Regression: quality → MAPE
log(f"\n{'='*60}")
log("OLS: quality + size → forecast MAPE")
log(f"{'='*60}")
reg_data = valid.dropna(subset=["api_gravity", "sulfur_pct", "oil_mean", "best_mape"])
y_reg = reg_data["best_mape"]
X_reg = sm.add_constant(reg_data[["api_gravity", "sulfur_pct", "oil_mean"]])
m = sm.OLS(y_reg, X_reg).fit(cov_type="HC1")
for var in m.params.index:
    sig = "***" if m.pvalues[var] < 0.01 else "**" if m.pvalues[var] < 0.05 else "*" if m.pvalues[var] < 0.1 else ""
    log(f"  {var:<20s} β={m.params[var]:>10.4f} (p={m.pvalues[var]:.3f}) {sig}")
log(f"  R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Example Arps fits — 6 fields across quality spectrum
# ═══════════════════════════════════════════════════════════════════════════

# Pick 6 representative fields: 2 heavy, 2 medium, 2 light
examples = []
for api_lo, api_hi, label in [(0, 30, "heavy"), (33, 42, "medium"), (42, 60, "light")]:
    cands = res[(res.api_gravity >= api_lo) & (res.api_gravity < api_hi)].sort_values("best_r2_fit", ascending=False)
    examples.extend(cands.head(2).field.tolist())

# Fallback if not enough in a category
if len(examples) < 6:
    extras = res[~res.field.isin(examples)].sort_values("best_r2_fit", ascending=False)
    examples.extend(extras.head(6 - len(examples)).field.tolist())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, field in enumerate(examples[:6]):
    ax = axes[idx]
    r = res[res.field == field].iloc[0]

    field_data = post_peak[post_peak.field == field].sort_values("months_since_peak")
    field_pre = field_data[field_data.year <= CUTOFF_YEAR]
    field_post = field_data[field_data.year > CUTOFF_YEAR]

    # Actual data
    ax.scatter(field_pre.months_since_peak, field_pre.oil_pct_peak,
               c="#90A4AE", s=8, alpha=0.6, label="Training data", zorder=2)
    ax.scatter(field_post.months_since_peak, field_post.oil_pct_peak,
               c="#263238", s=8, alpha=0.6, label="Actual (holdout)", zorder=2)

    # Forecast lines for each model
    t_all = np.linspace(field_data.months_since_peak.min(),
                        field_data.months_since_peak.max(), 300)

    model_colors = {"exp": "#2196F3", "hyp": "#F44336", "harm": "#4CAF50"}
    model_labels = {"exp": "Exponential", "hyp": "Hyperbolic", "harm": "Harmonic"}

    for model_name, color in model_colors.items():
        qi = r.get(f"{model_name}_qi")
        Di = r.get(f"{model_name}_Di")
        b = r.get(f"{model_name}_b")
        if pd.isna(qi) or pd.isna(Di):
            continue
        params = {"qi": qi, "Di": Di, "b": b}
        q_line = forecast_arps(t_all, params, model=model_name)
        q_line = np.clip(q_line, 0, None)

        r2 = r.get(f"{model_name}_r2_fit", np.nan)
        mape = r.get(f"{model_name}_mape_forecast", np.nan)
        lw = 2.5 if model_name == r.best_model else 1.0
        ls = "-" if model_name == r.best_model else "--"

        ax.plot(t_all, q_line, color=color, linewidth=lw, linestyle=ls, alpha=0.8,
                label=f"{model_labels[model_name]} (R²={r2:.2f}, MAPE={mape:.0f}%)", zorder=3)

    # Cutoff line
    cutoff_month = field_pre.months_since_peak.max()
    ax.axvline(cutoff_month, color="#FF9800", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(cutoff_month + 2, ax.get_ylim()[1] * 0.95, f"← fit | forecast →",
            fontsize=7, alpha=0.5, va="top")

    api = r.api_gravity
    ax.set_title(f"{field} (API={api:.0f}°)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Months since peak", fontsize=8)
    ax.set_ylabel("Production (% of peak)", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

fig.suptitle(f"Arps Decline Curve Backtest — Fit ≤{CUTOFF_YEAR}, Forecast {CUTOFF_YEAR+1}-2026",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_arps_examples.png", **SAVEKW)
plt.close()
log("\nSaved fig_arps_examples.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Forecast error vs. quality features
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax_idx, (x_col, x_label) in enumerate([
    ("api_gravity", "API Gravity (°)"),
    ("sulfur_pct", "Sulfur Content (%)"),
    ("oil_mean", "Avg Production (Mill Sm³/mnd)"),
]):
    ax = axes[ax_idx]
    v = valid.dropna(subset=[x_col, "best_mape"])
    ax.scatter(v[x_col], v.best_mape, c="#1565C0", alpha=0.6, edgecolors="white", s=60)

    slope, intercept, r, p, _ = stats.linregress(v[x_col], v.best_mape)
    x_fit = np.linspace(v[x_col].min(), v[x_col].max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="#455A64", linewidth=2)

    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.03, 0.97, f"r={r:.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Label worst-predicted fields
    for _, row in v.nlargest(3, "best_mape").iterrows():
        ax.annotate(row.field, (row[x_col], row.best_mape),
                    fontsize=6.5, alpha=0.6, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Forecast MAPE (%)" if ax_idx == 0 else "", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Arps Forecast Error vs. Oil Quality — Which Fields Are Harder to Predict?",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_arps_errors.png", **SAVEKW)
plt.close()
log("Saved fig_arps_errors.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Arps b-parameter vs. quality
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# b-parameter distribution
ax = axes[0]
hyp = res[res.best_model == "hyp"]
ax.hist(hyp.hyp_b, bins=15, color="#795548", alpha=0.8, edgecolor="white")
ax.axvline(0, color="red", linewidth=1, linestyle="--", label="b=0 (exponential)")
ax.axvline(1, color="blue", linewidth=1, linestyle="--", label="b=1 (harmonic)")
ax.set_xlabel("Arps b-parameter")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Hyperbolic b")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# b vs API
ax = axes[1]
v = hyp.dropna(subset=["api_gravity", "hyp_b"])
ax.scatter(v.api_gravity, v.hyp_b, c="#1565C0", alpha=0.6, edgecolors="white", s=60)
if len(v) >= 5:
    slope, intercept, r, p, _ = stats.linregress(v.api_gravity, v.hyp_b)
    x_fit = np.linspace(v.api_gravity.min(), v.api_gravity.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="#455A64", linewidth=2)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.03, 0.97, f"r={r:.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

for _, row in v.nlargest(3, "hyp_b").iterrows():
    ax.annotate(row.field, (row.api_gravity, row.hyp_b),
                fontsize=7, alpha=0.6, xytext=(5, 5), textcoords="offset points")

ax.axhline(0.5, color="#9E9E9E", linewidth=0.5, linestyle=":")
ax.axhline(1.0, color="#9E9E9E", linewidth=0.5, linestyle=":")
ax.set_xlabel("API Gravity (°)")
ax.set_ylabel("Arps b-parameter")
ax.set_title("API Gravity vs. Decline Curve Shape (b)")
ax.grid(True, alpha=0.2)

fig.suptitle("Arps Hyperbolic b-Parameter — Does Quality Predict Decline Shape?",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_arps_b_parameter.png", **SAVEKW)
plt.close()
log("Saved fig_arps_b_parameter.png")

# ── Save text summary ───────────────────────────────────────────────────────

with open(RESULTS / "arps_backtest_summary.txt", "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved arps_backtest_summary.txt")
print("Done.")
