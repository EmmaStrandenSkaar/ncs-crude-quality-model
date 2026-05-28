"""
Script 06: Pure Arps Decline Curve Analysis.

Arps (1945) three models:
  Exponential:  q(t) = qi · exp(−Di · t)                    [b = 0]
  Hyperbolic:   q(t) = qi · (1 + b·Di·t)^(−1/b)            [0 < b < 1]
  Harmonic:     q(t) = qi / (1 + Di·t)                      [b = 1]

Approach:
  1. For each field: use first 5 years of post-peak data to fit qi, Di, b
  2. Forecast remaining production life using fitted parameters
  3. Compare forecast vs. actual
  4. Analyze residuals: are they systematic? Does quality explain the miss?

Typical b-values by drive mechanism (petroleum engineering literature):
  b ≈ 0      — depletion drive, high-pressure gas reservoirs
  b ≈ 0.3–0.5 — solution-gas drive
  b ≈ 0.5    — "textbook" oil reservoir
  b ≈ 0.5–0.7 — partial water drive
  b ≈ 0.7–1.0 — strong water drive / water injection
  b > 1       — transient flow (not true boundary-dominated decline)

Most NCS fields use water injection → expect b in 0.5–1.0 range.

Outputs (in results/):
  - fig_arps_pure_examples.png    — 9 fields: actual vs. 3 Arps models
  - fig_arps_pure_residuals.png   — residual analysis (bias, quality link)
  - fig_arps_pure_summary.png     — fitted parameters overview
  - arps_pure_results.csv         — per-field fit + forecast metrics
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
SAVEKW = dict(bbox_inches="tight")

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")

# ── Arps formulas ───────────────────────────────────────────────────────────

def arps_exp(t, qi, Di):
    return qi * np.exp(-Di * t)

def arps_hyp(t, qi, Di, b):
    return qi * (1 + b * Di * t) ** (-1 / b)

def arps_harm(t, qi, Di):
    return qi / (1 + Di * t)

# ── Configuration ───────────────────────────────────────────────────────────

FIT_WINDOW_MONTHS = 60  # 5 years of post-peak data to calibrate
MIN_FIT_MONTHS = 36
MIN_FORECAST_MONTHS = 60

# ── Prepare post-peak data ──────────────────────────────────────────────────

post_peak = panel[panel.is_post_peak & (panel.oil_msm3 > 0.001)].copy()
post_peak = post_peak.sort_values(["field", "months_since_peak"])

# ── Fit each field ──────────────────────────────────────────────────────────

results = []

for field, grp in post_peak.groupby("field"):
    grp = grp.sort_values("months_since_peak").reset_index(drop=True)
    total_months = len(grp)

    # Split: first FIT_WINDOW_MONTHS for calibration, rest for validation
    fit_data = grp[grp.months_since_peak <= FIT_WINDOW_MONTHS]
    forecast_data = grp[grp.months_since_peak > FIT_WINDOW_MONTHS]

    if len(fit_data) < MIN_FIT_MONTHS or len(forecast_data) < MIN_FORECAST_MONTHS:
        continue

    t_fit = fit_data.months_since_peak.values.astype(float)
    q_fit = fit_data.oil_pct_peak.values

    t_fore = forecast_data.months_since_peak.values.astype(float)
    q_actual = forecast_data.oil_pct_peak.values

    row = {"field": field, "n_fit": len(fit_data), "n_forecast": len(forecast_data),
           "total_months_post_peak": total_months}

    for model_name, func, p0, bounds in [
        ("exp", arps_exp, None, None),
        ("hyp", arps_hyp, None, None),
        ("harm", arps_harm, None, None),
    ]:
        q0 = q_fit[0]
        try:
            if model_name == "exp":
                popt, _ = curve_fit(func, t_fit, q_fit,
                                    p0=[q0, 0.01],
                                    bounds=([0, 1e-6], [200, 0.5]),
                                    maxfev=10000)
                params = {"qi": popt[0], "Di": popt[1], "b": 0.0}
                q_pred_fit = func(t_fit, *popt)
                q_pred_fore = func(t_fore, *popt)

            elif model_name == "hyp":
                popt, _ = curve_fit(func, t_fit, q_fit,
                                    p0=[q0, 0.01, 0.5],
                                    bounds=([0, 1e-6, 0.01], [200, 0.5, 2.0]),
                                    maxfev=10000)
                params = {"qi": popt[0], "Di": popt[1], "b": popt[2]}
                q_pred_fit = func(t_fit, *popt)
                q_pred_fore = func(t_fore, *popt)

            elif model_name == "harm":
                popt, _ = curve_fit(func, t_fit, q_fit,
                                    p0=[q0, 0.01],
                                    bounds=([0, 1e-6], [200, 0.5]),
                                    maxfev=10000)
                params = {"qi": popt[0], "Di": popt[1], "b": 1.0}
                q_pred_fit = func(t_fit, *popt)
                q_pred_fore = func(t_fore, *popt)

        except (RuntimeError, ValueError):
            row[f"{model_name}_qi"] = np.nan
            continue

        q_pred_fore = np.clip(q_pred_fore, 0, None)

        # In-sample metrics
        ss_res = np.sum((q_fit - q_pred_fit) ** 2)
        ss_tot = np.sum((q_fit - q_fit.mean()) ** 2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Out-of-sample metrics
        errors = q_actual - q_pred_fore
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)
        mape = np.mean(np.abs(errors / q_actual.clip(min=1))) * 100

        # Cumulative production error (economically meaningful)
        cum_actual = np.sum(q_actual)
        cum_forecast = np.sum(q_pred_fore)
        cum_error_pct = (cum_forecast - cum_actual) / cum_actual * 100

        row[f"{model_name}_qi"] = params["qi"]
        row[f"{model_name}_Di"] = params["Di"]
        row[f"{model_name}_b"] = params["b"]
        row[f"{model_name}_r2_fit"] = r2_fit
        row[f"{model_name}_mae"] = mae
        row[f"{model_name}_rmse"] = rmse
        row[f"{model_name}_bias"] = bias
        row[f"{model_name}_mape"] = mape
        row[f"{model_name}_cum_err_pct"] = cum_error_pct

    results.append(row)

res = pd.DataFrame(results)

# Pick best model per field (by in-sample R²)
for _, r in res.iterrows():
    r2s = {m: r.get(f"{m}_r2_fit", -999) for m in ["exp", "hyp", "harm"]}
    r2s = {k: v for k, v in r2s.items() if not np.isnan(v)}
    if r2s:
        res.loc[res.field == r.field, "best_model"] = max(r2s, key=r2s.get)

for col_suffix in ["r2_fit", "mape", "rmse", "bias", "cum_err_pct", "qi", "Di", "b"]:
    res[f"best_{col_suffix}"] = res.apply(
        lambda r: r.get(f"{r.best_model}_{col_suffix}", np.nan)
        if pd.notna(r.get("best_model")) else np.nan, axis=1)

# Merge quality
qual_cols = ["field", "api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct",
             "oil_mean", "main_area", "D_annual", "field_age_mean", "grade"]
res = res.merge(summary[qual_cols], on="field", how="left")

res.to_csv(RESULTS / "arps_pure_results.csv", index=False)
print(f"Fields analyzed: {len(res)}")

# ── Print summary ───────────────────────────────────────────────────────────

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

log("PURE ARPS DECLINE CURVE ANALYSIS")
log(f"Fit window: first {FIT_WINDOW_MONTHS} months post-peak")
log(f"Forecast window: month {FIT_WINDOW_MONTHS}+ (median {res.n_forecast.median():.0f} months)")
log(f"Fields: {len(res)}")

log(f"\n{'═'*65}")
log("FITTED PARAMETERS")
log(f"{'═'*65}")

for model in ["exp", "hyp", "harm"]:
    qi = res[f"{model}_qi"].dropna()
    Di = res[f"{model}_Di"].dropna()
    r2 = res[f"{model}_r2_fit"].dropna()
    if len(qi) == 0:
        continue
    log(f"\n  {model.upper():12s} (n={len(qi)} fits)")
    log(f"    qi: median={qi.median():.1f}%, Di: median={Di.median():.5f}/mnd ({Di.median()*12:.4f}/yr)")
    log(f"    In-sample R²: median={r2.median():.3f}, range=[{r2.min():.3f}, {r2.max():.3f}]")
    if model == "hyp":
        b = res["hyp_b"].dropna()
        log(f"    b-parameter: median={b.median():.3f}, range=[{b.min():.3f}, {b.max():.3f}]")

log(f"\n{'═'*65}")
log("BEST MODEL PER FIELD")
log(f"{'═'*65}")
log(f"  {res.best_model.value_counts().to_dict()}")

log(f"\n{'═'*65}")
log("FORECAST ACCURACY (best model per field)")
log(f"{'═'*65}")
log(f"  In-sample R²:      median={res.best_r2_fit.median():.3f}")
log(f"  Out-of-sample MAPE: median={res.best_mape.median():.1f}%")
log(f"  Out-of-sample RMSE: median={res.best_rmse.median():.1f} pp")
log(f"  Forecast bias:     median={res.best_bias.median():.1f} pp ({'under' if res.best_bias.median() < 0 else 'over'}-forecasts)")
log(f"  Cumulative error:  median={res.best_cum_err_pct.median():.1f}% of actual production")

# By model type
log(f"\n  Per model type:")
for model in ["exp", "hyp", "harm"]:
    sub = res[res.best_model == model]
    if len(sub) < 2:
        continue
    log(f"    {model:5s} (n={len(sub):2d}): MAPE={sub.best_mape.median():.1f}%, "
        f"bias={sub.best_bias.median():+.1f}pp, cum_err={sub.best_cum_err_pct.median():+.1f}%")

# Residual analysis: does quality explain forecast miss?
log(f"\n{'═'*65}")
log("RESIDUAL ANALYSIS: what does Arps miss, and is it quality-linked?")
log(f"{'═'*65}")

valid = res.dropna(subset=["best_mape"])
for col, label in [("api_gravity", "API Gravity"), ("sulfur_pct", "Sulfur %"),
                    ("oil_mean", "Avg Production"), ("best_b", "b-parameter"),
                    ("field_age_mean", "Field Age")]:
    v = valid[[col, "best_mape", "best_bias", "best_cum_err_pct"]].dropna()
    if len(v) < 5:
        continue
    r_mape, p_mape = stats.pearsonr(v[col], v.best_mape)
    r_bias, p_bias = stats.pearsonr(v[col], v.best_bias)
    r_cum, p_cum = stats.pearsonr(v[col], v.best_cum_err_pct)
    sig_m = "**" if p_mape < 0.05 else "*" if p_mape < 0.1 else ""
    sig_b = "**" if p_bias < 0.05 else "*" if p_bias < 0.1 else ""
    sig_c = "**" if p_cum < 0.05 else "*" if p_cum < 0.1 else ""
    log(f"  {label:20s} vs MAPE: r={r_mape:+.3f}{sig_m:3s}  "
        f"vs bias: r={r_bias:+.3f}{sig_b:3s}  "
        f"vs cum_err: r={r_cum:+.3f}{sig_c:3s}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: 9 example fields — actual vs. all 3 Arps models
# ═══════════════════════════════════════════════════════════════════════════

# Select 9 fields across the range of fit quality
res_sorted = res.sort_values("best_r2_fit", ascending=False)
n = len(res_sorted)
indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]  # best, q1, median, q3, worst
# Add some interesting ones
examples = list(res_sorted.iloc[indices].field.unique())
for f in ["EKOFISK", "STATFJORD", "GULLFAKS", "OSEBERG", "GRANE", "HEIDRUN",
          "DRAUGEN", "NORNE", "SNORRE", "TROLL"]:
    if f in res.field.values and f not in examples:
        examples.append(f)
    if len(examples) >= 9:
        break

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

model_colors = {"exp": "#2196F3", "hyp": "#F44336", "harm": "#4CAF50"}
model_labels = {"exp": "Exponential (b=0)", "hyp": "Hyperbolic (fitted b)", "harm": "Harmonic (b=1)"}

for idx, field in enumerate(examples[:9]):
    ax = axes[idx]
    r = res[res.field == field].iloc[0]

    grp = post_peak[post_peak.field == field].sort_values("months_since_peak")
    fit_part = grp[grp.months_since_peak <= FIT_WINDOW_MONTHS]
    fore_part = grp[grp.months_since_peak > FIT_WINDOW_MONTHS]

    # Plot actual data
    ax.scatter(fit_part.months_since_peak, fit_part.oil_pct_peak,
               c="#B0BEC5", s=10, alpha=0.7, zorder=2, label="Calibration data")
    ax.scatter(fore_part.months_since_peak, fore_part.oil_pct_peak,
               c="#263238", s=10, alpha=0.7, zorder=2, label="Actual (holdout)")

    # Plot each Arps model
    t_line = np.linspace(1, grp.months_since_peak.max(), 500)

    for model_name, color in model_colors.items():
        qi = r.get(f"{model_name}_qi")
        Di = r.get(f"{model_name}_Di")
        b_val = r.get(f"{model_name}_b", 0)
        if pd.isna(qi) or pd.isna(Di):
            continue

        if model_name == "exp":
            q_line = arps_exp(t_line, qi, Di)
        elif model_name == "hyp":
            q_line = arps_hyp(t_line, qi, Di, b_val)
        else:
            q_line = arps_harm(t_line, qi, Di)

        q_line = np.clip(q_line, 0, None)
        is_best = (model_name == r.best_model)
        lw = 2.5 if is_best else 1.0
        ls = "-" if is_best else "--"
        alpha = 0.9 if is_best else 0.5

        r2 = r.get(f"{model_name}_r2_fit", np.nan)
        mape = r.get(f"{model_name}_mape", np.nan)
        lbl = f"{model_labels[model_name]}"
        if is_best:
            lbl += f" ★ (MAPE={mape:.0f}%)"

        ax.plot(t_line, q_line, color=color, linewidth=lw, linestyle=ls,
                alpha=alpha, label=lbl, zorder=3)

    # Cutoff line
    ax.axvline(FIT_WINDOW_MONTHS, color="#FF9800", linewidth=1.5, linestyle=":", alpha=0.6)

    api = r.api_gravity
    best = r.best_model
    r2_best = r.best_r2_fit
    ax.set_title(f"{field}  (API={api:.0f}°, best={best}, R²={r2_best:.2f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Months since peak", fontsize=8)
    ax.set_ylabel("% of peak production", fontsize=8)
    ax.legend(fontsize=5.5, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

    # Annotate fitted parameters
    b_hyp = r.get("hyp_b", np.nan)
    Di_best = r.get("best_Di", np.nan)
    txt = f"Di={Di_best:.4f}/mnd\n({Di_best*12:.3f}/yr)"
    if best == "hyp":
        txt += f"\nb={b_hyp:.2f}"
    ax.text(0.55, 0.95, txt, transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

fig.suptitle("Arps Decline Curve Fits — Calibrate on 5yr Post-Peak, Forecast Remaining Life",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(RESULTS / "fig_arps_pure_examples.png", **SAVEKW)
plt.close()
log("\nSaved fig_arps_pure_examples.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Residual analysis — forecast error patterns
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: Forecast error over time (do models systematically under/over-predict?)
ax = axes[0, 0]
for field in res.field:
    r = res[res.field == field].iloc[0]
    best = r.best_model
    if pd.isna(best):
        continue

    grp = post_peak[post_peak.field == field].sort_values("months_since_peak")
    fore = grp[grp.months_since_peak > FIT_WINDOW_MONTHS]

    t = fore.months_since_peak.values.astype(float)
    q_actual = fore.oil_pct_peak.values

    qi = r[f"{best}_qi"]
    Di = r[f"{best}_Di"]
    b_val = r[f"{best}_b"]

    if best == "exp":
        q_pred = arps_exp(t, qi, Di)
    elif best == "hyp":
        q_pred = arps_hyp(t, qi, Di, b_val)
    else:
        q_pred = arps_harm(t, qi, Di)

    errors = q_actual - np.clip(q_pred, 0, None)
    ax.plot(t, errors, alpha=0.15, linewidth=0.8, color="#1565C0")

# Average error curve
all_errors = []
for field in res.field:
    r = res[res.field == field].iloc[0]
    best = r.best_model
    if pd.isna(best):
        continue
    grp = post_peak[post_peak.field == field].sort_values("months_since_peak")
    fore = grp[grp.months_since_peak > FIT_WINDOW_MONTHS]
    t = fore.months_since_peak.values.astype(float)
    q_act = fore.oil_pct_peak.values
    qi, Di, b_val = r[f"{best}_qi"], r[f"{best}_Di"], r[f"{best}_b"]
    if best == "exp":
        q_p = arps_exp(t, qi, Di)
    elif best == "hyp":
        q_p = arps_hyp(t, qi, Di, b_val)
    else:
        q_p = arps_harm(t, qi, Di)
    for ti, ei in zip(t.astype(int), q_act - np.clip(q_p, 0, None)):
        all_errors.append({"t": ti, "error": ei})

err_df = pd.DataFrame(all_errors)
avg_err = err_df.groupby("t")["error"].median()
avg_smooth = avg_err.rolling(6, min_periods=3, center=True).median()
ax.plot(avg_smooth.index, avg_smooth.values, color="#C62828", linewidth=2.5, label="Median error")

ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Months since peak")
ax.set_ylabel("Actual − Predicted (pp of peak)")
ax.set_title("Forecast Error Over Time")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.text(0.03, 0.03, ">0 = Arps under-predicts\n<0 = Arps over-predicts",
        transform=ax.transAxes, fontsize=7, alpha=0.5, va="bottom")

# 2b: Cumulative error vs API
ax = axes[0, 1]
v = res.dropna(subset=["api_gravity", "best_cum_err_pct"])
ax.scatter(v.api_gravity, v.best_cum_err_pct, c="#1565C0", alpha=0.6, s=50, edgecolors="white")
slope, intercept, r_val, p_val, _ = stats.linregress(v.api_gravity, v.best_cum_err_pct)
x_fit = np.linspace(v.api_gravity.min(), v.api_gravity.max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, color="#455A64", linewidth=2)
sig = "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r={r_val:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("API Gravity (°)")
ax.set_ylabel("Cumulative Forecast Error (%)")
ax.set_title("API → How Much Does Arps Miss?")
ax.grid(True, alpha=0.2)
for _, row in v.nlargest(3, "best_cum_err_pct").iterrows():
    ax.annotate(row.field, (row.api_gravity, row.best_cum_err_pct),
                fontsize=7, alpha=0.6, xytext=(5, 5), textcoords="offset points")

# 2c: Cumulative error vs field size
ax = axes[1, 0]
v2 = res.dropna(subset=["oil_mean", "best_cum_err_pct"])
ax.scatter(v2.oil_mean, v2.best_cum_err_pct, c="#EF6C00", alpha=0.6, s=50, edgecolors="white")
slope, intercept, r_val, p_val, _ = stats.linregress(v2.oil_mean, v2.best_cum_err_pct)
x_fit = np.linspace(v2.oil_mean.min(), v2.oil_mean.max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, color="#455A64", linewidth=2)
sig = "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r={r_val:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Avg Production (Mill Sm³/mnd)")
ax.set_ylabel("Cumulative Forecast Error (%)")
ax.set_title("Field Size → How Much Does Arps Miss?")
ax.grid(True, alpha=0.2)

# 2d: In-sample R² vs out-of-sample MAPE (does good fit = good forecast?)
ax = axes[1, 1]
v3 = res.dropna(subset=["best_r2_fit", "best_mape"])
ax.scatter(v3.best_r2_fit, v3.best_mape, c="#2E7D32", alpha=0.6, s=50, edgecolors="white")
slope, intercept, r_val, p_val, _ = stats.linregress(v3.best_r2_fit, v3.best_mape)
x_fit = np.linspace(v3.best_r2_fit.min(), v3.best_r2_fit.max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, color="#455A64", linewidth=2)
sig = "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r={r_val:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("In-Sample R²")
ax.set_ylabel("Out-of-Sample MAPE (%)")
ax.set_title("Good Fit ≠ Good Forecast?")
ax.grid(True, alpha=0.2)
for _, row in v3.nlargest(3, "best_mape").iterrows():
    ax.annotate(row.field, (row.best_r2_fit, row.best_mape),
                fontsize=7, alpha=0.6, xytext=(5, 5), textcoords="offset points")

fig.suptitle("Arps Forecast Residual Analysis — What Does the Physics Miss?",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_arps_pure_residuals.png", **SAVEKW)
plt.close()
log("Saved fig_arps_pure_residuals.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Fitted parameters overview
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3a: Di distribution by model type
ax = axes[0]
for model, color in model_colors.items():
    Di_vals = res[f"{model}_Di"].dropna() * 12  # convert to annual
    if len(Di_vals) > 0:
        ax.hist(Di_vals, bins=12, color=color, alpha=0.5, label=f"{model.title()} (n={len(Di_vals)})",
                edgecolor="white")
ax.set_xlabel("Initial Decline Rate Di (yr⁻¹)")
ax.set_ylabel("Frequency")
ax.set_title("Fitted Di Across Models")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# 3b: Hyperbolic b distribution with reference lines
ax = axes[1]
b_vals = res["hyp_b"].dropna()
ax.hist(b_vals, bins=15, color="#795548", alpha=0.8, edgecolor="white")
ax.axvline(0.5, color="#2196F3", linewidth=2, linestyle="--", label="b=0.5 (solution-gas drive)")
ax.axvline(1.0, color="#F44336", linewidth=2, linestyle="--", label="b=1.0 (water drive)")
ax.set_xlabel("Arps b-parameter")
ax.set_ylabel("Frequency")
ax.set_title("Hyperbolic b Distribution")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

# 3c: MAPE by model type
ax = axes[2]
model_mapes = {}
for model in ["exp", "hyp", "harm"]:
    sub = res[res.best_model == model]["best_mape"].dropna()
    if len(sub) > 1:
        model_mapes[model] = sub

if model_mapes:
    bp = ax.boxplot(model_mapes.values(), labels=[k.title() for k in model_mapes.keys()],
                    patch_artist=True, widths=0.5)
    for patch, (_, color) in zip(bp["boxes"], model_colors.items()):
        if patch:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

ax.set_ylabel("Out-of-Sample MAPE (%)")
ax.set_title("Forecast Accuracy by Model Type")
ax.grid(True, alpha=0.2)

fig.suptitle("Arps Decline Curve — Fitted Parameters & Model Comparison",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_arps_pure_summary.png", **SAVEKW)
plt.close()
log("Saved fig_arps_pure_summary.png")

# ── Per-field table ─────────────────────────────────────────────────────────

log(f"\n{'═'*65}")
log("PER-FIELD RESULTS (sorted by forecast MAPE)")
log(f"{'═'*65}")
log(f"  {'Field':20s} {'Best':5s} {'b':>5s} {'Di/yr':>7s} {'R²fit':>6s} {'MAPE%':>6s} {'Bias':>6s} {'Cum%':>6s} {'API':>5s}")
log(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")

for _, r in res.sort_values("best_mape").iterrows():
    log(f"  {r.field:20s} {r.best_model:>5s} {r.best_b:>5.2f} "
        f"{r.best_Di*12:>7.4f} {r.best_r2_fit:>6.3f} {r.best_mape:>6.1f} "
        f"{r.best_bias:>+6.1f} {r.best_cum_err_pct:>+6.1f} {r.api_gravity:>5.1f}")

with open(RESULTS / "arps_pure_summary.txt", "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved arps_pure_summary.txt ({len(lines)} lines)")
