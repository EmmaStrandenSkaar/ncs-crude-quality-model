"""
Script 20b: Simplified Peak Forecast Model
═══════════════════════════════════════════════════════════════════════════

Forenklet modell — kun 3 input-variabler (alle ex-ante):
  - log(recoverable_oil_msm3)
  - log(n_wells_total)
  - facility_type dummies

Tester om n_wells blir signifikant når vi fjerner støy-variabler.
"""

import json, warnings, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir"

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

log("═" * 80)
log("SCRIPT 20b: FORENKLET PEAK FORECAST")
log("─ Kun 3 features: log_recoverable, log_n_wells, facility_type ─")
log("═" * 80)

# Load data
typecurve = pd.read_csv(DATA / "typecurve_library.csv")
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")
if "DatesyncNPD" in reserves.columns:
    reserves = reserves.sort_values("DatesyncNPD")
res_summary = (reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]]
               .rename(columns={"fldName": "field", "fldRecoverableOil": "recoverable_msm3"}))
typecurve = typecurve.merge(res_summary, on="field", how="left")

# Filter
df = typecurve.dropna(subset=["peak_oil_msm3", "recoverable_msm3", "facility_type", "n_wells_total"]).copy()
df = df[df.recoverable_msm3 > 0.5]
df = df[df.peak_oil_msm3 > 0.01]
df = df[df.n_wells_total > 0]

log(f"\nFelt: {len(df)}")
log(f"Facility-distribusjon:")
log(df.facility_type.value_counts().to_string())

# Features
df["log_peak"] = np.log(df.peak_oil_msm3)
df["log_recoverable"] = np.log(df.recoverable_msm3)
df["log_n_wells"] = np.log(df.n_wells_total)

facility_dummies = pd.get_dummies(df.facility_type, prefix="fac", drop_first=True).astype(int)
df = pd.concat([df, facility_dummies], axis=1)

feature_cols = ["log_recoverable", "log_n_wells"] + list(facility_dummies.columns)
X = df[feature_cols].values
y = df["log_peak"].values

log(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
log(f"Parametere: {len(feature_cols) + 1} (intercept)")
log(f"Ratio N:p: {len(df) / (len(feature_cols) + 1):.1f}:1")

# Fit
lr = LinearRegression().fit(X, y)
df["log_peak_pred"] = lr.predict(X)
df["peak_pred"] = np.exp(df.log_peak_pred)

in_R2 = lr.score(X, y)
df["resid_pct"] = (df.peak_pred - df.peak_oil_msm3) / df.peak_oil_msm3
median_ape = df.resid_pct.abs().median() * 100

# LOO-CV
loo = LeaveOneOut()
log_preds_cv = np.zeros(len(y))
for tr, te in loo.split(X):
    lr_cv = LinearRegression().fit(X[tr], y[tr])
    log_preds_cv[te] = lr_cv.predict(X[te])
cv_R2_log = 1 - ((y - log_preds_cv)**2).sum() / ((y - y.mean())**2).sum()
cv_preds = np.exp(log_preds_cv)
cv_R2_lin = 1 - ((df.peak_oil_msm3 - cv_preds)**2).sum() / ((df.peak_oil_msm3 - df.peak_oil_msm3.mean())**2).sum()
cv_resid_pct = (cv_preds - df.peak_oil_msm3.values) / df.peak_oil_msm3.values
cv_median_ape = np.median(np.abs(cv_resid_pct)) * 100

df["peak_pred_cv"] = cv_preds

log(f"\n── YTELSE ──")
log(f"  In-sample R² (log):    {in_R2:.3f}")
log(f"  LOO CV R² (log):       {cv_R2_log:.3f}")
log(f"  LOO CV R² (linear):    {cv_R2_lin:.3f}")
log(f"  In-sample median APE:  {median_ape:.0f}%")
log(f"  CV median APE:         {cv_median_ape:.0f}%")

# Bootstrap CIs
np.random.seed(42)
boot_coef = []
for _ in range(2000):
    idx = np.random.choice(len(df), len(df), replace=True)
    try:
        lr_b = LinearRegression().fit(X[idx], y[idx])
        boot_coef.append(np.concatenate([[lr_b.intercept_], lr_b.coef_]))
    except Exception:
        pass
boot_coef = np.array(boot_coef)
ci_low = np.percentile(boot_coef, 2.5, axis=0)
ci_high = np.percentile(boot_coef, 97.5, axis=0)

log(f"\n── KOEFFISIENTER MED 95% BOOTSTRAP CIs ──")
log(f"  {'Variable':22s} {'β':>9s}  {'95% CI':>22s}  {'Sig.':>5s}")
log("─" * 70)
all_names = ["Intercept"] + feature_cols
all_coefs = [lr.intercept_] + list(lr.coef_)
for i, name in enumerate(all_names):
    sig = "✓" if (ci_low[i] > 0) == (ci_high[i] > 0) else "✗"
    log(f"  {name:22s}  {all_coefs[i]:+8.4f}  [{ci_low[i]:+7.4f}, {ci_high[i]:+7.4f}]  {sig}")

# t-stats via OLS
import statsmodels.api as sm
X_sm = sm.add_constant(X)
ols = sm.OLS(y, X_sm).fit()
log(f"\n── OLS T-STATISTIKK ──")
log(f"  {'Variable':22s} {'t':>7s}  {'p-verdi':>10s}")
log("─" * 50)
for i, name in enumerate(all_names):
    p = ols.pvalues[i]
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    log(f"  {name:22s}  {ols.tvalues[i]:+6.2f}  {p:>10.4f}  {sig}")

# Aker BP
log(f"\n══════════════════════════════════════════════════════════════════════════")
log(f"AKER BP — Forenklet modell")
log(f"══════════════════════════════════════════════════════════════════════════")
akbp = df[df.operator.str.contains("Aker BP", na=False)].copy()
log(f"\n{'Felt':18s} {'R (MSm³)':>9s} {'Wells':>6s} {'Actual':>8s} {'CV Pred':>9s} {'Err':>7s}")
log("─" * 70)
for _, r in akbp.sort_values("peak_oil_msm3", ascending=False).iterrows():
    err_pct = (r.peak_pred_cv - r.peak_oil_msm3) / r.peak_oil_msm3 * 100
    log(f"  {r.field:16s} {r.recoverable_msm3:9.1f} {r.n_wells_total:6.0f} "
        f"{r.peak_oil_msm3:8.3f} {r.peak_pred_cv:9.3f} {err_pct:+6.0f}%")

akbp_median_err = (akbp.peak_pred_cv - akbp.peak_oil_msm3).abs().median() / akbp.peak_oil_msm3.median()
log(f"\n  Aker BP CV median |%-err|: {akbp_median_err*100:.0f}%")

# Sammenligning med komplisert modell
log(f"\n══════════════════════════════════════════════════════════════════════════")
log(f"SAMMENLIGNING: forenklet vs. fullmodell")
log(f"══════════════════════════════════════════════════════════════════════════")
log(f"\n  {'Metrikk':30s} {'Full (13 vars)':>15s} {'Forenklet (6 vars)':>18s}")
log(f"  {'CV R² (log)':30s} {0.800:15.3f} {cv_R2_log:18.3f}")
log(f"  {'CV R² (linear)':30s} {0.686:15.3f} {cv_R2_lin:18.3f}")
log(f"  {'CV median APE':30s} {30:>14d}% {cv_median_ape:>17.0f}%")
log(f"  {'N:p ratio':30s} {64/14:15.1f} {len(df)/(len(feature_cols)+1):18.1f}")

# Save
df_out = df[["field", "operator", "facility_type", "recoverable_msm3", "n_wells_total",
             "peak_oil_msm3", "peak_pred", "peak_pred_cv", "resid_pct"]].copy()
df_out["error_pct"] = df_out.resid_pct * 100
df_out = df_out.drop(columns=["resid_pct"])
df_out.to_csv(DATA / "peak_predictions_simplified.csv", index=False)
log(f"\nSaved: peak_predictions_simplified.csv")

model_artifact = {
    "coefficients": dict(zip(feature_cols, lr.coef_)),
    "intercept": float(lr.intercept_),
    "feature_cols": feature_cols,
    "facility_categories": list(facility_dummies.columns),
    "in_sample_R2": float(in_R2),
    "cv_R2_log": float(cv_R2_log),
    "cv_R2_linear": float(cv_R2_lin),
    "cv_median_ape": float(cv_median_ape),
    "n_fields": int(len(df)),
    "ci_low": ci_low.tolist(),
    "ci_high": ci_high.tolist(),
}
with open(DATA / "peak_forecast_simplified.pkl", "wb") as f:
    pickle.dump(model_artifact, f)
log(f"Saved: peak_forecast_simplified.pkl")

# Quick figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Forenklet Peak Forecast — 3 ex-ante variabler", fontsize=14, fontweight="bold", y=1.02)

# Panel 1: CV vs actual (log-log)
ax = axes[0]
others = df[~df.operator.str.contains("Aker BP", na=False)]
ax.scatter(others.peak_pred_cv, others.peak_oil_msm3, c="lightgray", s=40, alpha=0.6, label="Andre NCS")
ax.scatter(akbp.peak_pred_cv, akbp.peak_oil_msm3, c="#E91E63", s=60, alpha=0.85,
           edgecolors="white", lw=0.5, label="Aker BP")
for _, r in akbp.iterrows():
    ax.annotate(r.field, (r.peak_pred_cv, r.peak_oil_msm3), fontsize=6, alpha=0.85,
                xytext=(4, 3), textcoords="offset points")
lims = [df.peak_oil_msm3.min() * 0.5, df.peak_oil_msm3.max() * 2]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Predikert peak (LOO CV, MSm³/mnd)")
ax.set_ylabel("Faktisk peak (MSm³/mnd)")
ax.set_title(f"CV R² (log)={cv_R2_log:.3f}  median APE={cv_median_ape:.0f}%",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 2: Coefficient comparison
ax = axes[1]
y_pos = np.arange(len(all_names))
errs_low = [all_coefs[i] - ci_low[i] for i in range(len(all_names))]
errs_high = [ci_high[i] - all_coefs[i] for i in range(len(all_names))]
colors_c = []
for i in range(len(all_names)):
    if (ci_low[i] > 0) == (ci_high[i] > 0):
        colors_c.append("#2E7D32" if all_coefs[i] > 0 else "#C62828")
    else:
        colors_c.append("#9E9E9E")
ax.barh(y_pos, all_coefs, color=colors_c, alpha=0.85,
        xerr=[errs_low, errs_high], capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels(all_names, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Koeffisient (95% bootstrap CI)")
ax.set_title("Koeffisienter — grønn/rød = signifikant, grå = ikke", fontsize=11, fontweight="bold")

# Panel 3: Aker BP per felt
ax = axes[2]
akbp_s = akbp.sort_values("peak_oil_msm3")
y_pos = np.arange(len(akbp_s))
ax.barh(y_pos - 0.18, akbp_s.peak_oil_msm3, 0.36, color="#1565C0", alpha=0.85, label="Faktisk")
ax.barh(y_pos + 0.18, akbp_s.peak_pred_cv, 0.36, color="#2E7D32", alpha=0.85, label="CV-pred")
ax.set_yticks(y_pos)
ax.set_yticklabels(akbp_s.field, fontsize=8)
ax.set_xlabel("Peak (MSm³/mnd)")
ax.set_title("Aker BP", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(RESULTS / "fig_peak_forecast_simplified.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_peak_forecast_simplified.png")

with open(RESULTS / "peak_forecast_simplified.txt", "w") as f:
    f.write("\n".join(lines))
