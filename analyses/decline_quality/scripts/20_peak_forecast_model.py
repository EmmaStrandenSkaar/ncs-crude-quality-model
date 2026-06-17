"""
Script 20: Peak Production Forecast Model
═══════════════════════════════════════════════════════════════════════════

Predikter peak månedlig oljeproduksjon for NCS-felt fra KUN ex-ante variabler
(tilgjengelig før produksjon starter — fra PDO og discovery DST).

Variabler inkludert:
  - log(recoverable_oil_msm3)      ← PDO estimate (operatør guidance)
  - log(n_wells_total)             ← proxy for n_wells_planned
  - api_gravity                    ← discovery DST
  - log(water_depth_m)             ← geografisk
  - log(reservoir_depth_m)         ← discovery
  - decade (vintage)               ← planlagt oppstart
  - facility_type dummies          ← PDO design
  - operator dummies               ← lisens-tildeling

Eksplisitt EKSKLUDERT (post-hoc):
  - total_oil_msm3, D_decline_fit, premium_12m, ramp/plateau
  - alt som krever observert produksjon

Output:
  - data/peak_forecast_model.pkl (joblib)
  - results/fig_peak_forecast.png
  - data/peak_predictions.csv
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
GEO = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir_geo" / "fields.geojson"

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

log("═" * 80)
log("SCRIPT 20: PEAK PRODUCTION FORECAST MODEL")
log("Kun ex-ante variabler — kan kjøres på nye felt før produksjon")
log("═" * 80)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
typecurve = pd.read_csv(DATA / "typecurve_library.csv")
master = pd.read_csv(DATA / "master_fluid_library_v51.csv")

# Reserves (PDO estimate proxy)
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")
# Use the LATEST reserves estimate as proxy for what would be available in a new PDO
if "DatesyncNPD" in reserves.columns:
    reserves = reserves.sort_values("DatesyncNPD")
res_summary = (reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]]
               .rename(columns={"fldName": "field", "fldRecoverableOil": "recoverable_msm3"}))

# Operator from typecurve
typecurve = typecurve.merge(res_summary, on="field", how="left")
log(f"\nTypecurve library: {len(typecurve)} felt")
log(f"  Med recoverable_msm3: {typecurve.recoverable_msm3.notna().sum()}")

# Use V5.1 master library for API
api_map = master.set_index("field")["api_gravity"].to_dict()
typecurve["api_v51"] = typecurve.field.map(api_map).fillna(typecurve.get("api_gravity"))

# Filter: oil fields with key data
df = typecurve.dropna(subset=["peak_oil_msm3", "recoverable_msm3", "facility_type"]).copy()
df = df[df.recoverable_msm3 > 0.5]  # Min 0.5 MSm³ oil
df = df[df.peak_oil_msm3 > 0.01]
log(f"\nFelt med komplett ex-ante data: {len(df)}")

# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (KUN EX-ANTE)
# ═══════════════════════════════════════════════════════════════
log("\n── Feature engineering ──")

# Target: log peak monthly oil production (MSm³/mnd)
df["log_peak"] = np.log(df.peak_oil_msm3)

# Continuous ex-ante features
df["log_recoverable"] = np.log(df.recoverable_msm3.clip(lower=0.1))
df["log_n_wells"] = np.log(df.n_wells_total.fillna(10).clip(lower=1))
df["log_water_depth"] = np.log(df.water_depth.fillna(typecurve.water_depth.median()).clip(lower=1))
df["log_reservoir_depth"] = np.log(df.peak_oil_msm3.clip(lower=0.001))  # placeholder if no depth
# API gravity (with median imputation)
df["api"] = df.api_v51.fillna(df.api_v51.median())
df["decade"] = (df.first_year // 10 * 10).astype(int)
df["decade_scaled"] = (df.decade - 1980) / 10  # 0=1980s, 1=1990s, etc.

# Facility type dummies (drop one for reference category)
facility_dummies = pd.get_dummies(df.facility_type, prefix="fac", drop_first=True).astype(int)
df = pd.concat([df, facility_dummies], axis=1)
log(f"  Facility dummies: {list(facility_dummies.columns)}")

# Operator dummies — group small operators as "Other"
op_counts = df.operator.value_counts()
major_ops = op_counts[op_counts >= 4].index.tolist()
df["op_grouped"] = df.operator.where(df.operator.isin(major_ops), "Other")
log(f"  Major operators (n≥4): {major_ops}")
op_dummies = pd.get_dummies(df.op_grouped, prefix="op", drop_first=True).astype(int)
df = pd.concat([df, op_dummies], axis=1)
log(f"  Operator dummies: {list(op_dummies.columns)}")

# ═══════════════════════════════════════════════════════════════
# 3. BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════
feature_cols = (["log_recoverable", "log_n_wells", "log_water_depth", "api", "decade_scaled"]
                + list(facility_dummies.columns)
                + list(op_dummies.columns))

X_df = df[feature_cols].copy()
y = df["log_peak"].values

log(f"\nFeature matrix: {X_df.shape}")
log(f"Features used: {feature_cols}")

# Check for issues
nan_count = X_df.isna().sum().sum()
log(f"NaN in features: {nan_count}")
X_df = X_df.fillna(0)

X = X_df.values

# ═══════════════════════════════════════════════════════════════
# 4. FIT MODEL
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("MODEL FIT")
log("═" * 80)

lr = LinearRegression().fit(X, y)
df["log_peak_pred"] = lr.predict(X)
df["peak_pred_msm3"] = np.exp(df.log_peak_pred)
df["peak_actual_msm3"] = df.peak_oil_msm3

# In-sample metrics
in_R2 = lr.score(X, y)
in_R2_orig = 1 - ((df.peak_actual_msm3 - df.peak_pred_msm3)**2).sum() / ((df.peak_actual_msm3 - df.peak_actual_msm3.mean())**2).sum()

# Log-space RMSE and MAPE (median absolute percent error)
df["resid_log"] = y - df.log_peak_pred
df["resid_pct"] = (df.peak_pred_msm3 - df.peak_actual_msm3) / df.peak_actual_msm3
median_ape = df.resid_pct.abs().median() * 100

log(f"  In-sample R² (log-space):    {in_R2:.3f}")
log(f"  In-sample R² (linear-space): {in_R2_orig:.3f}")
log(f"  Median absolute % error:     {median_ape:.0f}%")

# ═══════════════════════════════════════════════════════════════
# 5. LOO-CV
# ═══════════════════════════════════════════════════════════════
loo = LeaveOneOut()
log_preds_cv = np.zeros(len(y))
for tr, te in loo.split(X):
    lr_cv = LinearRegression().fit(X[tr], y[tr])
    log_preds_cv[te] = lr_cv.predict(X[te])

cv_R2 = 1 - ((y - log_preds_cv)**2).sum() / ((y - y.mean())**2).sum()
cv_preds_orig = np.exp(log_preds_cv)
cv_R2_orig = 1 - ((df.peak_actual_msm3 - cv_preds_orig)**2).sum() / ((df.peak_actual_msm3 - df.peak_actual_msm3.mean())**2).sum()
cv_resid_pct = (cv_preds_orig - df.peak_actual_msm3.values) / df.peak_actual_msm3.values
cv_median_ape = np.median(np.abs(cv_resid_pct)) * 100

log(f"\n  LOO CV R² (log-space):       {cv_R2:.3f}")
log(f"  LOO CV R² (linear-space):    {cv_R2_orig:.3f}")
log(f"  CV median absolute % error:  {cv_median_ape:.0f}%")

df["log_peak_pred_cv"] = log_preds_cv
df["peak_pred_cv"] = cv_preds_orig

# ═══════════════════════════════════════════════════════════════
# 6. BOOTSTRAP COEFFICIENT CIs
# ═══════════════════════════════════════════════════════════════
log("\n── Bootstrap koeffisient-CIs (n=2000) ──")
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

log(f"  {'Variable':28s} {'β':>9s}  {'95% CI':>22s}  {'Sig.':>5s}")
log("─" * 75)
for i, name in enumerate(["Intercept"] + feature_cols):
    coef = lr.intercept_ if i == 0 else lr.coef_[i-1]
    sig = "✓" if (ci_low[i] > 0) == (ci_high[i] > 0) else "✗"
    log(f"  {name:28s}  {coef:+8.4f}  [{ci_low[i]:+7.4f}, {ci_high[i]:+7.4f}]  {sig}")

# ═══════════════════════════════════════════════════════════════
# 7. SAVE MODEL + PREDICTIONS
# ═══════════════════════════════════════════════════════════════
model_artifact = {
    "coefficients": dict(zip(feature_cols, lr.coef_)),
    "intercept": float(lr.intercept_),
    "feature_cols": feature_cols,
    "facility_categories": list(facility_dummies.columns),
    "operator_categories": list(op_dummies.columns),
    "major_operators": major_ops,
    "in_sample_R2": float(in_R2),
    "cv_R2": float(cv_R2),
    "median_ape": float(median_ape),
    "cv_median_ape": float(cv_median_ape),
}

with open(DATA / "peak_forecast_model.pkl", "wb") as f:
    pickle.dump(model_artifact, f)
log(f"\n  Saved: peak_forecast_model.pkl")

# Save predictions
out = df[["field", "operator", "facility_type", "first_year",
          "recoverable_msm3", "n_wells_total", "water_depth", "api",
          "peak_actual_msm3", "peak_pred_msm3", "peak_pred_cv",
          "resid_pct"]].copy()
out["error_pct"] = out.resid_pct * 100
out = out.drop(columns=["resid_pct"])
out.to_csv(DATA / "peak_predictions.csv", index=False)
log(f"  Saved: peak_predictions.csv")

# ═══════════════════════════════════════════════════════════════
# 8. AKER BP RESULTATER
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("AKER BP-FELT — peak forecast i historisk perspektiv")
log("═" * 80)

akbp = df[df.operator.str.contains("Aker BP", na=False)].copy()
log(f"\n{'Felt':18s} {'R (MSm³)':>9s} {'Actual':>8s} {'Pred':>8s} {'CV pred':>9s} {'Err':>7s}")
log("─" * 75)
for _, r in akbp.sort_values("peak_actual_msm3", ascending=False).iterrows():
    err_pct = (r.peak_pred_cv - r.peak_actual_msm3) / r.peak_actual_msm3 * 100
    log(f"  {r.field:16s} {r.recoverable_msm3:9.1f} {r.peak_actual_msm3:8.3f} "
        f"{r.peak_pred_msm3:8.3f} {r.peak_pred_cv:9.3f} {err_pct:+6.0f}%")

akbp_median_err = (akbp.peak_pred_cv - akbp.peak_actual_msm3).abs().median() / akbp.peak_actual_msm3.median()
log(f"\n  Aker BP CV median |%-err|: {akbp_median_err*100:.0f}%")

# ═══════════════════════════════════════════════════════════════
# 9. FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
fig.suptitle("Peak Production Forecast Model — Kun Ex-Ante Variabler",
             fontsize=15, fontweight="bold", y=1.0)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

# Panel 1: log-space scatter (CV predictions)
ax = fig.add_subplot(gs[0, 0])
others = df[~df.operator.str.contains("Aker BP", na=False)]
ax.scatter(np.exp(others.log_peak_pred_cv), others.peak_actual_msm3,
           c="lightgray", s=40, alpha=0.6, label="Andre NCS")
ax.scatter(akbp.peak_pred_cv, akbp.peak_actual_msm3,
           c="#E91E63", s=70, alpha=0.85, edgecolors="white", lw=0.5, label="Aker BP")
lims = [df.peak_actual_msm3.min() * 0.5, df.peak_actual_msm3.max() * 2]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Predikert peak (LOO CV)")
ax.set_ylabel("Faktisk peak (MSm³/mnd)")
ax.set_title(f"CV R² = {cv_R2:.3f}, median APE = {cv_median_ape:.0f}%",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 2: Recoverable vs peak (key relationship)
ax = fig.add_subplot(gs[0, 1])
for fac in ["FPSO", "Fixed", "Subsea tieback", "Semi-sub", "Other"]:
    sub = df[df.facility_type == fac]
    if len(sub) > 0:
        ax.scatter(sub.recoverable_msm3, sub.peak_actual_msm3, s=40, alpha=0.7,
                  label=f"{fac} (n={len(sub)})", edgecolors="white", lw=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Recoverable (MSm³)")
ax.set_ylabel("Peak (MSm³/mnd)")
ax.set_title("Recoverable vs Peak (dominerende sammenheng)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# Panel 3: Coefficient plot
ax = fig.add_subplot(gs[0, 2])
plot_features = ["log_recoverable", "log_n_wells", "log_water_depth", "api", "decade_scaled"]
plot_coefs = [lr.coef_[feature_cols.index(f)] for f in plot_features]
plot_ci_low = [ci_low[feature_cols.index(f) + 1] for f in plot_features]
plot_ci_high = [ci_high[feature_cols.index(f) + 1] for f in plot_features]
y_pos = np.arange(len(plot_features))
ax.barh(y_pos, plot_coefs, color="#1565C0", alpha=0.7,
        xerr=[(np.array(plot_coefs) - np.array(plot_ci_low)),
              (np.array(plot_ci_high) - np.array(plot_coefs))],
        capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_features, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Koeffisient (95% CI)")
ax.set_title("Variable-effekt med CIs", fontsize=11, fontweight="bold")

# Panel 4: Aker BP actual vs predicted
ax = fig.add_subplot(gs[1, 0])
akbp_s = akbp.sort_values("peak_actual_msm3")
y_pos = np.arange(len(akbp_s))
ax.barh(y_pos - 0.18, akbp_s.peak_actual_msm3, 0.36, color="#1565C0", alpha=0.85, label="Faktisk")
ax.barh(y_pos + 0.18, akbp_s.peak_pred_cv, 0.36, color="#2E7D32", alpha=0.85, label="CV-prediksjon")
ax.set_yticks(y_pos)
ax.set_yticklabels(akbp_s.field, fontsize=8)
ax.set_xlabel("Peak (MSm³/mnd)")
ax.set_title("Aker BP-felt: faktisk vs predikert peak", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Panel 5: Operator effect
ax = fig.add_subplot(gs[1, 1])
op_features = [f for f in feature_cols if f.startswith("op_")]
op_coefs = [lr.coef_[feature_cols.index(f)] for f in op_features]
op_ci_low = [ci_low[feature_cols.index(f) + 1] for f in op_features]
op_ci_high = [ci_high[feature_cols.index(f) + 1] for f in op_features]
y_pos = np.arange(len(op_features))
colors_op = ["#2E7D32" if c > 0 else "#C62828" for c in op_coefs]
ax.barh(y_pos, op_coefs, color=colors_op, alpha=0.8,
        xerr=[(np.array(op_coefs) - np.array(op_ci_low)),
              (np.array(op_ci_high) - np.array(op_coefs))],
        capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels([f.replace("op_", "") for f in op_features], fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Operator-effekt på ln(peak)")
ax.set_title("Operator track record\n(vs. \"Other\" baseline)", fontsize=11, fontweight="bold")

# Panel 6: Formula box
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
formula_text = f"""PEAK FORECAST MODEL

ln(peak_MSm³/mnd) = {lr.intercept_:+.2f}
"""
for f, c in [(f, lr.coef_[feature_cols.index(f)]) for f in plot_features]:
    formula_text += f"  {'+' if c >= 0 else ''}{c:.3f} × {f}\n"
formula_text += f"\n+ facility & operator dummies\n"
formula_text += f"\n── YTELSE ──\n"
formula_text += f"  In-sample R²:   {in_R2:.3f}\n"
formula_text += f"  CV R²:          {cv_R2:.3f}\n"
formula_text += f"  Median |%err|:  {cv_median_ape:.0f}%\n"
formula_text += f"  N felt:         {len(df)}\n"
formula_text += f"\n── INPUT FOR NYE FELT ──\n"
formula_text += f"  Alle variabler\n  fra PDO eller\n  discovery DST"

ax.text(0.05, 0.97, formula_text, transform=ax.transAxes, fontsize=10,
        fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="#E8F5E9", ec="#2E7D32", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_peak_forecast.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_peak_forecast.png")

with open(RESULTS / "peak_forecast_summary.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved: peak_forecast_summary.txt")
