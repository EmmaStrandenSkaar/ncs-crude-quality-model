"""
Script 21: Lifecycle Sub-models — Ramp, Plateau, Ex-ante Decline
═══════════════════════════════════════════════════════════════════════════

Tre regresjonsmodeller med KUN ex-ante variabler:

  RAMP  — predikerer ramp-up tid fra førsteolje til peak
          target: ramp_length_months
          features: facility, log_recoverable, decade, log_n_wells

  PLATÅ — predikerer platå-varighet
          target: plateau_length_months
          features: log_recoverable, log_n_wells, facility, water_depth

  DECLINE — predikerer årlig decline rate FØR feltet er i produksjon
          target: D_decline_fit (eller D_annual)
          features: api_gravity, facility, log_recoverable, operator

Output:
  - data/submodel_ramp.pkl
  - data/submodel_plateau.pkl
  - data/submodel_decline_exante.pkl
  - results/fig_lifecycle_submodels.png
"""

import json, warnings, pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
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
log("SCRIPT 21: LIFECYCLE SUB-MODELS")
log("Ramp + Platå + Ex-ante Decline (alle ex-ante features)")
log("═" * 80)

# Load data
typecurve = pd.read_csv(DATA / "typecurve_library.csv")
master = pd.read_csv(DATA / "master_fluid_library_v51.csv")
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")
if "DatesyncNPD" in reserves.columns:
    reserves = reserves.sort_values("DatesyncNPD")
res_summary = (reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]]
               .rename(columns={"fldName": "field", "fldRecoverableOil": "recoverable_msm3"}))

# Merge
typecurve = typecurve.merge(res_summary, on="field", how="left")
api_map = master.set_index("field")["api_gravity"].to_dict()
typecurve["api_v51"] = typecurve.field.map(api_map).fillna(typecurve.get("api_gravity"))

# Filter
df = typecurve.dropna(subset=["recoverable_msm3", "facility_type"]).copy()
df = df[df.recoverable_msm3 > 0.5]
df = df[df.n_wells_total > 0]
df["api_use"] = df.api_v51.fillna(df.api_v51.median())
log(f"\nGrunn-data: {len(df)} felt")

# Common feature builder
def build_features(data, include_api=False, include_operator=False):
    """Build ex-ante feature matrix."""
    data = data.copy()
    data["log_recoverable"] = np.log(data.recoverable_msm3)
    data["log_n_wells"] = np.log(data.n_wells_total.clip(lower=1))
    data["log_water_depth"] = np.log(data.water_depth.fillna(data.water_depth.median()).clip(lower=1))
    data["decade_scaled"] = (data.first_year // 10 * 10 - 1980) / 10

    fac_d = pd.get_dummies(data.facility_type, prefix="fac", drop_first=True).astype(int)
    data = pd.concat([data, fac_d], axis=1)

    base_cols = ["log_recoverable", "log_n_wells", "log_water_depth", "decade_scaled"]
    fac_cols = list(fac_d.columns)

    cols = base_cols + fac_cols
    if include_api:
        cols.append("api_use")
    if include_operator:
        # Group small operators
        op_counts = data.operator.value_counts()
        major = op_counts[op_counts >= 4].index.tolist()
        data["op_grouped"] = data.operator.where(data.operator.isin(major), "Other")
        op_d = pd.get_dummies(data.op_grouped, prefix="op", drop_first=True).astype(int)
        data = pd.concat([data, op_d], axis=1)
        cols += list(op_d.columns)

    return data, cols

# Helper: fit + LOO-CV + bootstrap
def fit_eval(X, y, name, feature_cols):
    """Fit OLS, LOO-CV, bootstrap CIs."""
    lr = LinearRegression().fit(X, y)
    in_R2 = lr.score(X, y)

    # LOO-CV
    loo = LeaveOneOut()
    preds_cv = np.zeros(len(y))
    for tr, te in loo.split(X):
        lr_cv = LinearRegression().fit(X[tr], y[tr])
        preds_cv[te] = lr_cv.predict(X[te])
    cv_R2 = 1 - ((y - preds_cv)**2).sum() / ((y - y.mean())**2).sum()
    rmse = np.sqrt(((y - preds_cv)**2).mean())

    # Bootstrap CIs
    np.random.seed(42)
    boot = []
    for _ in range(1500):
        idx = np.random.choice(len(y), len(y), replace=True)
        try:
            lr_b = LinearRegression().fit(X[idx], y[idx])
            boot.append(np.concatenate([[lr_b.intercept_], lr_b.coef_]))
        except Exception:
            pass
    boot = np.array(boot)
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)

    log(f"\n  {'Variabel':22s}  {'β':>9s}  {'95% CI':>22s}  {'Sig':>4s}")
    log("  " + "─" * 65)
    all_names = ["Intercept"] + feature_cols
    all_coefs = [lr.intercept_] + list(lr.coef_)
    for i, nm in enumerate(all_names):
        sig = "✓" if (ci_low[i] > 0) == (ci_high[i] > 0) else "✗"
        log(f"  {nm:22s}  {all_coefs[i]:+8.4f}  [{ci_low[i]:+7.3f}, {ci_high[i]:+7.3f}]  {sig:>4s}")
    log(f"\n  In-sample R²:  {in_R2:.3f}")
    log(f"  LOO CV R²:     {cv_R2:.3f}")
    log(f"  RMSE:          {rmse:.3f}")

    return {
        "name": name,
        "model": lr,
        "feature_cols": feature_cols,
        "coefficients": dict(zip(feature_cols, lr.coef_)),
        "intercept": float(lr.intercept_),
        "in_R2": float(in_R2),
        "cv_R2": float(cv_R2),
        "rmse": float(rmse),
        "ci_low": ci_low.tolist(),
        "ci_high": ci_high.tolist(),
        "preds_cv": preds_cv,
        "n": len(y),
    }

# ═══════════════════════════════════════════════════════════════
# MODEL 1: RAMP DURATION
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("MODEL 1: RAMP DURATION")
log("═" * 80)

ramp_df = df.dropna(subset=["ramp_length_months"]).copy()
ramp_df, ramp_cols = build_features(ramp_df, include_api=False, include_operator=False)
# Target: ramp + 1 for log-transform (some fields have ramp=0)
ramp_df["log_ramp_p1"] = np.log(ramp_df.ramp_length_months + 1)
X = ramp_df[ramp_cols].fillna(0).values
y = ramp_df["log_ramp_p1"].values
log(f"\nN = {len(ramp_df)}  features = {len(ramp_cols)}")
ramp_result = fit_eval(X, y, "ramp", ramp_cols)
ramp_df["pred_log_ramp_p1"] = ramp_result["model"].predict(X)
ramp_df["pred_ramp"] = np.exp(ramp_df.pred_log_ramp_p1) - 1
ramp_df["pred_ramp_cv"] = np.exp(ramp_result["preds_cv"]) - 1

# ═══════════════════════════════════════════════════════════════
# MODEL 2: PLATEAU DURATION
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("MODEL 2: PLATEAU DURATION")
log("═" * 80)

plat_df = df.dropna(subset=["plateau_length_months"]).copy()
plat_df = plat_df[plat_df.plateau_length_months > 0]
plat_df, plat_cols = build_features(plat_df, include_api=False, include_operator=False)
plat_df["log_plat_p1"] = np.log(plat_df.plateau_length_months + 1)
X = plat_df[plat_cols].fillna(0).values
y = plat_df["log_plat_p1"].values
log(f"\nN = {len(plat_df)}  features = {len(plat_cols)}")
plat_result = fit_eval(X, y, "plateau", plat_cols)
plat_df["pred_plat_cv"] = np.exp(plat_result["preds_cv"]) - 1

# ═══════════════════════════════════════════════════════════════
# MODEL 3: EX-ANTE DECLINE
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("MODEL 3: EX-ANTE DECLINE RATE")
log("═" * 80)

dec_df = df.dropna(subset=["D_decline_fit"]).copy()
dec_df = dec_df[dec_df.D_decline_fit > 0]
dec_df = dec_df[dec_df.D_decline_fit < 1.5]  # exclude extreme outliers
dec_df, dec_cols = build_features(dec_df, include_api=True, include_operator=True)
dec_df["log_D"] = np.log(dec_df.D_decline_fit)
X = dec_df[dec_cols].fillna(0).values
y = dec_df["log_D"].values
log(f"\nN = {len(dec_df)}  features = {len(dec_cols)}")
dec_result = fit_eval(X, y, "decline", dec_cols)
dec_df["pred_log_D"] = dec_result["model"].predict(X)
dec_df["pred_D"] = np.exp(dec_df.pred_log_D)
dec_df["pred_D_cv"] = np.exp(dec_result["preds_cv"])

# ═══════════════════════════════════════════════════════════════
# SAVE MODELS
# ═══════════════════════════════════════════════════════════════
for r in [ramp_result, plat_result, dec_result]:
    # Save without sklearn model (just store coefficients for portability)
    artifact = {k: v for k, v in r.items() if k not in ["model", "preds_cv"]}
    with open(DATA / f"submodel_{r['name']}.pkl", "wb") as f:
        pickle.dump(artifact, f)
    log(f"\nSaved: submodel_{r['name']}.pkl")

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Lifecycle Sub-models — Ex-ante Forecasts for Ramp / Platå / Decline",
             fontsize=15, fontweight="bold", y=1.0)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

# Panel 1: Ramp CV
ax = fig.add_subplot(gs[0, 0])
ax.scatter(ramp_df.pred_ramp_cv, ramp_df.ramp_length_months, s=40, alpha=0.7,
           c="#FF9800", edgecolors="white", lw=0.4)
lims = [0, ramp_df.ramp_length_months.max() * 1.1]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
ax.set_xlabel("Predikert ramp (mnd, CV)")
ax.set_ylabel("Faktisk ramp (mnd)")
ax.set_title(f"Ramp Duration\nCV R² = {ramp_result['cv_R2']:.3f}, n={ramp_result['n']}",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 2: Plateau CV
ax = fig.add_subplot(gs[0, 1])
ax.scatter(plat_df.pred_plat_cv, plat_df.plateau_length_months, s=40, alpha=0.7,
           c="#2E7D32", edgecolors="white", lw=0.4)
lims = [0, plat_df.plateau_length_months.max() * 1.1]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
ax.set_xlabel("Predikert platå (mnd, CV)")
ax.set_ylabel("Faktisk platå (mnd)")
ax.set_title(f"Plateau Duration\nCV R² = {plat_result['cv_R2']:.3f}, n={plat_result['n']}",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 3: Decline CV
ax = fig.add_subplot(gs[0, 2])
ax.scatter(dec_df.pred_D_cv, dec_df.D_decline_fit, s=40, alpha=0.7,
           c="#E91E63", edgecolors="white", lw=0.4)
lims = [0, max(dec_df.D_decline_fit.max(), dec_df.pred_D_cv.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
ax.set_xlabel("Predikert D (CV)")
ax.set_ylabel("Faktisk D")
ax.set_title(f"Ex-ante Decline\nCV R² = {dec_result['cv_R2']:.3f}, n={dec_result['n']}",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 4-6: Coefficients
for i, (result, color) in enumerate([(ramp_result, "#FF9800"),
                                       (plat_result, "#2E7D32"),
                                       (dec_result, "#E91E63")]):
    ax = fig.add_subplot(gs[1, i])
    names = ["Intercept"] + result["feature_cols"]
    coefs = [result["intercept"]] + list(result["coefficients"].values())
    ci_l = result["ci_low"]
    ci_h = result["ci_high"]
    errs_low = [coefs[k] - ci_l[k] for k in range(len(names))]
    errs_high = [ci_h[k] - coefs[k] for k in range(len(names))]
    colors = []
    for k in range(len(names)):
        if (ci_l[k] > 0) == (ci_h[k] > 0):
            colors.append(color)
        else:
            colors.append("#9E9E9E")
    y_pos = np.arange(len(names))
    ax.barh(y_pos, coefs, color=colors, alpha=0.85, xerr=[errs_low, errs_high], capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Koeffisient (95% CI)")
    ax.set_title(f"{result['name'].title()}: koeffisienter", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_lifecycle_submodels.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_lifecycle_submodels.png")

with open(RESULTS / "lifecycle_submodels.txt", "w") as f:
    f.write("\n".join(lines))
