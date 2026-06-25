"""
Script 25: Lifecycle V2 — Forenklede sub-modeller med JOINT BOOTSTRAP
═══════════════════════════════════════════════════════════════════════════

Bygger 4 sub-modeller med:
  - FORENKLET feature set (basert på QA-funn)
    * Peak:     log_recoverable + log_n_wells + facility
    * Ramp:     log_recoverable ONLY (komplekse features overfitter!)
    * Plateau:  log_recoverable ONLY
    * Decline:  test enklest mulig vs forenklet

  - JOINT BOOTSTRAP for å bevare parameter-korrelasjoner
    * Samme felt-resampling brukes på tvers av alle 4 modeller
    * Når Script 26 sampler en bootstrap-fit, får den korrelerte parametere

Output:
  - data/lifecycle_v2_models.pkl  — alle modeller + bootstrap fits
"""

import warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir"

B = 2000  # bootstrap resamples

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

log("═" * 80)
log("SCRIPT 25: LIFECYCLE V2 SUB-MODELS")
log("Forenklede modeller med joint bootstrap (B=" + str(B) + ")")
log("═" * 80)

# Load data
typecurve = pd.read_csv(DATA / "typecurve_library.csv")
master = pd.read_csv(DATA / "master_fluid_library_v51.csv")
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")
if "DatesyncNPD" in reserves.columns:
    reserves = reserves.sort_values("DatesyncNPD")
res_summary = (reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]]
               .rename(columns={"fldName": "field", "fldRecoverableOil": "recoverable_msm3"}))
typecurve = typecurve.merge(res_summary, on="field", how="left")
api_map = master.set_index("field")["api_gravity"].to_dict()
typecurve["api_v51"] = typecurve.field.map(api_map).fillna(typecurve.get("api_gravity"))

# Filter — KEEP ALL FIELDS, even if some targets are NaN
# We'll handle missing targets per model
df = typecurve.dropna(subset=["recoverable_msm3", "facility_type"]).copy()
df = df[df.recoverable_msm3 > 0.5]
df = df[df.n_wells_total > 0]
df["api_use"] = df.api_v51.fillna(df.api_v51.median())

log(f"\nGrunn-data: {len(df)} felt")

# ═══════════════════════════════════════════════════════════════
# BUILD FEATURES
# ═══════════════════════════════════════════════════════════════
df["log_recoverable"] = np.log(df.recoverable_msm3)
df["log_n_wells"] = np.log(df.n_wells_total.clip(lower=1))
df["log_water_depth"] = np.log(df.water_depth.fillna(df.water_depth.median()).clip(lower=1))

fac_d = pd.get_dummies(df.facility_type, prefix="fac", drop_first=True).astype(int)
df = pd.concat([df, fac_d], axis=1)
facility_cols = list(fac_d.columns)
log(f"Facility dummies: {facility_cols}")
log(f"Facility-distribusjon:")
log(df.facility_type.value_counts().to_string())

# Operator dummies (only for models that need them, e.g. decline)
op_counts = df.operator.value_counts()
major_ops = op_counts[op_counts >= 4].index.tolist()
df["op_grouped"] = df.operator.where(df.operator.isin(major_ops), "Other")
op_d = pd.get_dummies(df.op_grouped, prefix="op", drop_first=True).astype(int)
df = pd.concat([df, op_d], axis=1)
operator_cols = list(op_d.columns)
log(f"Operator dummies (n≥4 felt): {operator_cols}")

# ═══════════════════════════════════════════════════════════════
# MODEL SPECIFICATIONS (etter QA-funn)
# ═══════════════════════════════════════════════════════════════
model_specs = {
    "peak": {
        "target_col": "log_peak",
        "target_transform": lambda x: np.log(x.peak_oil_msm3),
        "target_inverse": lambda x: np.exp(x),
        "features": ["log_recoverable", "log_n_wells"] + facility_cols,
        "filter": lambda d: d.dropna(subset=["peak_oil_msm3"]),
    },
    "ramp": {
        "target_col": "log_ramp_p1",
        "target_transform": lambda x: np.log(x.ramp_length_months + 1),
        "target_inverse": lambda x: np.exp(x) - 1,
        "features": ["log_recoverable"],  # forenklet
        "filter": lambda d: d.dropna(subset=["ramp_length_months"]),
        "cap": (3, 96),  # mnd
    },
    "plateau": {
        "target_col": "log_plat_p1",
        "target_transform": lambda x: np.log(x.plateau_length_months + 1),
        "target_inverse": lambda x: np.exp(x) - 1,
        "features": ["log_recoverable"],  # forenklet
        "filter": lambda d: d.dropna(subset=["plateau_length_months"]).query("plateau_length_months > 0"),
        "cap": (0, 84),  # mnd
    },
    "decline": {
        "target_col": "log_D",
        "target_transform": lambda x: np.log(x.D_decline_fit),
        "target_inverse": lambda x: np.exp(x),
        # Include log_n_wells and operator dummies — QA viste de var signifikante
        "features": ["log_recoverable", "log_n_wells"] + facility_cols + operator_cols,
        "filter": lambda d: (d.dropna(subset=["D_decline_fit"])
                              .query("0 < D_decline_fit < 1.5")),
        "cap": (0.02, 0.40),
    },
}

# ═══════════════════════════════════════════════════════════════
# JOINT BOOTSTRAP
# ═══════════════════════════════════════════════════════════════
# Build the common dataset that has ALL targets (for joint bootstrap)
df_full = df.dropna(subset=["peak_oil_msm3", "ramp_length_months",
                              "plateau_length_months", "D_decline_fit"]).copy()
df_full = df_full[df_full.plateau_length_months > 0]
df_full = df_full[df_full.D_decline_fit < 1.5]
df_full = df_full[df_full.D_decline_fit > 0]

df_full["log_peak"] = np.log(df_full.peak_oil_msm3)
df_full["log_ramp_p1"] = np.log(df_full.ramp_length_months + 1)
df_full["log_plat_p1"] = np.log(df_full.plateau_length_months + 1)
df_full["log_D"] = np.log(df_full.D_decline_fit)

log(f"\nJoint bootstrap dataset (alle 4 targets): {len(df_full)} felt")

# Generate B bootstrap indices (same for ALL models)
np.random.seed(42)
bootstrap_indices = [np.random.choice(len(df_full), len(df_full), replace=True)
                     for _ in range(B)]

# ═══════════════════════════════════════════════════════════════
# FIT BASE MODELS + BOOTSTRAP
# ═══════════════════════════════════════════════════════════════
def loo_r2(X, y):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        lr_cv = LinearRegression().fit(X[tr], y[tr])
        preds[te] = lr_cv.predict(X[te])
    return 1 - ((y - preds)**2).sum() / ((y - y.mean())**2).sum()

all_models = {}

for model_name, spec in model_specs.items():
    log("\n" + "═" * 80)
    log(f"MODEL: {model_name.upper()}")
    log("═" * 80)

    # Base fit on filtered subset for honest R²
    sub = spec["filter"](df).copy()
    if model_name == "peak":
        sub["log_peak"] = np.log(sub.peak_oil_msm3)
    elif model_name == "ramp":
        sub["log_ramp_p1"] = np.log(sub.ramp_length_months + 1)
    elif model_name == "plateau":
        sub["log_plat_p1"] = np.log(sub.plateau_length_months + 1)
    elif model_name == "decline":
        sub["log_D"] = np.log(sub.D_decline_fit)

    X_sub = sub[spec["features"]].fillna(0).values
    y_sub = sub[spec["target_col"]].values

    base_lr = LinearRegression().fit(X_sub, y_sub)
    base_in_R2 = base_lr.score(X_sub, y_sub)
    cv_R2 = loo_r2(X_sub, y_sub)

    # Duan smearing factor — korrigerer log-retransformasjons-bias (Jensen)
    # Brukes på peak (ramp/plat/decline har egne mekanismer i Script 26)
    base_resids = y_sub - base_lr.predict(X_sub)
    smear_factor = float(np.mean(np.exp(base_resids)))

    log(f"  N = {len(sub)}")
    log(f"  Features ({len(spec['features'])}): {spec['features']}")
    log(f"  In-sample R²: {base_in_R2:.3f}")
    log(f"  LOO CV R²:    {cv_R2:.3f}")
    log(f"  Intercept:    {base_lr.intercept_:+.4f}")
    for f, c in zip(spec['features'], base_lr.coef_):
        log(f"    {f:30s}  {c:+.4f}")

    # Joint bootstrap fits — use df_full so all models use same field-resamples
    X_full = df_full[spec["features"]].fillna(0).values
    y_full = df_full[spec["target_col"]].values

    bootstrap_fits = []
    for b_idx in bootstrap_indices:
        try:
            X_b = X_full[b_idx]
            y_b = y_full[b_idx]
            lr_b = LinearRegression().fit(X_b, y_b)
            bootstrap_fits.append({
                "intercept": float(lr_b.intercept_),
                "coefs": lr_b.coef_.tolist(),
            })
        except Exception:
            pass

    log(f"  Bootstrap fits: {len(bootstrap_fits)}/{B}")

    # Bootstrap CI for inspection
    boot_int = np.array([f["intercept"] for f in bootstrap_fits])
    boot_coefs = np.array([f["coefs"] for f in bootstrap_fits])
    log(f"  95% CI on intercept: [{np.percentile(boot_int, 2.5):+.3f}, {np.percentile(boot_int, 97.5):+.3f}]")
    for i, f in enumerate(spec["features"]):
        ci_l = np.percentile(boot_coefs[:, i], 2.5)
        ci_h = np.percentile(boot_coefs[:, i], 97.5)
        sig = "✓" if (ci_l > 0) == (ci_h > 0) else "✗"
        log(f"  95% CI {f:30s}  [{ci_l:+.3f}, {ci_h:+.3f}]  {sig}")

    log(f"  Smearing factor: {smear_factor:.4f}")

    all_models[model_name] = {
        "features": spec["features"],
        "base_intercept": float(base_lr.intercept_),
        "base_coefs": base_lr.coef_.tolist(),
        "bootstrap_fits": bootstrap_fits,
        "in_sample_R2": float(base_in_R2),
        "cv_R2": float(cv_R2),
        "cap": spec.get("cap"),
        "smear_factor": smear_factor,
        "n_base": int(len(sub)),
        "n_joint": int(len(df_full)),
    }

# Save
all_models["facility_categories"] = facility_cols
all_models["operator_categories"] = operator_cols
all_models["major_operators"] = major_ops
all_models["joint_n_fields"] = len(df_full)
all_models["B"] = B
all_models["build_features_meta"] = {
    "median_water_depth": float(df.water_depth.median()),
    "median_api": float(df.api_use.median()),
    "facility_baseline": list(set(df.facility_type) - set(c.replace("fac_", "") for c in facility_cols)),
    "operator_baseline": list(set(df.op_grouped) - set(c.replace("op_", "") for c in operator_cols)),
}

with open(DATA / "lifecycle_v2_models.pkl", "wb") as f:
    pickle.dump(all_models, f)

log(f"\n══════════════════════════════════════════════════════════════════════════")
log(f"SAMMENLIGNING: V1 vs V2 CV R²")
log(f"══════════════════════════════════════════════════════════════════════════")
log(f"\n  {'Model':12s} {'V1 (full)':>12s} {'V2 (forenklet)':>16s} {'Δ':>8s}")
log("  " + "─" * 55)
v1_vals = {"peak": 0.843, "ramp": 0.104, "plateau": 0.187, "decline": 0.285}
for name, m in all_models.items():
    if name in v1_vals:
        delta = m["cv_R2"] - v1_vals[name]
        marker = " ✓" if delta > 0 else " ✗" if delta < -0.01 else ""
        log(f"  {name:12s} {v1_vals[name]:12.3f} {m['cv_R2']:16.3f} {delta:+8.3f}{marker}")

log(f"\nSaved: lifecycle_v2_models.pkl")
log(f"  → {len(all_models)-4} modeller, hver med {B} bootstrap fits")

with open(RESULTS / "lifecycle_v2_submodels.txt", "w") as f:
    f.write("\n".join(lines))
