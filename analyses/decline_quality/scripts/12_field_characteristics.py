"""
Script 12: Field-specific characteristics — what drives the residuals?

Joins model residuals with Sodir data on:
  - Facility type (FPSO, fixed, subsea)
  - Water depth
  - Well count (production, injection)
  - Reserves & recovery factor
  - CAPEX intensity
  - Operator

Outputs:
  - fig_field_characteristics.png  — main results
  - field_characteristics.txt      — tables
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
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir"
SAVEKW = dict(bbox_inches="tight")

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

def fit_D_window(grp, t_min, t_max, min_obs=10):
    d = grp[(grp.months_since_peak >= t_min) & (grp.months_since_peak < t_max)]
    d = d[d.oil_pct_peak > 0].dropna(subset=["months_since_peak", "oil_pct_peak"])
    if len(d) < min_obs:
        return np.nan
    t = d.months_since_peak.values
    ln_y = np.log(d.oil_pct_peak.values)
    return -stats.linregress(t, ln_y)[0] * 12

def loo_cv_r2(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    if len(y) < 10:
        return np.nan, len(y)
    loo = LeaveOneOut()
    lr = LinearRegression()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        lr.fit(X[tr], y[tr])
        preds[te] = lr.predict(X[te])
    return 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2), len(y)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD BASE MODEL DATA
# ═══════════════════════════════════════════════════════════════════════════

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")

post = panel[panel.is_post_peak].copy()
D_12 = post.groupby("field").apply(lambda g: fit_D_window(g, 0, 12)).rename("D_12")

df = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
df = df.merge(D_12, on="field")
df["ln_viscosity"] = np.log(beggs_robinson(df.api_gravity))
df = df.dropna(subset=["D_12", "D_annual", "ln_viscosity"])

# Compute residuals from best model (D_12 + visc)
X = sm.add_constant(df[["D_12", "ln_viscosity"]])
model = sm.OLS(df.D_annual, X).fit()
df["resid"] = model.resid
df["predicted"] = model.fittedvalues

log("BASE MODEL: D_12 + ln(viscosity)")
log(f"  R²={model.rsquared:.3f}, n={len(df)}")
log(f"  Residual range: {df.resid.min():.3f} to {df.resid.max():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD SODIR FIELD CHARACTERISTICS
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("LOADING SODIR FIELD DATA")
log(f"{'═'*65}")

# --- Facilities ---
fac = pd.read_csv(RAW / "sodir_facilities.csv", encoding="utf-8-sig")
fac = fac[fac.fclBelongsToKind == "FIELD"]

# Classify facility type per field
def classify_facility(grp):
    kinds = set(grp.fclKind.dropna().str.upper())
    if any("FPSO" in k for k in kinds):
        return "FPSO"
    elif any("SEMI" in k for k in kinds):
        return "Semi-sub"
    elif any("JACKET" in k or "CONDEEP" in k or "GRAVITY" in k for k in kinds):
        return "Fixed"
    elif any("TEMPLATE" in k or "SUBSEA" in k for k in kinds):
        return "Subsea tieback"
    else:
        return "Other"

fac_type = fac.groupby("fclBelongsToName").apply(classify_facility).rename("facility_type")
water_depth = fac.groupby("fclBelongsToName")["fclWaterDepth"].max().rename("water_depth")

log(f"  Facilities: {len(fac)} records, {fac.fclBelongsToName.nunique()} fields")
log(f"  Facility types: {fac_type.value_counts().to_dict()}")

# --- Wells ---
wells = pd.read_csv(RAW / "sodir_wells_dev.csv", encoding="utf-8-sig")
well_counts = wells.groupby(["wlbField", "wlbPurpose"]).size().unstack(fill_value=0)
well_counts.columns = [f"wells_{c.lower().replace(' ', '_')}" for c in well_counts.columns]
if "wells_production" not in well_counts.columns:
    well_counts["wells_production"] = 0
if "wells_injection" not in well_counts.columns:
    well_counts["wells_injection"] = 0
well_counts["wells_total"] = well_counts.sum(axis=1)
well_counts["injection_ratio"] = well_counts.wells_injection / well_counts.wells_total.clip(lower=1)

log(f"  Wells: {len(wells)} wellbores, {wells.wlbField.nunique()} fields")

# --- Reserves & in-place ---
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv", encoding="utf-8-sig")
reserves_latest = reserves[reserves.fldVersion == reserves.fldVersion.max()]
reserves_latest = reserves_latest.set_index("fldName")

inplace = pd.read_csv(RAW / "sodir_field_inplace.csv", encoding="utf-8-sig")
inplace = inplace.set_index("fldName")

# Recovery factor
rf = reserves_latest[["fldRecoverableOil"]].merge(
    inplace[["fldInplaceOil"]], left_index=True, right_index=True, how="inner")
rf["recovery_factor"] = rf.fldRecoverableOil / rf.fldInplaceOil.clip(lower=0.001)
rf = rf[(rf.recovery_factor > 0) & (rf.recovery_factor < 1)]

log(f"  Reserves: {len(reserves_latest)} fields (latest version)")
log(f"  Recovery factor: {len(rf)} fields, median={rf.recovery_factor.median():.2f}")

# --- CAPEX ---
capex = pd.read_csv(RAW / "sodir_field_capex.csv", encoding="utf-8-sig")
capex_total = capex.groupby("prfInformationCarrier")["prfInvestmentsMillNOK"].sum().rename("capex_total_mnok")

log(f"  CAPEX: {len(capex)} records, {capex.prfInformationCarrier.nunique()} fields")

# --- Reserves revision (how much did reserves change?) ---
reserves_pivot = reserves.pivot_table(
    index="fldName", columns="fldVersion", values="fldRecoverableOil")
if reserves_pivot.shape[1] >= 2:
    first_ver = reserves_pivot.columns.min()
    last_ver = reserves_pivot.columns.max()
    reserves_pivot["reserves_revision"] = (
        (reserves_pivot[last_ver] - reserves_pivot[first_ver])
        / reserves_pivot[first_ver].clip(lower=0.001))

# ═══════════════════════════════════════════════════════════════════════════
# MERGE WITH MODEL RESIDUALS
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("MERGING FIELD CHARACTERISTICS")
log(f"{'═'*65}")

# Field name matching (uppercase for join)
df["field_upper"] = df.field.str.upper()

fac_type_df = fac_type.reset_index()
fac_type_df.columns = ["field_upper", "facility_type"]
fac_type_df["field_upper"] = fac_type_df.field_upper.str.upper()

wd_df = water_depth.reset_index()
wd_df.columns = ["field_upper", "water_depth"]
wd_df["field_upper"] = wd_df.field_upper.str.upper()

wc_df = well_counts.reset_index()
wc_df.columns = ["field_upper"] + list(wc_df.columns[1:])
wc_df["field_upper"] = wc_df.field_upper.str.upper()

rf_df = rf[["recovery_factor"]].reset_index()
rf_df.columns = ["field_upper", "recovery_factor"]
rf_df["field_upper"] = rf_df.field_upper.str.upper()

capex_df = capex_total.reset_index()
capex_df.columns = ["field_upper", "capex_total_mnok"]
capex_df["field_upper"] = capex_df.field_upper.str.upper()

rec_oil = reserves_latest[["fldRecoverableOil"]].reset_index()
rec_oil.columns = ["field_upper", "recoverable_oil"]
rec_oil["field_upper"] = rec_oil.field_upper.str.upper()

rev_df = reserves_pivot[["reserves_revision"]].reset_index() if "reserves_revision" in reserves_pivot.columns else pd.DataFrame()
if len(rev_df) > 0:
    rev_df.columns = ["field_upper", "reserves_revision"]
    rev_df["field_upper"] = rev_df.field_upper.str.upper()

# Merge all
for merge_df in [fac_type_df, wd_df, wc_df, rf_df, capex_df, rec_oil]:
    df = df.merge(merge_df, on="field_upper", how="left")
if len(rev_df) > 0:
    df = df.merge(rev_df, on="field_upper", how="left")

# Derived features
df["capex_per_oil"] = df.capex_total_mnok / df.recoverable_oil.clip(lower=0.001)
df["wells_per_oil"] = df.wells_total / df.recoverable_oil.clip(lower=0.001)

log(f"  Merged dataset: {len(df)} fields")
log(f"  Facility type: {df.facility_type.value_counts().to_dict()}")
log(f"  Wells matched: {df.wells_total.notna().sum()}")
log(f"  Recovery factor matched: {df.recovery_factor.notna().sum()}")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Which characteristics correlate with residuals?
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("CORRELATIONS: field characteristics vs model residual")
log(f"{'═'*65}")

continuous_vars = [
    ("water_depth", "Water depth (m)"),
    ("wells_total", "Total wells"),
    ("wells_production", "Production wells"),
    ("wells_injection", "Injection wells"),
    ("injection_ratio", "Injection ratio"),
    ("recovery_factor", "Recovery factor"),
    ("capex_total_mnok", "Total CAPEX (MNOK)"),
    ("capex_per_oil", "CAPEX per recoverable oil"),
    ("wells_per_oil", "Wells per recoverable oil"),
    ("recoverable_oil", "Recoverable oil (Mill Sm3)"),
]

if "reserves_revision" in df.columns:
    continuous_vars.append(("reserves_revision", "Reserves revision (%)"))

corr_results = {}
for col, label in continuous_vars:
    sub = df.dropna(subset=[col, "resid"])
    if len(sub) < 10:
        continue
    r, p = stats.pearsonr(sub[col], sub.resid)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    corr_results[col] = {"r": r, "p": p, "n": len(sub), "label": label}
    log(f"  {label:35s} r={r:+.3f} (p={p:.3f}) {sig:3s} n={len(sub)}")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Categorical — facility type, operator
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("CATEGORICAL: residual by facility type")
log(f"{'═'*65}")

for ftype, grp in df.groupby("facility_type"):
    if len(grp) < 3:
        continue
    log(f"  {ftype:20s} n={len(grp):2d}  mean resid={grp.resid.mean():+.4f}  "
        f"median D={grp.D_annual.median():.3f}")

# ANOVA for facility type
groups = [grp.resid.values for _, grp in df.dropna(subset=["facility_type"]).groupby("facility_type")
          if len(grp) >= 3]
if len(groups) >= 2:
    f_stat, p_val = stats.f_oneway(*groups)
    log(f"\n  ANOVA F={f_stat:.2f}, p={p_val:.3f}")

# Operator
log(f"\n{'═'*65}")
log("CATEGORICAL: residual by operator (top operators)")
log(f"{'═'*65}")

# Get operator from GeoJSON
import json
with open(Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir_geo" / "fields.geojson") as f:
    geo = json.load(f)
op_map = {}
for feat in geo["features"]:
    p = feat["properties"]
    op_map[p["fldName"].upper()] = p.get("cmpLongName", "Unknown")

df["operator"] = df.field_upper.map(op_map)

for op, grp in df.groupby("operator"):
    if len(grp) >= 3:
        log(f"  {op:40s} n={len(grp):2d}  mean resid={grp.resid.mean():+.4f}  "
            f"median D={grp.D_annual.median():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Augmented model — add best characteristics
# ═══════════════════════════════════════════════════════════════════════════

log(f"\n{'═'*65}")
log("AUGMENTED MODELS: D_12 + visc + field characteristics")
log(f"{'═'*65}")

base_cols = ["D_12", "ln_viscosity"]

# Test adding each characteristic
aug_results = {}
for col, label in continuous_vars:
    sub = df.dropna(subset=base_cols + [col, "D_annual"])
    if len(sub) < 20:
        continue
    cv_r2, n = loo_cv_r2(sub[base_cols + [col]].values, sub.D_annual.values)

    X = sm.add_constant(sub[base_cols + [col]])
    m = sm.OLS(sub.D_annual, X).fit(cov_type="HC1")

    aug_results[col] = {"cv_r2": cv_r2, "r2": m.rsquared, "n": n, "label": label}
    log(f"  + {label:30s} CV R²={cv_r2:+.3f}  R²={m.rsquared:.3f}  n={n}")

# Baseline for comparison
cv_base, n_base = loo_cv_r2(df[base_cols].values, df.D_annual.values)
log(f"\n  Baseline (D_12 + visc):      CV R²={cv_base:+.3f}  R²={model.rsquared:.3f}  n={n_base}")

# Best combination
best_aug = max(aug_results, key=lambda k: aug_results[k]["cv_r2"])
log(f"  Best addition: {aug_results[best_aug]['label']} (CV R²={aug_results[best_aug]['cv_r2']:.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1: Residuals by facility type
ax = axes[0, 0]
ftype_data = df.dropna(subset=["facility_type"])
types_ordered = ftype_data.groupby("facility_type")["resid"].median().sort_values().index
type_colors = {"Fixed": "#1565C0", "FPSO": "#E65100", "Semi-sub": "#2E7D32",
               "Subsea tieback": "#7B1FA2", "Other": "#9E9E9E"}

positions = []
labels = []
for i, ft in enumerate(types_ordered):
    grp = ftype_data[ftype_data.facility_type == ft]
    if len(grp) < 2:
        continue
    positions.append(i)
    labels.append(f"{ft}\n(n={len(grp)})")
    ax.scatter(np.full(len(grp), i) + np.random.uniform(-0.15, 0.15, len(grp)),
               grp.resid, c=type_colors.get(ft, "#9E9E9E"), s=40, alpha=0.6,
               edgecolors="white", linewidths=0.5)
    ax.plot([i - 0.2, i + 0.2], [grp.resid.median(), grp.resid.median()],
            color="black", linewidth=2)

ax.axhline(0, color="red", linewidth=0.8, linestyle=":")
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
style_ax(ax, "", "Model Residual (actual − predicted D)",
         "Decline by Facility Type")

# 2: Water depth vs residual
ax = axes[0, 1]
sub = df.dropna(subset=["water_depth", "resid"])
ax.scatter(sub.water_depth, sub.resid, c="#1565C0", s=40, alpha=0.6, edgecolors="white")
if len(sub) >= 5:
    r, p = stats.pearsonr(sub.water_depth, sub.resid)
    slope, intercept, _, _, _ = stats.linregress(sub.water_depth, sub.resid)
    x_fit = np.linspace(sub.water_depth.min(), sub.water_depth.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="#C62828", linewidth=2)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.axhline(0, color="red", linewidth=0.8, linestyle=":")
style_ax(ax, "Water Depth (m)", "Model Residual", "Water Depth Effect")

for _, row in sub.nlargest(3, "resid").iterrows():
    ax.annotate(row.field, (row.water_depth, row.resid), fontsize=6, alpha=0.5,
                xytext=(5, 3), textcoords="offset points")

# 3: Wells vs residual
ax = axes[0, 2]
sub = df.dropna(subset=["injection_ratio", "resid"])
ax.scatter(sub.injection_ratio, sub.resid, c="#2E7D32", s=40, alpha=0.6, edgecolors="white")
if len(sub) >= 5:
    r, p = stats.pearsonr(sub.injection_ratio, sub.resid)
    slope, intercept, _, _, _ = stats.linregress(sub.injection_ratio, sub.resid)
    x_fit = np.linspace(sub.injection_ratio.min(), sub.injection_ratio.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="#C62828", linewidth=2)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.axhline(0, color="red", linewidth=0.8, linestyle=":")
style_ax(ax, "Injection Well Ratio", "Model Residual",
         "Water/Gas Injection Effect")

# 4: Recovery factor vs residual
ax = axes[1, 0]
sub = df.dropna(subset=["recovery_factor", "resid"])
ax.scatter(sub.recovery_factor * 100, sub.resid, c="#E65100", s=40, alpha=0.6, edgecolors="white")
if len(sub) >= 5:
    r, p = stats.pearsonr(sub.recovery_factor, sub.resid)
    slope, intercept, _, _, _ = stats.linregress(sub.recovery_factor * 100, sub.resid)
    x_fit = np.linspace(sub.recovery_factor.min() * 100, sub.recovery_factor.max() * 100, 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="#C62828", linewidth=2)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.axhline(0, color="red", linewidth=0.8, linestyle=":")
style_ax(ax, "Recovery Factor (%)", "Model Residual",
         "Recovery Factor vs Residual")

# 5: Augmented model comparison
ax = axes[1, 1]
aug_sorted = sorted(aug_results.items(), key=lambda x: x[1]["cv_r2"], reverse=True)
labels_aug = [aug_results[k]["label"][:20] for k, _ in aug_sorted[:8]]
cv_vals = [r["cv_r2"] for _, r in aug_sorted[:8]]
colors_bar = ["#E65100" if v > cv_base else "#78909C" for v in cv_vals]

ax.barh(range(len(labels_aug)), cv_vals, color=colors_bar, alpha=0.8, edgecolor="white")
ax.axvline(cv_base, color="#1565C0", linewidth=2, linestyle="--", label=f"Baseline ({cv_base:.3f})")
ax.set_yticks(range(len(labels_aug)))
ax.set_yticklabels(labels_aug, fontsize=8)
for i, v in enumerate(cv_vals):
    ax.text(max(v, 0) + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
ax.invert_yaxis()
ax.legend(fontsize=8)
style_ax(ax, "LOO-CV R²", "", "Adding Field Characteristics\nto Momentum+Physics")

# 6: Per-field residual map with characteristics
ax = axes[1, 2]
sorted_df = df.sort_values("resid")
colors_resid = []
for _, row in sorted_df.iterrows():
    ft = row.get("facility_type", "Other")
    colors_resid.append(type_colors.get(ft, "#9E9E9E"))

ax.barh(range(len(sorted_df)), sorted_df.resid, color=colors_resid,
        alpha=0.7, edgecolor="white", height=0.7)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df.field, fontsize=5)
ax.axvline(0, color="black", linewidth=0.8)
style_ax(ax, "Residual (actual − predicted D)", "",
         "Per-Field Residuals by Facility Type")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l, alpha=0.7) for l, c in type_colors.items()
                   if l in df.facility_type.values]
ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

fig.suptitle("What Drives Field-Specific Decline? Residual Analysis with Sodir Data",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_field_characteristics.png", **SAVEKW)
plt.close()
log("\nSaved fig_field_characteristics.png")

with open(RESULTS / "field_characteristics.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved field_characteristics.txt ({len(lines)} lines)")
