"""
Script 07: Reservoir Physics Models for Production Decline.

Three physics-based approaches:

1. DARCY + BEGGS-ROBINSON VISCOSITY MODEL
   Darcy's law: q ∝ k·A·ΔP / (μ·L)
   → Decline rate should scale with viscosity (μ)
   → μ estimated from API gravity via Beggs-Robinson (1975):
       μ_od = 10^(x·T^(-1.163)) - 1  where x = 10^(3.0324 - 0.02023·API)
   NCS reservoir temp estimated at ~80-100°C (typical North Sea)

2. BUCKLEY-LEVERETT WATER CUT MODEL
   Fractional flow: f_w = 1 / (1 + (k_ro/k_rw)·(μ_w/μ_o))
   → Water cut progression drives oil decline
   → Rate of water cut increase predicts remaining oil production
   → μ_w/μ_o ratio (mobility ratio) is the key physics parameter

3. PRODUCTIVITY INDEX DECLINE MODEL
   PI = q / ΔP → as reservoir depletes, PI declines
   → Rate of PI decline relates to reservoir/fluid properties
   → Combines Darcy flow with pressure depletion

Outputs:
  - fig_physics_viscosity.png    — viscosity vs. decline relationship
  - fig_physics_watercut.png     — water cut dynamics vs. oil decline
  - fig_physics_combined.png     — combined physics model vs. Arps
  - physics_model_results.csv    — per-field physics parameters
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

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

# ═══════════════════════════════════════════════════════════════════════════
# 1. BEGGS-ROBINSON VISCOSITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Dead oil viscosity (Beggs & Robinson, 1975):
#   μ_od = 10^(x · T^(-1.163)) - 1
#   where x = 10^(3.0324 - 0.02023 · API)
#   T in °F, μ in centipoise
#
# NCS reservoir temperatures: typically 80-110°C (176-230°F)
# Deeper fields (HP/HT): up to 150°C
# We use 90°C (194°F) as a representative NCS average

T_RESERVOIR_F = 194  # 90°C in Fahrenheit

def beggs_robinson_viscosity(api_gravity, T_F=T_RESERVOIR_F):
    """Estimate dead oil viscosity (cp) from API gravity and temperature."""
    x = 10 ** (3.0324 - 0.02023 * api_gravity)
    mu_od = 10 ** (x * T_F ** (-1.163)) - 1
    return mu_od

def api_to_density(api):
    """Convert API gravity to specific gravity."""
    return 141.5 / (api + 131.5)

# Water viscosity at reservoir conditions (~0.3-0.5 cp at 90°C)
MU_WATER_CP = 0.35

# Compute for each field
fields = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
fields["viscosity_cp"] = beggs_robinson_viscosity(fields.api_gravity)
fields["ln_viscosity"] = np.log(fields.viscosity_cp)
fields["density_sg"] = api_to_density(fields.api_gravity)
fields["mobility_ratio"] = fields.viscosity_cp / MU_WATER_CP  # M = μ_o / μ_w
fields["ln_mobility"] = np.log(fields.mobility_ratio)

log("═══════════════════════════════════════════════════════════════")
log("1. BEGGS-ROBINSON VISCOSITY MODEL (Darcy's Law)")
log("═══════════════════════════════════════════════════════════════")
log(f"   Reservoir temperature: {(T_RESERVOIR_F-32)*5/9:.0f}°C ({T_RESERVOIR_F}°F)")
log(f"   Water viscosity: {MU_WATER_CP} cp")
log(f"\n   Estimated oil viscosity across NCS fields:")
log(f"   API 25° → μ = {beggs_robinson_viscosity(25):.2f} cp")
log(f"   API 35° → μ = {beggs_robinson_viscosity(35):.2f} cp")
log(f"   API 45° → μ = {beggs_robinson_viscosity(45):.2f} cp")
log(f"   API 53° → μ = {beggs_robinson_viscosity(53):.2f} cp")
log(f"\n   Range: {fields.viscosity_cp.min():.2f} – {fields.viscosity_cp.max():.2f} cp")
log(f"   Mobility ratio M (μ_o/μ_w): {fields.mobility_ratio.min():.1f} – {fields.mobility_ratio.max():.1f}")

# Darcy prediction: D ∝ 1/μ → ln(D) ∝ -ln(μ)
# Or equivalently: D ∝ μ → positive correlation with viscosity = faster decline for heavy oil
r_visc, p_visc = stats.pearsonr(fields.ln_viscosity, fields.D_annual)
r_mob, p_mob = stats.pearsonr(fields.ln_mobility, fields.D_annual)
log(f"\n   Correlations with decline rate D:")
sig_v = "***" if p_visc < 0.01 else "**" if p_visc < 0.05 else "*" if p_visc < 0.1 else ""
sig_m = "***" if p_mob < 0.01 else "**" if p_mob < 0.05 else "*" if p_mob < 0.1 else ""
log(f"   ln(viscosity) vs D:     r={r_visc:+.3f} (p={p_visc:.4f}) {sig_v}")
log(f"   ln(mobility ratio) vs D: r={r_mob:+.3f} (p={p_mob:.4f}) {sig_m}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. BUCKLEY-LEVERETT WATER CUT DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
#
# Water cut evolution carries information about reservoir sweep efficiency.
# Key physics: dfw/dSw = (1/qt) · ∂(fw·qt)/∂Sw
#
# We model water cut as logistic growth:
#   f_w(t) = 1 / (1 + exp(-k·(t - t50)))
# where k = rate of water cut increase, t50 = time to 50% water cut
#
# The logistic rate k should relate to mobility ratio:
#   High M (heavy oil) → faster water breakthrough → higher k
#   Low M (light oil) → more piston-like displacement → lower k

log(f"\n{'═'*65}")
log("2. BUCKLEY-LEVERETT WATER CUT MODEL")
log(f"{'═'*65}")

def logistic_wc(t, k, t50):
    """Logistic water cut model: f_w(t) = 1 / (1 + exp(-k·(t - t50)))"""
    return 1 / (1 + np.exp(-k * (t - t50)))

wc_results = []
for field, grp in panel.groupby("field"):
    grp = grp.sort_values("months_since_start").dropna(subset=["water_cut"])
    if len(grp) < 24:
        continue

    t = grp.months_since_start.values.astype(float)
    wc = grp.water_cut.values

    # Skip fields with almost no water cut variation
    if wc.max() - wc.min() < 0.1:
        continue

    try:
        popt, _ = curve_fit(logistic_wc, t, wc,
                            p0=[0.02, t.mean()],
                            bounds=([0.001, 0], [0.2, t.max() * 2]),
                            maxfev=5000)
        k_wc, t50_wc = popt

        wc_pred = logistic_wc(t, k_wc, t50_wc)
        ss_res = np.sum((wc - wc_pred) ** 2)
        ss_tot = np.sum((wc - wc.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        wc_results.append({
            "field": field,
            "wc_k": k_wc,
            "wc_t50": t50_wc,
            "wc_r2": r2,
            "wc_final": wc[-1],
            "wc_rate_pp_yr": k_wc * 12 * 100,  # approximate annual wc increase in pp
        })
    except (RuntimeError, ValueError):
        continue

wc_df = pd.DataFrame(wc_results)
fields = fields.merge(wc_df, on="field", how="left")

log(f"   Logistic water cut model fitted for {len(wc_df)} fields")
log(f"   Fit R²: median={wc_df.wc_r2.median():.3f}")
log(f"   Water cut rate k: median={wc_df.wc_k.median():.4f}/mnd")
log(f"   Time to 50% water cut: median={wc_df.wc_t50.median():.0f} months")

# Test: does water cut rate correlate with decline?
wc_valid = fields.dropna(subset=["wc_k", "D_annual"])
r_wck, p_wck = stats.pearsonr(wc_valid.wc_k, wc_valid.D_annual)
sig_wc = "***" if p_wck < 0.01 else "**" if p_wck < 0.05 else "*" if p_wck < 0.1 else ""
log(f"\n   Water cut rate (k) vs D:  r={r_wck:+.3f} (p={p_wck:.4f}) {sig_wc}")

# Test: does mobility ratio predict water cut rate? (Buckley-Leverett link)
mob_wc = fields.dropna(subset=["ln_mobility", "wc_k"])
r_mwc, p_mwc = stats.pearsonr(mob_wc.ln_mobility, mob_wc.wc_k)
sig_mwc = "***" if p_mwc < 0.01 else "**" if p_mwc < 0.05 else "*" if p_mwc < 0.1 else ""
log(f"   Mobility ratio vs wc rate: r={r_mwc:+.3f} (p={p_mwc:.4f}) {sig_mwc}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. COMBINED PHYSICS REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
#
# Can physics variables (viscosity, mobility, water cut dynamics) predict
# decline better than raw quality features (API, sulfur)?

log(f"\n{'═'*65}")
log("3. PHYSICS MODEL vs RAW QUALITY: regression comparison")
log(f"{'═'*65}")

reg_data = fields.dropna(subset=["D_annual", "ln_viscosity", "ln_mobility",
                                  "wc_k", "oil_mean"]).copy()
y = reg_data.D_annual

# Model A: Raw quality features (baseline)
X_raw = sm.add_constant(reg_data[["api_gravity", "sulfur_pct", "oil_mean"]])
m_raw = sm.OLS(y, X_raw).fit(cov_type="HC1")

# Model B: Physics-derived variables
X_phys = sm.add_constant(reg_data[["ln_viscosity", "wc_k", "oil_mean"]])
m_phys = sm.OLS(y, X_phys).fit(cov_type="HC1")

# Model C: Physics + mobility ratio
X_full = sm.add_constant(reg_data[["ln_viscosity", "ln_mobility", "wc_k", "oil_mean"]])
m_full = sm.OLS(y, X_full).fit(cov_type="HC1")

# Model D: Kitchen sink — physics + raw
X_all = sm.add_constant(reg_data[["ln_viscosity", "wc_k", "sulfur_pct", "oil_mean"]])
m_all = sm.OLS(y, X_all).fit(cov_type="HC1")

models = {"Raw quality": m_raw, "Physics": m_phys, "Physics+mobility": m_full,
          "Physics+sulfur": m_all}

for name, m in models.items():
    log(f"\n  {name} (N={m.nobs:.0f}, R²={m.rsquared:.3f}, Adj-R²={m.rsquared_adj:.3f}, AIC={m.aic:.1f})")
    for var in m.params.index:
        if var == "const":
            continue
        sig = "***" if m.pvalues[var] < 0.01 else "**" if m.pvalues[var] < 0.05 else "*" if m.pvalues[var] < 0.1 else ""
        log(f"    {var:<20s} β={m.params[var]:>10.5f} (p={m.pvalues[var]:.3f}) {sig}")

# Standardized β for best model
log(f"\n  Standardized β (Physics model):")
reg_std = reg_data.copy()
for col in ["ln_viscosity", "wc_k", "oil_mean"]:
    reg_std[col] = (reg_std[col] - reg_std[col].mean()) / reg_std[col].std()
reg_std["D_z"] = (reg_std.D_annual - reg_std.D_annual.mean()) / reg_std.D_annual.std()
X_std = sm.add_constant(reg_std[["ln_viscosity", "wc_k", "oil_mean"]])
m_std = sm.OLS(reg_std.D_z, X_std).fit(cov_type="HC1")
for var in ["ln_viscosity", "wc_k", "oil_mean"]:
    sig = "***" if m_std.pvalues[var] < 0.01 else "**" if m_std.pvalues[var] < 0.05 else "*" if m_std.pvalues[var] < 0.1 else ""
    log(f"    {var:<20s} β={m_std.params[var]:>+7.3f} (p={m_std.pvalues[var]:.3f}) {sig}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Viscosity → Decline (Darcy's Law)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1a: API → Viscosity (Beggs-Robinson curve)
ax = axes[0]
api_range = np.linspace(15, 55, 200)
mu_range = beggs_robinson_viscosity(api_range)
ax.plot(api_range, mu_range, color="#1565C0", linewidth=2.5)
ax.scatter(fields.api_gravity, fields.viscosity_cp, c="#EF6C00", s=40, alpha=0.7,
           edgecolors="white", zorder=3)
ax.set_xlabel("API Gravity (°)", fontsize=10)
ax.set_ylabel("Dead Oil Viscosity (cp)", fontsize=10)
ax.set_title("Beggs-Robinson Viscosity\n(at 90°C reservoir temp)", fontsize=11)
ax.set_yscale("log")
ax.grid(True, alpha=0.2)
ax.text(0.97, 0.97, "μ ↓ as API ↑\n(lighter oil flows easier)",
        transform=ax.transAxes, va="top", ha="right", fontsize=8, style="italic", alpha=0.6)

# 1b: ln(viscosity) vs D
ax = axes[1]
ax.scatter(fields.viscosity_cp, fields.D_annual, c="#1565C0", s=50, alpha=0.6, edgecolors="white")
slope, intercept, r, p, _ = stats.linregress(fields.ln_viscosity, fields.D_annual)
x_fit = np.exp(np.linspace(fields.ln_viscosity.min(), fields.ln_viscosity.max(), 100))
y_fit = intercept + slope * np.log(x_fit)
ax.plot(x_fit, y_fit, color="#455A64", linewidth=2)
sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r(ln(μ), D) = {r:+.3f} ({sig})", transform=ax.transAxes, va="top",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Oil Viscosity (cp)", fontsize=10)
ax.set_ylabel("Annual Decline Rate D (yr⁻¹)", fontsize=10)
ax.set_title("Darcy's Law Test:\nViscosity → Decline Rate", fontsize=11)
ax.set_xscale("log")
ax.grid(True, alpha=0.2)

for _, row in fields.nlargest(2, "D_annual").iterrows():
    ax.annotate(row.field, (row.viscosity_cp, row.D_annual),
                fontsize=7, alpha=0.6, xytext=(5, 5), textcoords="offset points")

# 1c: Mobility ratio vs D
ax = axes[2]
ax.scatter(fields.mobility_ratio, fields.D_annual, c="#2E7D32", s=50, alpha=0.6, edgecolors="white")
slope, intercept, r, p, _ = stats.linregress(fields.ln_mobility, fields.D_annual)
x_fit = np.exp(np.linspace(fields.ln_mobility.min(), fields.ln_mobility.max(), 100))
y_fit = intercept + slope * np.log(x_fit)
ax.plot(x_fit, y_fit, color="#455A64", linewidth=2)
sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r(ln(M), D) = {r:+.3f} ({sig})", transform=ax.transAxes, va="top",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Mobility Ratio M = μ_oil / μ_water", fontsize=10)
ax.set_ylabel("Annual Decline Rate D (yr⁻¹)", fontsize=10)
ax.set_title("Buckley-Leverett Test:\nMobility Ratio → Decline", fontsize=11)
ax.set_xscale("log")
ax.grid(True, alpha=0.2)
ax.axvline(1, color="red", linewidth=1, linestyle="--", alpha=0.5)
ax.text(1.05, ax.get_ylim()[1] * 0.95, "M=1\n(favorable)", fontsize=7, alpha=0.5, va="top")

fig.suptitle("Reservoir Physics: Darcy's Law & Viscosity → Production Decline",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_physics_viscosity.png", **SAVEKW)
plt.close()
log("\nSaved fig_physics_viscosity.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Water Cut Dynamics
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: Water cut curves colored by API class
ax = axes[0, 0]
panel_wc = panel.dropna(subset=["water_cut"]).copy()
panel_wc["api_class"] = pd.cut(panel_wc.api_gravity, bins=[0, 30, 40, 100],
                                labels=["Heavy (<30°)", "Medium (30-40°)", "Light (>40°)"])
class_colors = {"Heavy (<30°)": "#C62828", "Medium (30-40°)": "#F57C00", "Light (>40°)": "#2E7D32"}

for field, grp in panel_wc.groupby("field"):
    cls = grp.api_class.iloc[0]
    if pd.isna(cls):
        continue
    ax.plot(grp.months_since_start, grp.water_cut * 100,
            color=class_colors[cls], alpha=0.12, linewidth=0.5)

for cls, color in class_colors.items():
    d = panel_wc[panel_wc.api_class == cls]
    med = d.groupby("months_since_start")["water_cut"].median() * 100
    smoothed = med.rolling(12, min_periods=6, center=True).median()
    n = d.field.nunique()
    ax.plot(smoothed.index, smoothed.values, color=color, linewidth=2.5,
            label=f"{cls} (n={n})")

ax.set_xlabel("Months since first production")
ax.set_ylabel("Water Cut (%)")
ax.set_title("Water Cut Evolution by API Class")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 500)

# 2b: Water cut rate (k) vs decline rate (D)
ax = axes[0, 1]
v = fields.dropna(subset=["wc_k", "D_annual"])
ax.scatter(v.wc_k * 12, v.D_annual, c="#1565C0", s=50, alpha=0.6, edgecolors="white")
slope, intercept, r, p, _ = stats.linregress(v.wc_k, v.D_annual)
x_fit = np.linspace(v.wc_k.min(), v.wc_k.max(), 100)
ax.plot(x_fit * 12, intercept + slope * x_fit, color="#455A64", linewidth=2)
sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Water Cut Rate k (yr⁻¹)")
ax.set_ylabel("Annual Decline Rate D (yr⁻¹)")
ax.set_title("Water Cut Dynamics → Oil Decline")
ax.grid(True, alpha=0.2)

# 2c: Mobility ratio vs water cut rate (physics chain: μ → M → k_wc → D)
ax = axes[1, 0]
v2 = fields.dropna(subset=["mobility_ratio", "wc_k"])
ax.scatter(v2.mobility_ratio, v2.wc_k * 12, c="#2E7D32", s=50, alpha=0.6, edgecolors="white")
slope, intercept, r, p, _ = stats.linregress(np.log(v2.mobility_ratio), v2.wc_k)
x_fit = np.exp(np.linspace(np.log(v2.mobility_ratio.min()), np.log(v2.mobility_ratio.max()), 100))
ax.plot(x_fit, (intercept + slope * np.log(x_fit)) * 12, color="#455A64", linewidth=2)
sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r(ln(M), k) = {r:+.3f} ({sig})", transform=ax.transAxes, va="top",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_xlabel("Mobility Ratio M = μ_oil / μ_water")
ax.set_ylabel("Water Cut Rate k (yr⁻¹)")
ax.set_title("Mobility Ratio → Water Breakthrough Speed")
ax.set_xscale("log")
ax.grid(True, alpha=0.2)

# 2d: The full physics chain visualization
ax = axes[1, 1]
chain_data = fields.dropna(subset=["api_gravity", "viscosity_cp", "mobility_ratio",
                                    "wc_k", "D_annual"]).copy()
# Scatter: x = predicted D from physics chain, y = actual D
X_chain = sm.add_constant(chain_data[["ln_viscosity", "wc_k"]])
m_chain = sm.OLS(chain_data.D_annual, X_chain).fit()
D_predicted = m_chain.predict(X_chain)

ax.scatter(D_predicted, chain_data.D_annual, c="#1565C0", s=50, alpha=0.6, edgecolors="white")
lims = [0, max(D_predicted.max(), chain_data.D_annual.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
slope, intercept, r, p, _ = stats.linregress(D_predicted, chain_data.D_annual)
ax.plot(lims, [intercept + slope * x for x in lims], color="#C62828", linewidth=2,
        label=f"Fit (R²={r**2:.3f})")
ax.set_xlabel("Physics-Predicted D (yr⁻¹)")
ax.set_ylabel("Actual D (yr⁻¹)")
ax.set_title(f"Physics Model: Predicted vs. Actual Decline\n(ln(μ) + water cut rate → D)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlim(lims)
ax.set_ylim(lims)

for _, row in chain_data.nlargest(3, "D_annual").iterrows():
    ax.annotate(row.field, (D_predicted[chain_data.field == row.field].values[0], row.D_annual),
                fontsize=7, alpha=0.6, xytext=(5, 5), textcoords="offset points")

fig.suptitle("Reservoir Physics: Water Cut Dynamics & The Viscosity–Decline Chain",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_physics_watercut.png", **SAVEKW)
plt.close()
log("Saved fig_physics_watercut.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Model comparison — Physics vs Raw Quality vs Arps
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3a: R² comparison bar chart
ax = axes[0]
model_names = list(models.keys())
r2_vals = [m.rsquared for m in models.values()]
adj_r2_vals = [m.rsquared_adj for m in models.values()]

x = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x - w/2, r2_vals, w, color="#1565C0", alpha=0.8, label="R²", edgecolor="white")
bars2 = ax.bar(x + w/2, adj_r2_vals, w, color="#90CAF9", alpha=0.8, label="Adj R²", edgecolor="white")

for i, (r2, ar2) in enumerate(zip(r2_vals, adj_r2_vals)):
    ax.text(i - w/2, r2 + 0.005, f"{r2:.3f}", ha="center", fontsize=7)
    ax.text(i + w/2, ar2 + 0.005, f"{ar2:.3f}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=8, rotation=15)
ax.set_ylabel("R²")
ax.set_title("Model Comparison: Explaining D")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis="y")

# 3b: Standardized coefficients — physics model
ax = axes[1]
std_vars = ["ln_viscosity", "wc_k", "oil_mean"]
std_labels = ["ln(Viscosity)\n[Darcy]", "Water Cut Rate\n[Buckley-Leverett]", "Field Size\n[Scale]"]
coefs = [m_std.params[v] for v in std_vars]
ci_lo = [m_std.conf_int().loc[v, 0] for v in std_vars]
ci_hi = [m_std.conf_int().loc[v, 1] for v in std_vars]
pvals = [m_std.pvalues[v] for v in std_vars]
colors = ["#C62828" if p < 0.05 else "#FF8F00" if p < 0.1 else "#9E9E9E" for p in pvals]

y_pos = range(len(std_vars))
ax.barh(y_pos, coefs, color=colors, alpha=0.7, edgecolor="white", height=0.6, zorder=2)
for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
    ax.plot([lo, hi], [i, i], color="#37474F", linewidth=2, zorder=3)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(std_labels, fontsize=9)
ax.set_xlabel("Standardized β")
ax.set_title("Physics Model: Relative Importance")
ax.grid(True, alpha=0.2, axis="x")

for i, (c, p) in enumerate(zip(coefs, pvals)):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.text(c + (0.02 if c >= 0 else -0.02), i, f"β={c:+.3f}{sig}",
            va="center", ha="left" if c >= 0 else "right", fontsize=8)

# 3c: Physics chain diagram (conceptual)
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("The Physics Chain", fontsize=12, fontweight="bold")

boxes = [
    (1, 8, "API Gravity\n(observable)"),
    (1, 6, "Oil Viscosity μ\n(Beggs-Robinson)"),
    (1, 4, "Mobility Ratio M\n(μ_oil / μ_water)"),
    (5.5, 6, "Water Cut Rate\n(Buckley-Leverett)"),
    (5.5, 3, "Production\nDecline Rate D"),
    (1, 2, "Field Size\n(infrastructure)"),
]

for x, y, text in boxes:
    ax.annotate(text, (x, y), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5))

arrows = [
    ((1, 7.5), (1, 6.5), "Beggs-\nRobinson"),
    ((1, 5.5), (1, 4.5), "÷ μ_water"),
    ((2.5, 4), (4, 6), "drives"),
    ((5.5, 5.5), (5.5, 3.8), "oil replaced\nby water"),
    ((2.5, 6), (4, 6), "Darcy"),
    ((2.5, 2), (4, 3), "controls\ncapacity"),
]

for (x1, y1), (x2, y2), label in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#455A64", linewidth=1.5))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx - 0.3, my + 0.15, label, fontsize=6, alpha=0.6, style="italic")

fig.suptitle("Physics-Based Decline Model vs. Raw Quality Features",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_physics_combined.png", **SAVEKW)
plt.close()
log("Saved fig_physics_combined.png")

# ── Save results ────────────────────────────────────────────────────────────

output_cols = ["field", "grade", "api_gravity", "sulfur_pct", "viscosity_cp",
               "mobility_ratio", "density_sg", "D_annual", "oil_mean", "main_area",
               "wc_k", "wc_t50", "wc_r2", "wc_final"]
fields[output_cols].to_csv(RESULTS / "physics_model_results.csv", index=False)

with open(RESULTS / "physics_model_summary.txt", "w") as f:
    f.write("\n".join(lines))

log(f"\nSaved physics_model_results.csv and physics_model_summary.txt")
