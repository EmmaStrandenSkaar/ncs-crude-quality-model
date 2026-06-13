"""
Script 18: Final V5 Model Figure + Validation
═══════════════════════════════════════════════════════════════════════════

Produserer endelig figur for V5-modellen (hybrid + felt-spesifikk T)
for bruk i metodikkdokumentet.
"""

import json, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
GEO = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir_geo" / "fields.geojson"

with open(GEO) as f:
    gj = json.load(f)
op_map = {feat["properties"].get("fldName"): (feat["properties"].get("cmpLongName") or "")
          for feat in gj["features"] if feat["properties"].get("fldName")}
akerbp_fields = {k for k, v in op_map.items() if "Aker BP" in v}

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")
master = pd.read_csv(DATA / "master_fluid_library.csv")

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

def loo_cv(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    if len(y) < 10: return np.nan, len(y)
    loo = LeaveOneOut()
    lr = LinearRegression()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        lr.fit(X[tr], y[tr])
        preds[te] = lr.predict(X[te])
    return 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2), len(y)

# Build V5: medium-conf hybrid + field-specific T
df = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
df = df.rename(columns={"api_gravity": "api_old"})
master_subset = master[["field", "api_gravity", "api_quality_tier", "api_confidence",
                        "reservoir_temp_c"]].rename(columns={"api_gravity": "api_master"})
df = df.merge(master_subset, on="field", how="left")

MED_TIERS = {"operator_direct", "dst_robust", "operator_medium"}
HIGH_TIERS = {"operator_direct", "dst_robust"}

df["api_v5"] = df.apply(
    lambda r: r.api_master if (
        pd.notna(r.api_master) and
        r.api_quality_tier in MED_TIERS and
        r.api_confidence in ["high", "medium"]
    ) else r.api_old, axis=1
)
df["T_used_F"] = df.apply(
    lambda r: (r.reservoir_temp_c * 9/5 + 32) if (
        pd.notna(r.reservoir_temp_c) and r.reservoir_temp_c > 30 and
        r.api_quality_tier in HIGH_TIERS and r.api_confidence == "high"
    ) else 194, axis=1
)
df["T_used_C"] = (df.T_used_F - 32) * 5/9
df["visc_v5"] = df.apply(lambda r: beggs_robinson(r.api_v5, r.T_used_F), axis=1)
df["ln_visc_v5"] = np.log(df.visc_v5)

# Track which fields used field T
df["uses_field_T"] = df.T_used_F != 194

# Premium
post = panel[panel.is_post_peak].copy()
post["prod_frac"] = post.oil_pct_peak / 100.0
valid = df.dropna(subset=["ln_visc_v5", "D_annual"])
lr_p = LinearRegression().fit(valid[["ln_visc_v5"]].values, valid["D_annual"].values)
valid["D_physics"] = lr_p.predict(valid[["ln_visc_v5"]].values)
D_phys_map = valid.set_index("field")["D_physics"].to_dict()

rows = []
for field, grp in post.groupby("field"):
    if field not in D_phys_map: continue
    D_phys = D_phys_map[field]
    grp = grp.sort_values("months_since_peak")
    grp = grp[grp.prod_frac > 0.01]
    if len(grp) < 12: continue
    t = grp.months_since_peak.values
    log_prem = np.log(grp.prod_frac.values) - np.log(np.exp(-D_phys / 12 * t))
    mask = t >= t.max() - 12
    if mask.sum() < 6: continue
    rows.append({"field": field, "premium_12m": log_prem[mask].mean()})

prem = pd.DataFrame(rows)
full = valid.merge(prem, on="field", how="inner")
full["abs_premium_12m"] = full.premium_12m.abs()
full["is_akerbp"] = full.field.isin(akerbp_fields)

X = full[["ln_visc_v5", "premium_12m", "abs_premium_12m"]].values
y = full["D_annual"].values
lr = LinearRegression().fit(X, y)
cv_r2, n = loo_cv(X, y)
in_r2 = lr.score(X, y)
rmse = np.sqrt(np.mean((y - lr.predict(X))**2))
full["D_pred"] = lr.predict(X)
full["resid"] = y - full.D_pred
ak_rmse = np.sqrt((full[full.is_akerbp].resid**2).mean())
ak_mae = full[full.is_akerbp].resid.abs().mean()

# Standardized
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
lr_std = LinearRegression().fit(X_std, y)

# t-stats via statsmodels
import statsmodels.api as sm
X_sm = sm.add_constant(X)
ols = sm.OLS(y, X_sm).fit()

print(f"V5 MODEL — Final Results")
print(f"  In-sample R²: {in_r2:.3f}")
print(f"  CV R²:        {cv_r2:.3f}")
print(f"  RMSE:         {rmse:.4f}")
print(f"  Aker BP RMSE: {ak_rmse:.4f}")
print(f"  Aker BP MAE:  {ak_mae:.4f}")
print(f"  Hit rate (±0.05): {(full[full.is_akerbp].resid.abs() < 0.05).mean()*100:.0f}%")
print(f"\nFormula:")
print(f"  D = {lr.intercept_:+.4f}")
print(f"    {'+' if lr.coef_[0] >= 0 else ''}{lr.coef_[0]:.4f} × ln(viskositet)")
print(f"    {'+' if lr.coef_[1] >= 0 else ''}{lr.coef_[1]:.4f} × premium_12m")
print(f"    {'+' if lr.coef_[2] >= 0 else ''}{lr.coef_[2]:.4f} × |premium_12m|")
print(f"\nStandardized β:")
print(f"  ln(visc):    {lr_std.coef_[0]:+.3f}")
print(f"  premium:     {lr_std.coef_[1]:+.3f}")
print(f"  |premium|:   {lr_std.coef_[2]:+.3f}")
print(f"\nt-stats:")
for i, name in enumerate(["Intercept", "ln(visc)", "premium", "|premium|"]):
    print(f"  {name:15s}  t={ols.tvalues[i]:+.2f}  p={ols.pvalues[i]:.4f}")

# Stats on field T usage
n_field_T = full.uses_field_T.sum()
print(f"\nFields using field-specific T: {n_field_T}/{len(full)}")
print(f"Fields using reservoir API: {(full.api_old != full.api_v5).sum()}/{len(full)}")

# Aker BP details
akbp = full[full.is_akerbp].sort_values("D_annual", ascending=False).copy()
print(f"\n── AKER BP ──")
print(f"{'Field':18s} {'D_act':>6s} {'D_pred':>7s} {'Miss':>7s} {'API_v5':>7s} {'T_used':>8s} {'Prem':>6s}")
print("─" * 75)
for _, r in akbp.iterrows():
    T_str = f"{r.T_used_C:.0f}°C" if r.uses_field_T else "90°C"
    print(f"{r.field:18s} {r.D_annual:6.3f} {r.D_pred:7.3f} {r.resid:+7.3f} "
          f"{r.api_v5:7.1f} {T_str:>8s} {r.premium_12m:+5.2f}")

# Save predictions
full[["field", "api_old", "api_v5", "T_used_C", "uses_field_T", "D_annual", "D_pred",
       "resid", "premium_12m", "abs_premium_12m", "api_quality_tier", "api_confidence",
       "is_akerbp"]].to_csv(DATA / "predictions_v5_final.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# FINAL V5 FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 13))
fig.suptitle(f"V5 Endelig ER-Modell  |  CV R² = {cv_r2:.3f}  |  Reservoir-API + Field-T",
             fontsize=15, fontweight="bold", y=1.0)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

# Panel 1: Predicted vs actual
ax = fig.add_subplot(gs[0, 0])
others = full[~full.is_akerbp]
ax.scatter(others.D_pred, others.D_annual, c="lightgray", s=40, alpha=0.6, label="Andre NCS")
for _, r in akbp.iterrows():
    ax.scatter(r.D_pred, r.D_annual, c="#E91E63", s=80, zorder=5, edgecolors="white", lw=0.5)
    ax.annotate(r.field, (r.D_pred, r.D_annual), fontsize=6, alpha=0.85,
                xytext=(4, 3), textcoords="offset points")
lims = [-0.02, max(full.D_annual.max(), full.D_pred.max()) + 0.03]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
ax.set_xlabel("Predikert D")
ax.set_ylabel("Faktisk D")
ax.set_title(f"Predikert vs. Faktisk (in-R²={in_r2:.3f})", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(lims); ax.set_ylim(lims)

# Panel 2: Residual distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(full.resid, bins=15, color="#2E7D32", alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.axvline(full.resid.mean(), color="red", ls="--", lw=1.5, label=f"Mean: {full.resid.mean():+.3f}")
ax.set_xlabel("Residual"); ax.set_ylabel("Antall felt")
ax.set_title("Residual-fordeling (sentrert)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# Panel 3: Standardized coefficients
ax = fig.add_subplot(gs[0, 2])
labels_short = ["ln(visc)", "Premium\n12m", "|Premium|"]
colors_v = ["#FF9800" if c > 0 else "#2196F3" for c in lr_std.coef_]
ax.bar(range(3), lr_std.coef_, color=colors_v, alpha=0.85, edgecolor="white")
ax.set_xticks(range(3))
ax.set_xticklabels(labels_short, fontsize=10)
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Standardisert β")
ax.set_title("Variabel-viktighet", fontsize=11, fontweight="bold")
for i, c in enumerate(lr_std.coef_):
    ax.text(i, c + 0.005 * (1 if c >= 0 else -1), f"{c:+.3f}", ha="center", fontsize=9,
            fontweight="bold")

# Panel 4: Old API vs new API
ax = fig.add_subplot(gs[1, 0])
plot_df = full.dropna(subset=["api_old", "api_v5"])
ax.scatter(plot_df[~plot_df.is_akerbp].api_old, plot_df[~plot_df.is_akerbp].api_v5,
           c="lightgray", s=40, alpha=0.6, label="Andre")
ax.scatter(plot_df[plot_df.is_akerbp].api_old, plot_df[plot_df.is_akerbp].api_v5,
           c="#E91E63", s=70, alpha=0.85, label="Aker BP", edgecolors="white", lw=0.5)
lims = [15, 60]
ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
big = plot_df[(plot_df.api_v5 - plot_df.api_old).abs() > 3]
for _, r in big.iterrows():
    ax.annotate(r.field, (r.api_old, r.api_v5), fontsize=6, alpha=0.85,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Gammel API (blend)"); ax.set_ylabel("Ny API (reservoar/hybrid)")
ax.set_title("API-oppdatering: gammel vs ny", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# Panel 5: Premium vs decline
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(full.premium_12m, full.D_annual, c=full.ln_visc_v5, cmap="RdYlBu_r",
                s=60, alpha=0.75, edgecolors="white", lw=0.4)
for _, r in akbp.iterrows():
    ax.annotate(r.field, (r.premium_12m, r.D_annual), fontsize=6, alpha=0.85,
                xytext=(4, 3), textcoords="offset points")
prem_range = np.linspace(full.premium_12m.min(), full.premium_12m.max(), 100)
mean_visc = full.ln_visc_v5.mean()
y_curve = (lr.intercept_ + lr.coef_[0] * mean_visc + lr.coef_[1] * prem_range +
           lr.coef_[2] * np.abs(prem_range))
ax.plot(prem_range, y_curve, "k-", lw=2, alpha=0.5, label="Modell")
plt.colorbar(sc, ax=ax, label="ln(viskositet)")
ax.axvline(0, color="gray", ls=":", lw=0.7)
ax.set_xlabel("Premium 12m"); ax.set_ylabel("D_annual")
ax.set_title("U-formet premium-effekt", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper center")

# Panel 6: Aker BP predictions
ax = fig.add_subplot(gs[1, 2])
akbp_s = akbp.sort_values("D_annual")
y_pos = np.arange(len(akbp_s))
ax.barh(y_pos - 0.18, akbp_s.D_annual, 0.36, color="#1565C0", alpha=0.85, label="Faktisk")
ax.barh(y_pos + 0.18, akbp_s.D_pred, 0.36, color="#2E7D32", alpha=0.85, label="Modell")
ax.set_yticks(y_pos)
ax.set_yticklabels(akbp_s.field, fontsize=8)
ax.set_xlabel("D_annual")
ax.set_title(f"Aker BP: RMSE={ak_rmse:.3f}, MAE={ak_mae:.3f}", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Panel 7: Quality tier breakdown
ax = fig.add_subplot(gs[2, 0])
tier_counts = full.api_quality_tier.fillna("baseline").value_counts()
colors_tier = {
    "operator_direct": "#2E7D32",
    "dst_robust": "#1565C0",
    "operator_medium": "#FF9800",
    "dst_limited": "#9E9E9E",
    "operator_low": "#9C27B0",
    "blend_inherited": "#E91E63",
    "blend_direct": "#757575",
    "baseline": "#BDBDBD"
}
ax.barh(range(len(tier_counts)),
        tier_counts.values,
        color=[colors_tier.get(t, "#757575") for t in tier_counts.index],
        alpha=0.85)
ax.set_yticks(range(len(tier_counts)))
ax.set_yticklabels(tier_counts.index, fontsize=8)
ax.set_xlabel("Antall felt")
ax.set_title("API-kvalitet per felt", fontsize=11, fontweight="bold")

# Panel 8: Sources used
ax = fig.add_subplot(gs[2, 1])
ax.axis("off")
sources_text = f"""DATAKILDER (master fluid library)

──────── Direct measurements ────────
Sodir DST database:      1186 målinger
  → 85 felt, 39 med ≥5 samples
Operatør direct assays:  17 felt
  (Equinor, ExxonMobil, TotalEnergies)

──────── Documented research ────────
Operator presentations:  43 felt
  (Equinor, Aker BP, Vår Energi, OKEA,
   ConocoPhillips, Harbour, DNO)

──────── Total coverage ────────
Master fluid library:    110 felt
Quality tiers:
  High (direct):   {(full.api_quality_tier == 'operator_direct').sum()} felt
  Medium (DST≥5):  {(full.api_quality_tier == 'dst_robust').sum()} felt
  Med (op med):    {(full.api_quality_tier == 'operator_medium').sum()} felt
  Limited (DST<5): {(full.api_quality_tier == 'dst_limited').sum()} felt
  Low (op low):    {(full.api_quality_tier == 'operator_low').sum()} felt
  Blend fallback:  {(full.api_quality_tier == 'blend_inherited').sum()} felt"""

ax.text(0.05, 0.97, sources_text, transform=ax.transAxes, fontsize=8,
        fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="#E3F2FD", ec="#1565C0", alpha=0.9))

# Panel 9: Final formula box
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
formula = f"""V5 ER-MODELL — FORMEL

D = {lr.intercept_:+.4f}
  {'+' if lr.coef_[0]>=0 else ''}{lr.coef_[0]:.4f} × ln(μ)
  {'+' if lr.coef_[1]>=0 else ''}{lr.coef_[1]:.4f} × P₁₂
  {'+' if lr.coef_[2]>=0 else ''}{lr.coef_[2]:.4f} × |P₁₂|

der:
  μ  = BR-viskositet med
       reservoar-API + felt-T
  P₁₂ = log-premium 12 mnd
       vs fysikk-baseline

── Performance ──
  CV R²        {cv_r2:.3f}
  RMSE         {rmse:.4f}
  Aker BP RMSE {ak_rmse:.4f}
  Aker BP MAE  {ak_mae:.4f}
  Hit rate     {(full[full.is_akerbp].resid.abs() < 0.05).mean()*100:.0f}%

── Forbedring vs V1 ──
  Ærligere fysikk
  Reservoar-API (110 felt)
  Field-spesifikk T ({n_field_T} felt)"""

ax.text(0.05, 0.97, formula, transform=ax.transAxes, fontsize=9,
        fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="#E8F5E9", ec="#2E7D32", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_final_v5_model.png", dpi=160, bbox_inches="tight")
print(f"\nSaved: fig_final_v5_model.png")
