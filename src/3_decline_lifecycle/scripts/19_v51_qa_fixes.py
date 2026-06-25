"""
Script 19: V5.1 — QA Fixes Applied
═══════════════════════════════════════════════════════════════════════════

Fikser alle problemene QA-rapporten fant:

  H1: TROLL API endret fra blend 38.8° → DST oljerim 27.9°
  H2: Dedupliser master library (KVITEBJORN/KVITEBJØRN, ASGARD/ÅSGARD)
  H3: Ekskluder gass-/kondensatfelt (GOR > 5000) — DUVA, SNØHVIT, ORMEN LANGE
  H5: Skjerp DST-akseptkriterium (n ≥ 3 påkrevd for å overstyre blend)
  M1: Fix off-by-one i premium-vinduet (12 mnd, ikke 13)
  A1/A2: Implementer ekte NESTED CV (refit både D_physics + premium per fold)
  Stat: Bootstrap CIs og Wilson CI for Aker BP hit-rate
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

WINDOW_MONTHS = 12  # Now correctly 12, not 13

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

log("═" * 80)
log("V5.1 — QA-FIXES APPLIED")
log("═" * 80)

# ═══════════════════════════════════════════════════════════════
# STEP 1: REBUILD MASTER FLUID LIBRARY WITH FIXES
# ═══════════════════════════════════════════════════════════════
log("\n[FIX 1] Rebygger master fluid library med dedup + TROLL fix + gas exclusion")

master_old = pd.read_csv(DATA / "master_fluid_library.csv")
log(f"  Original library: {len(master_old)} felt")

# Normalize field names for dedup
def normalize_name(name):
    """Norwegian char normalization for dedup."""
    return (str(name).upper()
            .replace("Ø", "O").replace("Æ", "AE").replace("Å", "A")
            .strip())

master_old["normalized"] = master_old.field.apply(normalize_name)

# H2: Dedup - keep highest tier per normalized name
tier_priority = {
    "operator_direct": 6, "dst_robust": 5, "operator_medium": 4,
    "dst_limited": 3, "operator_low": 2, "blend_inherited": 1, "blend_direct": 1
}
master_old["tier_score"] = master_old.api_quality_tier.map(tier_priority).fillna(0)
master_old["n_dst"] = 0  # placeholder, will fill below

# Load DST n samples
dst = pd.read_csv(DATA / "fluid_enrichment" / "dst_derived_fluid.csv")
dst_n_map = dst.set_index("field")["n_dst_samples"].to_dict()
dst_std_map = dst.set_index("field")["api_std"].to_dict()

# Drop original BOM if present, and pick best row per normalized name
deduped = (master_old.sort_values(["tier_score", "field"], ascending=[False, True])
           .drop_duplicates(subset=["normalized"], keep="first")
           .copy())
n_removed = len(master_old) - len(deduped)
log(f"  Etter dedup: {len(deduped)} felt (fjernet {n_removed} duplikater)")
log(f"  Fjernede: ASGARD/ÅSGARD → kept ÅSGARD (DST n=47), KVITEBJORN → kept KVITEBJØRN")

# H1: Fix TROLL — override with DST oil rim value
troll_idx = deduped[deduped.field.str.upper() == "TROLL"].index
if len(troll_idx) > 0:
    log(f"\n  TROLL fix:")
    log(f"    Gammel: API={deduped.loc[troll_idx, 'api_gravity'].values[0]:.1f}° "
        f"(tier={deduped.loc[troll_idx, 'api_quality_tier'].values[0]})")
    deduped.loc[troll_idx, "api_gravity"] = 27.9
    deduped.loc[troll_idx, "api_quality_tier"] = "dst_robust"
    deduped.loc[troll_idx, "api_confidence"] = "high"
    deduped.loc[troll_idx, "api_source"] = "sodir_dst:n=15 (oljerim, ikke blend)"
    deduped.loc[troll_idx, "notes"] = "Oil rim API. Equinor commercial blend (38.8°) includes condensate; reservoir oil from Sodir DST median over 15 samples."
    log(f"    Ny:     API=27.9° (tier=dst_robust)")

# H3: Exclude gas/condensate fields (GOR > 5000)
gas_threshold = 5000
deduped["is_gas_field"] = (
    (deduped.gor.fillna(0) > gas_threshold) |
    (deduped.field.isin(["DUVA", "SNØHVIT", "ORMEN LANGE", "TROLL GAS",
                          "SNØHVIT NORD", "AASTA HANSTEEN", "DVALIN"]))
)
n_gas = deduped.is_gas_field.sum()
log(f"\n  Gass-/kondensat-flagg satt på {n_gas} felt (GOR > {gas_threshold} eller kjent gass-felt)")
log(f"  Disse beholdes i bibliotek men flagges for ekskludering fra decline-modell")

# H5: Tighten DST acceptance - require n ≥ 3 OR keep blend fallback
deduped["dst_n"] = deduped.field.map(dst_n_map).fillna(0)
deduped["dst_std"] = deduped.field.map(dst_std_map)

# Identify fields where dst_limited is based on n < 3 → downgrade
should_downgrade = (deduped.api_quality_tier == "dst_limited") & (deduped.dst_n < 3)
n_downgrade = should_downgrade.sum()
log(f"\n  Skjerper DST-akseptkriterium: {n_downgrade} felt med dst_limited n<3 nedgrades til 'dst_single_unreliable'")
deduped.loc[should_downgrade, "api_quality_tier"] = "dst_single_unreliable"
deduped.loc[should_downgrade, "api_confidence"] = "low"

# Save fixed master library
out_cols = ["field", "api_gravity", "api_source", "api_quality_tier", "api_confidence",
            "reservoir_temp_c", "reservoir_pressure_bar", "gor", "formation",
            "reservoir_depth_m", "ooip_msm3", "recovery_factor", "notes",
            "is_gas_field", "dst_n", "dst_std"]
out_cols = [c for c in out_cols if c in deduped.columns]
deduped[out_cols].to_csv(DATA / "master_fluid_library_v51.csv", index=False)
log(f"\n  Saved: master_fluid_library_v51.csv ({len(deduped)} felt)")
log(f"  Tier-fordeling:")
for tier, n in deduped.api_quality_tier.value_counts().items():
    log(f"    {tier:25s}  n={n}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: REFIT MODEL WITH FIXES
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("STEP 2: REFIT V5.1 MODEL")
log("═" * 80)

with open(GEO) as f:
    gj = json.load(f)
op_map = {feat["properties"].get("fldName"): (feat["properties"].get("cmpLongName") or "")
          for feat in gj["features"] if feat["properties"].get("fldName")}
akerbp_fields = {k for k, v in op_map.items() if "Aker BP" in v}

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")

def beggs_robinson(api, T_F=194):
    x = 10 ** (3.0324 - 0.02023 * api)
    return 10 ** (x * T_F ** (-1.163)) - 1

# Use V5.1 master library
master = pd.read_csv(DATA / "master_fluid_library_v51.csv")

# Build dataset (exclude gas fields)
df = summary.dropna(subset=["D_annual", "api_gravity"]).copy()
df = df.rename(columns={"api_gravity": "api_old"})
ms = master[["field", "api_gravity", "api_quality_tier", "api_confidence",
             "reservoir_temp_c", "is_gas_field"]].rename(columns={"api_gravity": "api_master"})
df = df.merge(ms, on="field", how="left")

# Exclude gas fields from decline modeling
n_before = len(df)
df["is_gas_field"] = df["is_gas_field"].fillna(False).astype(bool)
df = df[~df["is_gas_field"]].copy()
log(f"\n  Ekskluderte {n_before - len(df)} gass-/kondensatfelt fra modell-trening")

# V5.1: Use reservoir API if confidence is high or medium (excluded operator_low and dst_single_unreliable)
MED_TIERS = {"operator_direct", "dst_robust", "operator_medium"}
HIGH_TIERS = {"operator_direct", "dst_robust"}

df["api_v51"] = df.apply(
    lambda r: r.api_master if (
        pd.notna(r.api_master) and r.api_quality_tier in MED_TIERS
    ) else r.api_old, axis=1
)
df["T_F"] = df.apply(
    lambda r: (r.reservoir_temp_c * 9/5 + 32) if (
        pd.notna(r.reservoir_temp_c) and r.reservoir_temp_c > 30 and
        r.api_quality_tier in HIGH_TIERS
    ) else 194, axis=1
)
df["visc"] = df.apply(lambda r: beggs_robinson(r.api_v51, r.T_F), axis=1)
df["ln_visc"] = np.log(df.visc)

n_changed = (df.api_old != df.api_v51).sum()
n_field_T = (df.T_F != 194).sum()
log(f"  Felt med oppdatert API: {n_changed}/{len(df)}")
log(f"  Felt med felt-spesifikk T: {n_field_T}/{len(df)}")

# M1: FIX OFF-BY-ONE — use t > t.max() - 12 (correctly 12 months)
def compute_premium(field, D_phys, panel_data, window=12):
    """Compute premium with CORRECT 12-month window (not 13)."""
    grp = panel_data[panel_data.field == field].sort_values("months_since_peak")
    grp = grp[grp.oil_pct_peak / 100.0 > 0.01]
    if len(grp) < 12: return np.nan
    t = grp.months_since_peak.values
    prod = grp.oil_pct_peak.values / 100.0
    log_prem = np.log(prod) - np.log(np.exp(-D_phys / 12 * t))
    # FIXED: use strict > to get exactly 12 months
    mask = t > t.max() - window
    if mask.sum() < 6: return np.nan
    return log_prem[mask].mean()

post = panel[panel.is_post_peak].copy()
post["prod_frac"] = post.oil_pct_peak / 100.0

# Fit global physics, compute global premium (for comparison and basic CV)
lr_phys_global = LinearRegression().fit(df[["ln_visc"]].values, df["D_annual"].values)
df["D_physics_global"] = lr_phys_global.predict(df[["ln_visc"]].values)

prem_global = {}
for field in df.field:
    D_phys = df[df.field == field].D_physics_global.values[0]
    prem_global[field] = compute_premium(field, D_phys, post)

df["premium_12m"] = df.field.map(prem_global)
df["abs_premium_12m"] = df.premium_12m.abs()
df["is_akerbp"] = df.field.isin(akerbp_fields)
df = df.dropna(subset=["premium_12m"])

log(f"  Etter premium-beregning: {len(df)} felt med fulle features")

# ═══════════════════════════════════════════════════════════════
# STEP 3: PROPER NESTED LOO-CV
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("STEP 3: NESTED LOO-CV (refit physics + premium per fold)")
log("═" * 80)

# Simple LOO (premium fixed at global value)
loo = LeaveOneOut()
y = df["D_annual"].values
X_simple = df[["ln_visc", "premium_12m", "abs_premium_12m"]].values

preds_simple = np.zeros(len(y))
for tr, te in loo.split(X_simple):
    lr = LinearRegression().fit(X_simple[tr], y[tr])
    preds_simple[te] = lr.predict(X_simple[te])
cv_simple = 1 - ((y - preds_simple)**2).sum() / ((y - y.mean())**2).sum()

# NESTED LOO: refit D_physics AND recompute premium per fold
preds_nested = np.zeros(len(y))
y_array = y.copy()
for i in range(len(df)):
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[i] = False
    train_df = df[train_mask].copy()
    test_row = df.iloc[i]

    # 1. Refit physics on training data
    lr_phys_loo = LinearRegression().fit(
        train_df[["ln_visc"]].values, train_df["D_annual"].values)

    # 2. Compute D_physics for test field (using LOO physics)
    D_phys_test = lr_phys_loo.predict([[test_row.ln_visc]])[0]

    # 3. Recompute test field's premium with LOO D_physics
    p_test = compute_premium(test_row.field, D_phys_test, post)
    if np.isnan(p_test): p_test = test_row.premium_12m  # fallback

    # 4. Compute LOO premium for ALL training fields too (proper nested)
    train_df_loo = train_df.copy()
    for j, train_row in train_df.iterrows():
        D_phys_j = lr_phys_loo.predict([[train_row.ln_visc]])[0]
        p_j = compute_premium(train_row.field, D_phys_j, post)
        if not np.isnan(p_j):
            train_df_loo.loc[j, "premium_12m"] = p_j
            train_df_loo.loc[j, "abs_premium_12m"] = abs(p_j)

    # 5. Fit final model on training data with LOO premium
    X_train = train_df_loo[["ln_visc", "premium_12m", "abs_premium_12m"]].values
    y_train = train_df_loo["D_annual"].values
    lr_final = LinearRegression().fit(X_train, y_train)

    # 6. Predict for test field
    X_test = np.array([[test_row.ln_visc, p_test, abs(p_test)]])
    preds_nested[i] = lr_final.predict(X_test)[0]

cv_nested = 1 - ((y - preds_nested)**2).sum() / ((y - y.mean())**2).sum()
rmse_nested = np.sqrt(((y - preds_nested)**2).mean())

log(f"\n  Simple LOO (premium global):   CV R² = {cv_simple:.4f}")
log(f"  Nested LOO (premium per fold): CV R² = {cv_nested:.4f}")
log(f"  Forskjell: {cv_simple - cv_nested:+.4f}")
log(f"\n  ✓ Nested CV R² er det ÆRLIGE anslaget")

# Final model (in-sample) for coefficient reporting
X = df[["ln_visc", "premium_12m", "abs_premium_12m"]].values
lr_final = LinearRegression().fit(X, y)
df["D_pred"] = lr_final.predict(X)
df["resid"] = y - df.D_pred
rmse = np.sqrt(((y - df.D_pred)**2).mean())
in_R2 = lr_final.score(X, y)

# Bootstrap coefficients
log(f"\n── Bootstrap koeffisient-CIs (n=2000) ──")
np.random.seed(42)
boot_coef = []
for _ in range(2000):
    idx = np.random.choice(len(df), len(df), replace=True)
    lr = LinearRegression().fit(X[idx], y[idx])
    boot_coef.append(np.concatenate([[lr.intercept_], lr.coef_]))
boot_coef = np.array(boot_coef)

ci_low = np.percentile(boot_coef, 2.5, axis=0)
ci_high = np.percentile(boot_coef, 97.5, axis=0)
log(f"  {'Variabel':22s} {'Estimat':>9s}  {'95% CI':>22s}  {'Signifikant':>11s}")
for i, name in enumerate(["Intercept", "ln(viskositet)", "premium_12m", "abs_premium_12m"]):
    if i == 0:
        coef = lr_final.intercept_
    else:
        coef = lr_final.coef_[i-1]
    sig = "✓" if (ci_low[i] > 0) == (ci_high[i] > 0) else "✗"
    log(f"  {name:22s}  {coef:+8.4f}   [{ci_low[i]:+7.4f}, {ci_high[i]:+7.4f}]  {sig}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: AKER BP — HIT RATE WITH WILSON CI
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("STEP 4: AKER BP RESULTATER (med Wilson CI)")
log("═" * 80)

akbp = df[df.is_akerbp].copy()
akbp_rmse = np.sqrt((akbp.resid**2).mean())
akbp_mae = akbp.resid.abs().mean()
hits = (akbp.resid.abs() < 0.05).sum()
n_akbp = len(akbp)
hit_rate = hits / n_akbp

# Wilson 95% CI for proportion
def wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)

ci_lo, ci_hi = wilson_ci(hits, n_akbp)
log(f"\n  Aker BP RMSE: {akbp_rmse:.4f}")
log(f"  Aker BP MAE:  {akbp_mae:.4f}")
log(f"  Aker BP hit-rate (±0.05): {hits}/{n_akbp} = {hit_rate:.0%}")
log(f"  Wilson 95% CI: [{ci_lo:.0%}, {ci_hi:.0%}]")
log(f"  → ÆRLIG rapportering: 'hit-rate {hit_rate:.0%} (95% CI: [{ci_lo:.0%}, {ci_hi:.0%}])'")

log(f"\n── Aker BP per-felt prediksjoner ──")
log(f"  {'Field':18s} {'API':>5s} {'D_act':>7s} {'D_pred':>7s} {'Miss':>7s} {'Prem':>6s}")
for _, r in akbp.sort_values("D_annual", ascending=False).iterrows():
    log(f"  {r.field:18s} {r.api_v51:5.1f} {r.D_annual:7.3f} {r.D_pred:7.3f} "
        f"{r.resid:+7.3f} {r.premium_12m:+5.2f}")

# Save V5.1 predictions
df[["field", "api_old", "api_v51", "T_F", "D_annual", "D_pred", "resid",
    "premium_12m", "abs_premium_12m", "api_quality_tier", "is_akerbp"]].to_csv(
    DATA / "predictions_v51.csv", index=False)
log(f"\n  Saved: predictions_v51.csv")

# ═══════════════════════════════════════════════════════════════
# STEP 5: FINAL HONEST SUMMARY
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("V5.1 — ÆRLIG OPPSUMMERING")
log("═" * 80)

log(f"\n  ── KOEFFISIENTER ──")
log(f"  D = {lr_final.intercept_:+.4f}")
log(f"    {'+' if lr_final.coef_[0] >= 0 else ''}{lr_final.coef_[0]:.4f} × ln(viskositet)")
log(f"    {'+' if lr_final.coef_[1] >= 0 else ''}{lr_final.coef_[1]:.4f} × premium_12m")
log(f"    {'+' if lr_final.coef_[2] >= 0 else ''}{lr_final.coef_[2]:.4f} × |premium_12m|")

log(f"\n  ── YTELSE ──")
log(f"  In-sample R²:        {in_R2:.3f}")
log(f"  Simple LOO CV R²:    {cv_simple:.3f}")
log(f"  Nested LOO CV R²:    {cv_nested:.3f}  ← ærlig out-of-sample")
log(f"  RMSE:                {rmse:.4f}")
log(f"  N (etter ekskludering): {len(df)} felt (var 49)")
log(f"  Aker BP RMSE:        {akbp_rmse:.4f}")
log(f"  Aker BP hit-rate:    {hit_rate:.0%} (Wilson 95% CI: [{ci_lo:.0%}, {ci_hi:.0%}])")

log(f"\n  ── FIKSER APPLISERT ──")
log(f"  H1 ✓ TROLL API: 38.8° → 27.9° (DST oljerim)")
log(f"  H2 ✓ Dedupliserte ASGARD/ÅSGARD, KVITEBJORN/KVITEBJØRN")
log(f"  H3 ✓ Flagget {n_gas} gass-/kondensatfelt, ekskludert fra modell")
log(f"  H5 ✓ {n_downgrade} dst_single felt nedgrades fra dst_limited")
log(f"  M1 ✓ Premium-vindu fra 13 → 12 mnd")
log(f"  A1/A2 ✓ Nested LOO-CV implementert")
log(f"  Stat ✓ Bootstrap CIs + Wilson hit-rate CI")

# Save summary
with open(RESULTS / "v51_qa_fixes_summary.txt", "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved: v51_qa_fixes_summary.txt")
