"""
Script 01: Compute monthly decline rates and build analysis datasets.

Outputs:
  - data/panel_monthly.csv   — monthly panel (field × month) with decline rates + quality
  - data/field_summary.csv   — one row per field with avg decline + quality features

Production is normalized as % of peak for each field, removing scale effects.
An exponential decline constant (D) is fitted per field on post-peak data:
  P(t) = P_peak * exp(-D * t)   →   D > 0 means declining
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ── Field → assay grade mapping (from script 49) ────────────────────────────

DIRECT_ASSAY = {
    "JOHAN SVERDRUP": "Johan Sverdrup",
    "ALVHEIM": "Alvheim",
    "GRANE": "Grane",
    "OSEBERG": "Oseberg",
    "TROLL": "Troll",
    "EKOFISK": "Ekofisk",
    "STATFJORD": "Statfjord",
    "GULLFAKS": "Gullfaks",
    "HEIDRUN": "Heidrun",
    "NORNE": "Norne",
    "SKARV": "Skarv",
    "BALDER": "Balder",
    "DRAUGEN": "Draugen",
    "ÅSGARD": "Asgard",
    "GUDRUN": "Gudrun",
    "GOLIAT": "Goliat",
    "GINA KROG": "Gina Krog",
    "JOTUN": "Jotun",
    "KNARR": "Knarr",
    "MARTIN LINGE": "Martin Linge",
    "NJORD": "Njord",
}

BLEND_PROXY = {
    "BØYLA": "Alvheim",
    "SKOGUL": "Alvheim",
    "VOLUND": "Alvheim",
    "VILJE": "Alvheim",
    "EDVARD GRIEG": "Grane",
    "IVAR AASEN": "Grane",
    "SOLVEIG": "Grane",
    "SVERDRUP PHASE 2": "Johan Sverdrup",
    "VALHALL": "Ekofisk",
    "HOD": "Ekofisk",
    "ELDFISK": "Ekofisk",
    "EMBLA": "Ekofisk",
    "ULA": "Ekofisk",
    "TAMBAR": "Ekofisk",
    "TAMBAR ØST": "Ekofisk",
    "GYDA": "Ekofisk",
    "OSEBERG ØST": "Oseberg",
    "OSEBERG SØR": "Oseberg",
    "FRAM": "Troll",
    "FRAM H-NORD": "Troll",
    "STATFJORD NORD": "Statfjord",
    "STATFJORD ØST": "Statfjord",
    "SYGNA": "Statfjord",
    "SNORRE": "Statfjord",
    "VIGDIS": "Statfjord",
    "TORDIS": "Statfjord",
    "VISUND": "Gullfaks",
    "KVITEBJØRN": "Gullfaks",
    "VALEMON": "Gullfaks",
    "MARULK": "Norne",
    "SKULD": "Norne",
    "URD": "Norne",
    "AASTA HANSTEEN": "Skarv",
    "MARIA": "Heidrun",
}

FIELD_TO_GRADE = {**DIRECT_ASSAY, **{k: v for k, v in BLEND_PROXY.items()}}

# ── 1. Load production data ─────────────────────────────────────────────────

prod = pd.read_csv(ROOT / "data/raw/sodir/sodir_field_production_monthly.csv")
prod = prod.rename(columns={
    "prfInformationCarrier": "field",
    "prfYear": "year",
    "prfMonth": "month",
    "prfPrdOilNetMillSm3": "oil_msm3",
    "prfPrdGasNetBillSm3": "gas_bsm3",
    "prfPrdProducedWaterInFieldMillSm3": "water_msm3",
})

prod = prod[prod.field.isin(FIELD_TO_GRADE)].copy()
prod["date"] = pd.to_datetime(prod[["year", "month"]].assign(day=1))
prod = prod.sort_values(["field", "date"]).reset_index(drop=True)

print(f"Fields with assay mapping: {prod.field.nunique()}")
print(f"Rows after filtering to mapped fields: {len(prod):,}")

# ── 2. Compute decline rates ────────────────────────────────────────────────

prod["oil_prev"] = prod.groupby("field")["oil_msm3"].shift(1)
prod["decline_rate"] = (prod["oil_msm3"] - prod["oil_prev"]) / prod["oil_prev"]

# GOR (gas-oil ratio)
prod["gor"] = np.where(
    prod["oil_msm3"] > 0.001,
    prod["gas_bsm3"] / prod["oil_msm3"],
    np.nan,
)

# Water cut
total_fluid = prod["oil_msm3"] + prod["water_msm3"]
prod["water_cut"] = np.where(total_fluid > 0.001, prod["water_msm3"] / total_fluid, np.nan)

# ── 3. Identify first production month per field (for ramp-up filter) ───────

first_prod = (
    prod[prod.oil_msm3 > 0.001]
    .groupby("field")["date"]
    .min()
    .rename("first_prod_date")
)
prod = prod.merge(first_prod, on="field", how="left")
prod["months_since_start"] = (
    (prod.date.dt.year - prod.first_prod_date.dt.year) * 12
    + (prod.date.dt.month - prod.first_prod_date.dt.month)
)

# ── 4. Filter: remove ramp-up, shutdowns, and extreme outliers ──────────────

n_before = len(prod)

# Remove first 12 months (ramp-up)
prod = prod[prod.months_since_start >= 12].copy()
print(f"After removing ramp-up (<12 months): {len(prod):,} rows (dropped {n_before - len(prod):,})")

# Remove near-zero production months (maintenance shutdowns)
n_before = len(prod)
prod = prod[prod.oil_msm3 > 0.001].copy()
print(f"After removing near-zero months: {len(prod):,} rows (dropped {n_before - len(prod):,})")

# Remove first row per field after filtering (no valid decline_rate)
prod["oil_prev"] = prod.groupby("field")["oil_msm3"].shift(1)
prod["decline_rate"] = (prod["oil_msm3"] - prod["oil_prev"]) / prod["oil_prev"]
prod = prod.dropna(subset=["decline_rate"]).copy()

# Winsorize at 1st/99th percentile
p01 = prod.decline_rate.quantile(0.01)
p99 = prod.decline_rate.quantile(0.99)
n_clipped = ((prod.decline_rate < p01) | (prod.decline_rate > p99)).sum()
prod["decline_rate"] = prod.decline_rate.clip(p01, p99)
print(f"Winsorized {n_clipped:,} obs at [{p01:.3f}, {p99:.3f}]")

# ── 4b. Normalize production to % of peak per field ─────────────────────────

peak_prod = prod.groupby("field")["oil_msm3"].max().rename("peak_oil_msm3")
prod = prod.merge(peak_prod, on="field", how="left")
prod["oil_pct_peak"] = (prod["oil_msm3"] / prod["peak_oil_msm3"]) * 100

# Months since peak (for exponential decline fitting)
peak_dates = prod.loc[prod.groupby("field")["oil_msm3"].idxmax(), ["field", "date"]].rename(columns={"date": "peak_date"})
prod = prod.merge(peak_dates, on="field", how="left")
prod["months_since_peak"] = (
    (prod.date.dt.year - prod.peak_date.dt.year) * 12
    + (prod.date.dt.month - prod.peak_date.dt.month)
)
prod["is_post_peak"] = prod.months_since_peak > 0

# Decline rate on normalized series (% of peak)
prod["oil_pct_peak_prev"] = prod.groupby("field")["oil_pct_peak"].shift(1)
prod["decline_rate_norm"] = (prod["oil_pct_peak"] - prod["oil_pct_peak_prev"]) / prod["oil_pct_peak_prev"]

print(f"Post-peak observations: {prod.is_post_peak.sum():,} ({prod.is_post_peak.mean():.1%})")

# ── 4c. Fit exponential decline constant D per field ────────────────────────
# P(t) = 100 * exp(-D * t)  where t = months since peak, D > 0 = declining

def fit_exp_decline(grp):
    """Fit D from post-peak data. Returns D (monthly), R², n_post_peak."""
    post = grp[grp.is_post_peak & (grp.oil_pct_peak > 1)].sort_values("months_since_peak")
    if len(post) < 12:
        return pd.Series({"D_monthly": np.nan, "D_annual": np.nan, "D_r2": np.nan,
                          "n_post_peak": len(post), "half_life_months": np.nan})

    t = post.months_since_peak.values.astype(float)
    y = post.oil_pct_peak.values

    # Log-linear fit: ln(y) = ln(100) - D*t
    log_y = np.log(y)
    slope, intercept = np.polyfit(t, log_y, 1)
    D = -slope  # positive D = declining

    y_pred = np.exp(intercept + slope * t)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    half_life = np.log(2) / D if D > 0 else np.nan

    return pd.Series({
        "D_monthly": D,
        "D_annual": D * 12,
        "D_r2": r2,
        "n_post_peak": len(post),
        "half_life_months": half_life,
    })

print("Fitting exponential decline curves...")
decline_fits = prod.groupby("field").apply(fit_exp_decline).reset_index()
valid_fits = decline_fits.D_monthly.notna().sum()
print(f"Fitted D for {valid_fits}/{decline_fits.field.nunique()} fields")
print(f"  D_annual: mean={decline_fits.D_annual.mean():.4f}, median={decline_fits.D_annual.median():.4f}")
print(f"  Half-life: median={decline_fits.half_life_months.median():.0f} months")
print(f"  Fit R²: median={decline_fits.D_r2.median():.3f}")

# ── 5. Join assay quality data ──────────────────────────────────────────────

assays = pd.read_csv(ROOT / "data/processed/unified_crude_assays.csv")

QUALITY_COLS = [
    "api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct",
    "naphtha_pct", "middle_distillate_pct", "bottom_of_barrel_pct",
    "ccr_pct", "vanadium_ppm", "nickel_ppm", "nitrogen_ppm",
    "viscosity_cst_40c", "tan_mgkoh", "wax_pct",
]

existing_cols = [c for c in QUALITY_COLS if c in assays.columns]
assay_subset = assays[["grade"] + existing_cols].copy()

prod["grade"] = prod.field.map(FIELD_TO_GRADE)
prod["is_direct_assay"] = prod.field.isin(DIRECT_ASSAY)
prod = prod.merge(assay_subset, on="grade", how="left")

matched = prod.api_gravity.notna().sum()
print(f"Assay match rate: {matched}/{len(prod)} ({matched/len(prod):.1%})")

# ── 6. Join GeoJSON metadata ────────────────────────────────────────────────

with open(ROOT / "data/raw/sodir_geo/fields.geojson") as f:
    geo = json.load(f)

geo_rows = []
for feat in geo["features"]:
    p = feat["properties"]
    geo_rows.append({
        "geo_field": p["fldName"],
        "discovery_year": p.get("fldDiscoveryYear"),
        "hc_type": p.get("fldHcType"),
        "main_area": p.get("fldMainArea"),
        "operator": p.get("cmpLongName"),
        "status": p.get("fldCurrentActivitySatus"),
    })

geo_df = pd.DataFrame(geo_rows)
geo_df["field"] = geo_df.geo_field.str.upper()

# Some fields have multiple entries — keep unique
geo_df = geo_df.drop_duplicates(subset="field", keep="first")

prod = prod.merge(
    geo_df[["field", "discovery_year", "hc_type", "main_area", "operator"]],
    on="field",
    how="left",
)

# Field age at observation time
prod["field_age"] = prod.year - prod.discovery_year

# ── 7. Join monthly Brent price (control) ───────────────────────────────────

brent = pd.read_csv(ROOT / "data/processed/regression_panel.csv")
brent_monthly = (
    brent.groupby(["year", "month"])["brent_price"]
    .first()
    .reset_index()
)
brent_monthly = brent_monthly.rename(columns={"brent_price": "brent_usd"})
brent_monthly["year"] = brent_monthly["year"].astype(int)
brent_monthly["month"] = brent_monthly["month"].astype(int)

prod = prod.merge(brent_monthly, on=["year", "month"], how="left")
prod["ln_brent"] = np.log(prod.brent_usd.clip(lower=1))

# ── 8. Output datasets ─────────────────────────────────────────────────────

panel_cols = [
    "field", "grade", "is_direct_assay", "year", "month", "date",
    "oil_msm3", "oil_prev", "decline_rate",
    "peak_oil_msm3", "oil_pct_peak", "months_since_peak", "is_post_peak",
    "decline_rate_norm", "gor", "water_cut",
    "months_since_start", "field_age",
    "discovery_year", "hc_type", "main_area", "operator",
    "brent_usd", "ln_brent",
] + existing_cols

panel = prod[panel_cols].copy()
panel.to_csv(OUT / "panel_monthly.csv", index=False)
print(f"\nPanel: {len(panel):,} obs, {panel.field.nunique()} fields")

# Field-level summary
summary = (
    panel.groupby(["field", "grade", "is_direct_assay"])
    .agg(
        n_months=("decline_rate", "count"),
        decline_mean=("decline_rate", "mean"),
        decline_median=("decline_rate", "median"),
        decline_std=("decline_rate", "std"),
        decline_p25=("decline_rate", lambda x: x.quantile(0.25)),
        decline_p75=("decline_rate", lambda x: x.quantile(0.75)),
        oil_mean=("oil_msm3", "mean"),
        gor_mean=("gor", "mean"),
        water_cut_mean=("water_cut", "mean"),
        field_age_mean=("field_age", "mean"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        **{col: (col, "first") for col in existing_cols},
        main_area=("main_area", "first"),
        discovery_year=("discovery_year", "first"),
    )
    .reset_index()
)

# Merge exponential decline fits
summary = summary.merge(decline_fits, on="field", how="left")

summary.to_csv(OUT / "field_summary.csv", index=False)
print(f"Summary: {len(summary)} fields")

# Quick sanity check
print(f"\n── Sanity check ──")
print(f"MoM decline rate: mean={panel.decline_rate.mean():.4f}, median={panel.decline_rate.median():.4f}")
print(f"Exp. D (annual): mean={summary.D_annual.mean():.4f}, median={summary.D_annual.median():.4f}")
print(f"Half-life: median={summary.half_life_months.median():.0f} months ({summary.half_life_months.median()/12:.1f} yr)")
print(f"API range: {panel.api_gravity.min():.1f} – {panel.api_gravity.max():.1f}")
print(f"Sulfur range: {panel.sulfur_pct.min():.3f} – {panel.sulfur_pct.max():.3f}")
print(f"Brent coverage: {panel.brent_usd.notna().mean():.1%}")
print(f"Field age range: {panel.field_age.min():.0f} – {panel.field_age.max():.0f} years")
