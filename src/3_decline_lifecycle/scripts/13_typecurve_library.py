"""
Script 13: Type-Curve Bibliotek (FULL Sodir-datasett)
═══════════════════════════════════════════════════════════════════════════

Bygger et systematisk bibliotek over ALLE NCS-felt (inkl. nedstengte) med
fase-parametere. Kilde: Sodir månedlig produksjonsdata (132 felt) i stedet
for vårt begrensede panel_monthly (52 felt).

  RAMP    : Førsteolje → produksjon når 85% av peak
            - Lengde (mnd)
            - Form (logistisk fit: midpoint, steepness)
            - Type (rask FPSO / medium plattform / langsom hub)

  PLATEAU : Måneder produksjon holder seg > 85% av peak
            - Lengde (mnd)
            - Volatilitet (std)
            - Peak-rate (kSm3/mnd)

  DECLINE : Etter siste platå-måned
            - D_12 (første år)
            - D_decline_fit (full decline)
            - Premium vs fysikk

Output:
  - data/typecurve_library.csv    — én rad per felt med alle parametre
  - results/fig_typecurve_library.png — visualisering
  - results/typecurve_library.txt  — analyse-tabell
"""

import json, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir"
GEO = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir_geo" / "fields.geojson"

# Phase detection thresholds
PEAK_THRESHOLD = 0.85       # Plateau = months at > 85% of smoothed peak
SMOOTHING_WINDOW = 6        # Months for rolling mean
MIN_PLATEAU_MONTHS = 3      # Min consecutive months to count as plateau
MIN_RAMP_OBS_FOR_FIT = 6    # Min observations to fit logistic ramp
MIN_OIL_PRODUCTION = 0.5    # Min cum oil (Mill Sm3) to include field
MIN_MONTHS = 24             # Min months of production
MIN_OIL_FRACTION = 0.5      # Min fraction of OE that is oil (filter out gas fields)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

# ═══════════════════════════════════════════════════════════════
# LOAD FULL SODIR DATASET
# ═══════════════════════════════════════════════════════════════
log("═" * 80)
log("SCRIPT 13: TYPE-CURVE BIBLIOTEK (utvidet med full Sodir)")
log("═" * 80)

sodir_raw = pd.read_csv(RAW / "sodir_field_production_monthly.csv")
sodir_raw["date"] = pd.to_datetime(
    sodir_raw.prfYear.astype(str) + "-" + sodir_raw.prfMonth.astype(str) + "-01",
    format="%Y-%m-%d", errors="coerce"
)

log(f"\nSodir total: {len(sodir_raw)} rader, {sodir_raw.prfInformationCarrier.nunique()} felt")

# Filter: oil-dominated fields, min production, min months
field_stats = sodir_raw.groupby("prfInformationCarrier").agg(
    cum_oil=("prfPrdOilNetMillSm3", "sum"),
    cum_oe=("prfPrdOeNetMillSm3", "sum"),
    n_months=("prfPrdOilNetMillSm3", "size"),
).reset_index()
field_stats["oil_fraction"] = field_stats.cum_oil / field_stats.cum_oe.replace(0, np.nan)

oil_fields = field_stats[
    (field_stats.cum_oil >= MIN_OIL_PRODUCTION) &
    (field_stats.n_months >= MIN_MONTHS) &
    (field_stats.oil_fraction >= MIN_OIL_FRACTION)
].prfInformationCarrier.tolist()

log(f"Oljefelt etter filtrering: {len(oil_fields)}")
log(f"  Krav: ≥ {MIN_OIL_PRODUCTION} MSm³ olje totalt")
log(f"        ≥ {MIN_MONTHS} mnd produksjon")
log(f"        ≥ {MIN_OIL_FRACTION*100:.0f}% av OE er olje")

# Build panel for these fields
panel = sodir_raw[sodir_raw.prfInformationCarrier.isin(oil_fields)].copy()
panel = panel.rename(columns={
    "prfInformationCarrier": "field",
    "prfPrdOilNetMillSm3": "oil_msm3",
    "prfPrdGasNetBillSm3": "gas_bsm3",
    "prfPrdOeNetMillSm3": "oe_msm3",
    "prfPrdProducedWaterInFieldMillSm3": "water_msm3",
})
panel = panel.sort_values(["field", "date"]).reset_index(drop=True)

# Compute oil_pct_peak per field (months_since_start, peak normalization)
panel["months_since_start"] = panel.groupby("field").cumcount()
panel["peak_oil"] = panel.groupby("field").oil_msm3.transform("max")
panel["oil_pct_peak"] = panel.oil_msm3 / panel.peak_oil * 100

log(f"\nFinal panel: {len(panel)} rader, {panel.field.nunique()} felt")
log(f"Tidsperiode: {panel.date.min().strftime('%Y-%m')} – {panel.date.max().strftime('%Y-%m')}")

# ═══════════════════════════════════════════════════════════════
# LOAD CHARACTERISTICS
# ═══════════════════════════════════════════════════════════════
summary = pd.read_csv(DATA / "field_summary.csv")
facilities = pd.read_csv(RAW / "sodir_facilities.csv")
wells = pd.read_csv(RAW / "sodir_wells_dev.csv")
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")

with open(GEO) as f:
    gj = json.load(f)
op_map = {}
for feat in gj["features"]:
    p = feat["properties"]
    name = (p.get("fldName") or "").strip()
    if name:
        op_map[name] = {
            "operator": (p.get("cmpLongName") or "").strip(),
            "main_area": (p.get("fldMainArea") or "").strip(),
            "discovery_year": p.get("fldDiscoveryYear"),
            "status": (p.get("fldCurrentActivitySatus") or "").strip(),
            "hc_type": (p.get("fldHcType") or "").strip(),
        }

# Facility classification
def classify_facility(fac_type):
    if pd.isna(fac_type): return "Unknown"
    fac_type = str(fac_type).lower()
    if "fpso" in fac_type or "fpu" in fac_type: return "FPSO"
    if "subsea" in fac_type: return "Subsea tieback"
    if "semi" in fac_type or "tlp" in fac_type: return "Semi-sub"
    if "fixed" in fac_type or "concrete" in fac_type or "jacket" in fac_type or "steel" in fac_type: return "Fixed"
    if "ship" in fac_type or "vessel" in fac_type: return "FPSO"
    return "Other"

if "fclKind" in facilities.columns:
    facilities["fac_class"] = facilities["fclKind"].apply(classify_facility)
else:
    facilities["fac_class"] = "Unknown"

priority = {"FPSO": 4, "Fixed": 3, "Semi-sub": 2, "Subsea tieback": 1, "Other": 0, "Unknown": 0}
def field_facility(field_facs):
    field_facs = field_facs.copy()
    field_facs["prio"] = field_facs["fac_class"].map(priority).fillna(0)
    best = field_facs.sort_values("prio", ascending=False).iloc[0]
    wd = pd.to_numeric(field_facs.get("fclWaterDepth"), errors="coerce").median() if "fclWaterDepth" in field_facs.columns else np.nan
    return pd.Series({"facility_type": best["fac_class"], "water_depth": wd})

field_col = "fclBelongsToName" if "fclBelongsToName" in facilities.columns else facilities.columns[0]
fac_summary = facilities.groupby(field_col).apply(field_facility).reset_index()
fac_summary.columns = ["field", "facility_type", "water_depth"]

# Wells
well_field_col = "wlbField" if "wlbField" in wells.columns else wells.columns[14]
wells_summary = wells.groupby(well_field_col).size().reset_index(name="n_wells_total")
wells_summary.columns = ["field", "n_wells_total"]

# Reserves (recoverable oil)
res_summary = pd.DataFrame(columns=["field", "recoverable_oil_msm3"])
if "fldName" in reserves.columns and "fldRecoverableOil" in reserves.columns:
    if "DatesyncNPD" in reserves.columns:
        reserves = reserves.sort_values("DatesyncNPD")
    res_summary = reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]].rename(
        columns={"fldName": "field", "fldRecoverableOil": "recoverable_oil_msm3"}
    )

# ═══════════════════════════════════════════════════════════════
# PHASE DETECTION
# ═══════════════════════════════════════════════════════════════

def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def detect_phases(grp):
    grp = grp.sort_values("date").reset_index(drop=True)
    grp["prod_norm"] = grp.oil_pct_peak / 100.0
    grp["prod_smooth"] = grp.prod_norm.rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()

    out = {
        "field": grp.field.iloc[0],
        "n_months_total": len(grp),
        "first_year": int(grp.date.dt.year.min()),
        "last_year": int(grp.date.dt.year.max()),
        "peak_oil_msm3": float(grp.oil_msm3.max()),
        "total_oil_msm3": float(grp.oil_msm3.sum()),
    }

    peak_idx = grp.prod_smooth.idxmax()
    peak_smooth_val = grp.prod_smooth.iloc[peak_idx]
    out["peak_date"] = grp.date.iloc[peak_idx]
    out["peak_month_idx"] = int(peak_idx)

    # Plateau
    above_thresh = grp.prod_smooth > (PEAK_THRESHOLD * peak_smooth_val)
    if above_thresh.sum() < MIN_PLATEAU_MONTHS:
        out["has_plateau"] = False
        out["plateau_length_months"] = 0
        plateau_start = plateau_end = peak_idx
    else:
        plateau_start = peak_idx
        while plateau_start > 0 and above_thresh.iloc[plateau_start - 1]:
            plateau_start -= 1
        plateau_end = peak_idx
        while plateau_end < len(grp) - 1 and above_thresh.iloc[plateau_end + 1]:
            plateau_end += 1
        out["has_plateau"] = True
        out["plateau_length_months"] = int(plateau_end - plateau_start + 1)
    out["plateau_start_idx"] = int(plateau_start)
    out["plateau_end_idx"] = int(plateau_end)

    # First oil
    first_oil_series = grp[grp.prod_norm > 0.05].index
    first_oil_idx = int(first_oil_series.min()) if len(first_oil_series) > 0 else 0
    out["first_oil_idx"] = first_oil_idx

    ramp_length = plateau_start - first_oil_idx
    out["ramp_length_months"] = int(ramp_length)
    out["plateau_start_date"] = grp.date.iloc[int(plateau_start)]
    out["plateau_end_date"] = grp.date.iloc[int(plateau_end)]

    # Logistic ramp fit
    out["ramp_k"] = np.nan
    out["ramp_t0"] = np.nan
    out["ramp_r2"] = np.nan
    if ramp_length >= MIN_RAMP_OBS_FOR_FIT:
        ramp_data = grp.iloc[first_oil_idx:plateau_start + 1]
        t = np.arange(len(ramp_data))
        y = ramp_data.prod_smooth.values
        try:
            popt, _ = curve_fit(logistic, t, y, p0=[y.max(), 0.3, len(t)/2],
                               bounds=([0, 0.01, 0], [2, 5, len(t)*2]),
                               maxfev=2000)
            y_pred = logistic(t, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            out["ramp_k"] = float(popt[1])
            out["ramp_t0"] = float(popt[2])
            out["ramp_r2"] = float(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        except Exception:
            pass

    # Plateau metrics
    if out["has_plateau"]:
        plat = grp.iloc[plateau_start:plateau_end + 1]
        out["plateau_avg_pct"] = float(plat.prod_norm.mean())
        out["plateau_std"] = float(plat.prod_norm.std())
        out["plateau_cv"] = float(out["plateau_std"] / out["plateau_avg_pct"]) if out["plateau_avg_pct"] > 0 else np.nan
    else:
        out["plateau_avg_pct"] = np.nan
        out["plateau_std"] = np.nan
        out["plateau_cv"] = np.nan

    # Decline
    if plateau_end < len(grp) - 12:
        decline_data = grp.iloc[plateau_end:].copy()
        decline_data = decline_data[decline_data.prod_norm > 0.01]
        if len(decline_data) >= 12:
            t = np.arange(len(decline_data))
            d12_data = decline_data.iloc[:12]
            slope_12 = st.linregress(np.arange(12), np.log(d12_data.prod_norm.values))[0]
            out["D_12"] = float(-slope_12 * 12)
            slope_full = st.linregress(t, np.log(decline_data.prod_norm.values))[0]
            out["D_decline_fit"] = float(-slope_full * 12)
            out["decline_length_months"] = len(decline_data)
        else:
            out["D_12"] = np.nan
            out["D_decline_fit"] = np.nan
            out["decline_length_months"] = len(decline_data)
    else:
        out["D_12"] = np.nan
        out["D_decline_fit"] = np.nan
        out["decline_length_months"] = 0

    return out

# ═══════════════════════════════════════════════════════════════
# PROCESS ALL FIELDS
# ═══════════════════════════════════════════════════════════════
log("\nAnalyserer faser for alle felt...")

results = []
for field, grp in panel.groupby("field"):
    if len(grp) < MIN_MONTHS:
        continue
    try:
        result = detect_phases(grp)
        results.append(result)
    except Exception as e:
        log(f"  Failed: {field} — {e}")

lib = pd.DataFrame(results)
log(f"Behandlet: {len(lib)} felt")

# ═══════════════════════════════════════════════════════════════
# MERGE CHARACTERISTICS
# ═══════════════════════════════════════════════════════════════
lib = lib.merge(summary[["field", "api_gravity", "D_annual"]], on="field", how="left")
lib["operator"] = lib.field.map(lambda f: op_map.get(f, {}).get("operator", ""))
lib["main_area"] = lib.field.map(lambda f: op_map.get(f, {}).get("main_area", ""))
lib["discovery_year"] = lib.field.map(lambda f: op_map.get(f, {}).get("discovery_year"))
lib["status"] = lib.field.map(lambda f: op_map.get(f, {}).get("status", ""))
lib["hc_type"] = lib.field.map(lambda f: op_map.get(f, {}).get("hc_type", ""))
lib = lib.merge(fac_summary, on="field", how="left")
lib = lib.merge(wells_summary, on="field", how="left")
if "recoverable_oil_msm3" in res_summary.columns:
    lib = lib.merge(res_summary, on="field", how="left")

# Mark fields as shut-in or producing
lib["is_shut_in"] = lib["status"].str.contains("Shut", case=False, na=False)
lib["is_producing"] = lib["status"].str.contains("Producing", case=False, na=False)

# Archetype classification
def archetype(row):
    fac = row.get("facility_type")
    ramp = row.get("ramp_length_months", np.nan)
    plat = row.get("plateau_length_months", 0)
    if pd.isna(ramp): return "Unknown"
    if fac == "FPSO" and ramp < 12: return "Fast FPSO"
    if fac == "Subsea tieback" and ramp < 12: return "Subsea satellite"
    if plat > 60: return "Long-plateau giant"
    if ramp > 60: return "Slow ramp mega"
    if ramp < 12 and plat < 12: return "Quick-peak small"
    return "Standard development"

lib["archetype"] = lib.apply(archetype, axis=1)

# ═══════════════════════════════════════════════════════════════
# SUMMARY STATS
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("BIBLIOTEK-SAMMENDRAG")
log("═" * 80)

n_prod = lib.is_producing.sum()
n_shut = lib.is_shut_in.sum()
n_other = len(lib) - n_prod - n_shut
log(f"\n── Status-fordeling ──")
log(f"  Producing:     {n_prod}")
log(f"  Shut in:       {n_shut}")
log(f"  Annet/ukjent:  {n_other}")

log(f"\n── Fase-fordeling ──")
log(f"  Felt med platå:         {lib.has_plateau.sum()}/{len(lib)}")
log(f"  Felt med decline-data:  {lib.D_decline_fit.notna().sum()}/{len(lib)}")
log(f"  Felt med ramp-fit:      {lib.ramp_r2.notna().sum()}/{len(lib)}")

log(f"\n── Ramp-up (alle felt) ──")
log(f"  Median:    {lib.ramp_length_months.median():>5.0f} mnd  ({lib.ramp_length_months.median()/12:.1f} år)")
log(f"  25-75 pct: {lib.ramp_length_months.quantile(0.25):>5.0f} – {lib.ramp_length_months.quantile(0.75):.0f} mnd")
log(f"  Min–maks:  {lib.ramp_length_months.min():>5.0f} – {lib.ramp_length_months.max():.0f} mnd")

log(f"\n── Platå-lengde (felt med platå) ──")
plat_df = lib[lib.has_plateau]
log(f"  Median:    {plat_df.plateau_length_months.median():>5.0f} mnd  ({plat_df.plateau_length_months.median()/12:.1f} år)")
log(f"  25-75 pct: {plat_df.plateau_length_months.quantile(0.25):>5.0f} – {plat_df.plateau_length_months.quantile(0.75):.0f} mnd")

log(f"\n── Decline (etter platå) ──")
dec_df = lib[lib.D_decline_fit.notna()]
log(f"  Median D:  {dec_df.D_decline_fit.median():.3f}")
log(f"  25-75 pct: {dec_df.D_decline_fit.quantile(0.25):.3f} – {dec_df.D_decline_fit.quantile(0.75):.3f}")

log(f"\n── Arketyper ──")
for arch, cnt in lib.archetype.value_counts().items():
    sub = lib[lib.archetype == arch]
    log(f"  {arch:25s}  n={cnt:>3d}  median ramp={sub.ramp_length_months.median():>4.0f} mnd  "
        f"platå={sub.plateau_length_months.median():>4.0f} mnd")

log(f"\n── Etter facility type ──")
for fac in lib.facility_type.dropna().unique():
    sub = lib[lib.facility_type == fac]
    log(f"  {fac:18s}  n={len(sub):>3d}  ramp={sub.ramp_length_months.median():>4.0f} mnd  "
        f"platå={sub.plateau_length_months.median():>4.0f} mnd")

log(f"\n── Decade-effekt (når feltet kom online) ──")
lib["decade"] = (lib.first_year // 10 * 10).astype(int)
for dec in sorted(lib.decade.unique()):
    sub = lib[lib.decade == dec]
    log(f"  {dec}s:  n={len(sub):>3d}  ramp={sub.ramp_length_months.median():>4.0f} mnd  "
        f"platå={sub.plateau_length_months.median():>4.0f} mnd")

# ═══════════════════════════════════════════════════════════════
# SAVE LIBRARY
# ═══════════════════════════════════════════════════════════════
out_cols = [
    "field", "operator", "main_area", "facility_type", "archetype", "status",
    "discovery_year", "first_year", "last_year",
    "api_gravity", "water_depth", "n_wells_total",
    "peak_oil_msm3", "total_oil_msm3", "peak_date",
    "ramp_length_months", "ramp_k", "ramp_t0", "ramp_r2",
    "plateau_length_months", "plateau_avg_pct", "plateau_std", "plateau_cv", "has_plateau",
    "decline_length_months", "D_12", "D_decline_fit", "D_annual",
    "is_shut_in", "is_producing",
]
out_cols = [c for c in out_cols if c in lib.columns]
if "recoverable_oil_msm3" in lib.columns:
    out_cols.insert(10, "recoverable_oil_msm3")

lib_out = lib[out_cols].copy()
csv_path = DATA / "typecurve_library.csv"
lib_out.to_csv(csv_path, index=False)
log(f"\nSaved: {csv_path}")
log(f"  ({len(lib_out)} felt × {len(out_cols)} kolonner)")

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 15))
fig.suptitle(f"NCS Type-Curve Bibliotek: {len(lib)} felt (alle Sodir-felt, inkl. nedstengte)",
             fontsize=15, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Panel 1: Ramp vs size by facility
ax = fig.add_subplot(gs[0, 0])
fac_colors = {"FPSO": "#FF9800", "Subsea tieback": "#9C27B0", "Fixed": "#1565C0",
              "Semi-sub": "#2E7D32", "Other": "#757575", "Unknown": "#BDBDBD"}
for fac in ["FPSO", "Subsea tieback", "Fixed", "Semi-sub"]:
    sub = lib[lib.facility_type == fac]
    if len(sub) > 2:
        ax.scatter(sub.peak_oil_msm3, sub.ramp_length_months,
                  c=fac_colors[fac], s=50, alpha=0.7, label=f"{fac} (n={len(sub)})",
                  edgecolors="white", lw=0.5)
ax.set_xscale("log")
ax.set_xlabel("Peak produksjon (MSm³/mnd)")
ax.set_ylabel("Ramp-up lengde (mnd)")
ax.set_title("Ramp-tid vs feltstørrelse", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3)

# Panel 2: Plateau vs size
ax = fig.add_subplot(gs[0, 1])
for fac in ["FPSO", "Subsea tieback", "Fixed", "Semi-sub"]:
    sub = lib[(lib.facility_type == fac) & lib.has_plateau]
    if len(sub) > 2:
        ax.scatter(sub.peak_oil_msm3, sub.plateau_length_months,
                  c=fac_colors[fac], s=50, alpha=0.7, label=f"{fac} (n={len(sub)})",
                  edgecolors="white", lw=0.5)
ax.set_xscale("log")
ax.set_xlabel("Peak produksjon (MSm³/mnd)")
ax.set_ylabel("Platå-lengde (mnd)")
ax.set_title("Platå-lengde vs feltstørrelse", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3)

# Panel 3: Decline vs API (with shut-in marking)
ax = fig.add_subplot(gs[0, 2])
dec_data = lib.dropna(subset=["D_decline_fit", "api_gravity"])
for shut_in, marker, label in [(False, "o", "Producing"), (True, "x", "Shut in")]:
    sub = dec_data[dec_data.is_shut_in == shut_in]
    if len(sub) > 0:
        sc = ax.scatter(sub.api_gravity, sub.D_decline_fit,
                       c=sub.peak_oil_msm3.apply(np.log10), s=50, alpha=0.75,
                       cmap="viridis", marker=marker, label=label,
                       edgecolors="white", lw=0.4)
plt.colorbar(sc, ax=ax, label="log10(peak prod)")
ax.set_xlabel("API gravity")
ax.set_ylabel("D (post-platå)")
ax.set_title("Decline vs oljekvalitet", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 4-9: Type-curves grouped by archetype
top_archs = lib.archetype.value_counts().head(6).index.tolist()
archetype_colors = {"Fast FPSO": "#FF9800", "Subsea satellite": "#9C27B0",
                    "Long-plateau giant": "#1565C0", "Slow ramp mega": "#2E7D32",
                    "Quick-peak small": "#E91E63", "Standard development": "#757575"}

for i, arch in enumerate(top_archs[:6]):
    row, col = (i // 3) + 1, i % 3
    ax = fig.add_subplot(gs[row, col])

    arch_fields = lib[lib.archetype == arch].field.tolist()
    for fname in arch_fields:
        fd = panel[panel.field == fname].sort_values("date").copy()
        fd["prod_norm"] = fd.oil_pct_peak / 100.0
        fd["prod_smooth"] = fd.prod_norm.rolling(6, center=True, min_periods=1).mean()
        first_oil_idx = fd[fd.prod_norm > 0.05].index
        if len(first_oil_idx) == 0:
            continue
        first_oil = first_oil_idx.min()
        fd_plot = fd.loc[first_oil:].copy()
        t_yr = np.arange(len(fd_plot)) / 12
        ax.plot(t_yr, fd_plot.prod_smooth, color=archetype_colors.get(arch, "gray"),
                alpha=0.35, lw=0.8)

    # Median curve
    all_curves = []
    max_len = 360  # 30 years
    for fname in arch_fields:
        fd = panel[panel.field == fname].sort_values("date")
        fd_norm = fd.oil_pct_peak.values / 100.0
        first_oil_arr = np.where(fd_norm > 0.05)[0]
        if len(first_oil_arr) == 0:
            continue
        first = first_oil_arr[0]
        curve = fd_norm[first:first + max_len]
        if len(curve) < max_len:
            curve = np.concatenate([curve, np.full(max_len - len(curve), np.nan)])
        all_curves.append(curve)
    if all_curves:
        median = np.nanmedian(np.vstack(all_curves), axis=0)
        valid = ~np.isnan(median)
        t_med = np.arange(max_len)[valid] / 12
        ax.plot(t_med, median[valid], color="black", lw=2.5, label="Median")

    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 25)
    ax.set_xlabel("År siden første olje")
    ax.set_ylabel("Produksjon / peak")
    ax.set_title(f"{arch} (n={len(arch_fields)})", fontsize=11, fontweight="bold",
                color=archetype_colors.get(arch, "gray"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.axhline(1, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(0.85, color="red", ls=":", lw=0.5, alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(RESULTS / "fig_typecurve_library.png", dpi=160, bbox_inches="tight")
log(f"Saved: fig_typecurve_library.png")

# ═══════════════════════════════════════════════════════════════
# AKER BP ANALOG-PREVIEW
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("AKER BP-FELT (potensielle analoger)")
log("═" * 80)

aker = lib[lib.operator.str.contains("Aker BP", na=False)].sort_values("ramp_length_months")
log(f"\n{'Field':18s} {'Arketype':22s} {'Facility':15s} {'API':>5s} {'Ramp':>5s} {'Platå':>6s} {'D':>6s}")
log("─" * 90)
for _, r in aker.iterrows():
    ramp = f"{r.ramp_length_months:.0f}" if not pd.isna(r.ramp_length_months) else "—"
    plat = f"{r.plateau_length_months:.0f}" if not pd.isna(r.plateau_length_months) else "—"
    d = f"{r.D_decline_fit:.3f}" if not pd.isna(r.D_decline_fit) else "—"
    api = f"{r.api_gravity:.1f}" if not pd.isna(r.api_gravity) else "—"
    log(f"{r.field:18s} {str(r.archetype):22s} {str(r.facility_type):15s} "
        f"{api:>5s} {ramp:>5s} {plat:>6s} {d:>6s}")

log(f"\n── Klassiske historiske felt (gode analoger) ──")
historical = lib[(lib.is_shut_in) | (lib.first_year < 1995)].sort_values("first_year")
log(f"\n{'Field':18s} {'Operator':25s} {'Years':12s} {'Ramp':>5s} {'Platå':>6s} {'D':>6s}")
log("─" * 90)
for _, r in historical.head(20).iterrows():
    yrs = f"{int(r.first_year)}-{int(r.last_year)}"
    op = str(r.operator)[:25]
    ramp = f"{r.ramp_length_months:.0f}" if not pd.isna(r.ramp_length_months) else "—"
    plat = f"{r.plateau_length_months:.0f}" if not pd.isna(r.plateau_length_months) else "—"
    d = f"{r.D_decline_fit:.3f}" if not pd.isna(r.D_decline_fit) else "—"
    log(f"{r.field:18s} {op:25s} {yrs:12s} {ramp:>5s} {plat:>6s} {d:>6s}")

with open(RESULTS / "typecurve_library.txt", "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved: typecurve_library.txt")
