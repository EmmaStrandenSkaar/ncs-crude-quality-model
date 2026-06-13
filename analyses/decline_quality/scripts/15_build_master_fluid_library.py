"""
Script 15: Build Master Fluid Library
═══════════════════════════════════════════════════════════════════════════

Integrer alle datakilder til én master-database med reservoar-API per felt:

  Prioritet (høyest → lavest):
    1. Direct operator assay (high confidence from research)
    2. Sodir DST wellbore measurements (n_samples > 5)
    3. Sodir DST (n_samples 1-5)
    4. Operator research (medium confidence)
    5. Web research / aggregator data
    6. Existing blend assignment (fallback)

Output:
  - data/master_fluid_library.csv  — én rad per felt, alle parametere
  - results/fluid_enrichment_summary.txt
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
ENRICH = DATA / "fluid_enrichment"

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

log("═" * 80)
log("MASTER FLUID LIBRARY — INTEGRASJON")
log("═" * 80)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD ALL SOURCES
# ═══════════════════════════════════════════════════════════════

# Source 1: Existing field_summary (baseline)
existing = pd.read_csv(DATA / "field_summary.csv")
log(f"\n[1] Existing field_summary:    {len(existing)} fields")
log(f"    Direct assays: {existing.is_direct_assay.sum()}")
log(f"    Inherited blends: {(~existing.is_direct_assay).sum()}")

# Source 2: DST-derived from Sodir wellbore data
dst = pd.read_csv(ENRICH / "dst_derived_fluid.csv")
log(f"\n[2] DST-derived fluid:        {len(dst)} fields")
log(f"    With ≥5 samples:  {(dst.n_dst_samples >= 5).sum()}")
log(f"    With temperature: {dst.temp_count.gt(0).sum()}")
log(f"    With GOR:         {dst.gor_count.gt(0).sum()}")

# Source 3-7: Operator research (parse JSONs)
operator_data = {}
for fp in sorted(ENRICH.glob("operator_*.json")):
    try:
        with open(fp) as f:
            data = json.load(f)
        op_name = fp.stem.replace("operator_", "").replace("_research", "")
        fields_data = data.get("fields_with_data", data.get("fields", []))
        if isinstance(fields_data, list):
            for fd in fields_data:
                if isinstance(fd, dict) and fd.get("field") and fd.get("api_gravity"):
                    field_key = fd["field"].upper().strip()
                    operator_data.setdefault(field_key, []).append({
                        "field": field_key,
                        "operator_source": op_name,
                        "api_gravity": fd.get("api_gravity"),
                        "api_confidence": fd.get("api_confidence", "medium"),
                        "api_source": fd.get("api_source", op_name),
                        "reservoir_temp_c": fd.get("reservoir_temp_c"),
                        "reservoir_pressure_bar": fd.get("reservoir_pressure_bar"),
                        "gor": fd.get("gor"),
                        "formation": fd.get("formation", ""),
                        "reservoir_depth_m": fd.get("reservoir_depth_m"),
                        "ooip_msm3": fd.get("ooip_msm3"),
                        "recovery_factor": fd.get("recovery_factor"),
                        "notes": fd.get("notes", ""),
                        "sources": fd.get("sources", []),
                    })
    except Exception as e:
        log(f"   Warning: couldn't parse {fp.name}: {e}")

log(f"\n[3] Operator research:        {len(operator_data)} unique fields covered")
n_high = sum(1 for f, recs in operator_data.items() if any(r.get("api_confidence") == "high" for r in recs))
log(f"    With high confidence:     {n_high}")

# ═══════════════════════════════════════════════════════════════
# 2. BUILD MASTER LIBRARY WITH PRIORITY MERGE
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("MERGE LOGIC: Priority-based integration")
log("═" * 80)

def select_best_api(field):
    """Select best API value for a field using priority order."""
    field_upper = field.upper().strip()

    # Priority 1: High-confidence operator research
    if field_upper in operator_data:
        high_conf = [r for r in operator_data[field_upper] if r.get("api_confidence") == "high"]
        if high_conf:
            r = high_conf[0]
            return {
                "api_gravity": r["api_gravity"],
                "api_source": f"operator:{r['operator_source']}",
                "api_quality_tier": "operator_direct",
                "api_confidence": "high",
                "reservoir_temp_c": r.get("reservoir_temp_c"),
                "reservoir_pressure_bar": r.get("reservoir_pressure_bar"),
                "gor": r.get("gor"),
                "formation": r.get("formation"),
                "reservoir_depth_m": r.get("reservoir_depth_m"),
                "ooip_msm3": r.get("ooip_msm3"),
                "recovery_factor": r.get("recovery_factor"),
                "notes": r.get("notes"),
            }

    # Priority 2: Sodir DST with ≥5 samples (reliable reservoir measurement)
    dst_row = dst[dst.field == field_upper]
    if not dst_row.empty:
        r = dst_row.iloc[0]
        if r.n_dst_samples >= 5:
            # Use temp if available, else NaN
            temp = r.temp_mean if r.temp_count > 0 else None
            gor = r.gor_mean if r.gor_count > 0 else None
            return {
                "api_gravity": r.api_median,
                "api_source": f"sodir_dst:n={int(r.n_dst_samples)}",
                "api_quality_tier": "dst_robust",
                "api_confidence": "high" if r.n_dst_samples >= 10 else "medium",
                "reservoir_temp_c": temp,
                "reservoir_pressure_bar": None,
                "gor": gor,
                "formation": None,
                "reservoir_depth_m": r.depth_mean,
                "ooip_msm3": None,
                "recovery_factor": None,
                "notes": f"DST median across {int(r.n_dst_samples)} wells, std={r.api_std:.1f}°",
            }

    # Priority 3: Medium-confidence operator research
    if field_upper in operator_data:
        med_conf = [r for r in operator_data[field_upper] if r.get("api_confidence") == "medium"]
        if med_conf:
            r = med_conf[0]
            return {
                "api_gravity": r["api_gravity"],
                "api_source": f"operator:{r['operator_source']}",
                "api_quality_tier": "operator_medium",
                "api_confidence": "medium",
                "reservoir_temp_c": r.get("reservoir_temp_c"),
                "reservoir_pressure_bar": r.get("reservoir_pressure_bar"),
                "gor": r.get("gor"),
                "formation": r.get("formation"),
                "reservoir_depth_m": r.get("reservoir_depth_m"),
                "ooip_msm3": r.get("ooip_msm3"),
                "recovery_factor": r.get("recovery_factor"),
                "notes": r.get("notes"),
            }

    # Priority 4: Sodir DST with 1-4 samples
    if not dst_row.empty:
        r = dst_row.iloc[0]
        if r.n_dst_samples >= 1:
            return {
                "api_gravity": r.api_median,
                "api_source": f"sodir_dst:n={int(r.n_dst_samples)}",
                "api_quality_tier": "dst_limited",
                "api_confidence": "low",
                "reservoir_temp_c": r.temp_mean if r.temp_count > 0 else None,
                "reservoir_pressure_bar": None,
                "gor": r.gor_mean if r.gor_count > 0 else None,
                "formation": None,
                "reservoir_depth_m": r.depth_mean,
                "ooip_msm3": None,
                "recovery_factor": None,
                "notes": f"DST with only {int(r.n_dst_samples)} sample(s)",
            }

    # Priority 5: Low-confidence operator research
    if field_upper in operator_data:
        low_conf = [r for r in operator_data[field_upper] if r.get("api_confidence") == "low"]
        if low_conf:
            r = low_conf[0]
            return {
                "api_gravity": r["api_gravity"],
                "api_source": f"operator:{r['operator_source']}",
                "api_quality_tier": "operator_low",
                "api_confidence": "low",
                "reservoir_temp_c": r.get("reservoir_temp_c"),
                "reservoir_pressure_bar": r.get("reservoir_pressure_bar"),
                "gor": r.get("gor"),
                "formation": r.get("formation"),
                "reservoir_depth_m": r.get("reservoir_depth_m"),
                "ooip_msm3": r.get("ooip_msm3"),
                "recovery_factor": r.get("recovery_factor"),
                "notes": r.get("notes"),
            }

    # Priority 6: Existing blend assignment (fallback)
    ex_row = existing[existing.field == field_upper]
    if not ex_row.empty and pd.notna(ex_row.iloc[0].api_gravity):
        ex = ex_row.iloc[0]
        return {
            "api_gravity": ex.api_gravity,
            "api_source": f"blend:{ex.grade}",
            "api_quality_tier": "blend_direct" if ex.is_direct_assay else "blend_inherited",
            "api_confidence": "low",
            "reservoir_temp_c": None,
            "reservoir_pressure_bar": None,
            "gor": None,
            "formation": None,
            "reservoir_depth_m": None,
            "ooip_msm3": None,
            "recovery_factor": None,
            "notes": f"Inherited from {ex.grade} blend",
        }

    return None

# Get full field list from all sources
all_fields = set(existing.field.dropna().unique()) | set(dst.field.dropna().unique()) | set(operator_data.keys())
log(f"\nTotal unique fields to process: {len(all_fields)}")

# Build master library
records = []
for field in sorted(all_fields):
    result = select_best_api(field)
    if result is None:
        continue
    result["field"] = field
    records.append(result)

master = pd.DataFrame(records)
log(f"Master library built: {len(master)} fields")

# Stats by quality tier
log(f"\n── Quality tier distribution ──")
for tier, count in master.api_quality_tier.value_counts().items():
    log(f"  {tier:25s}  n={count}")

# Validate API ranges
master = master[(master.api_gravity >= 10) & (master.api_gravity <= 70)].copy()
log(f"\nAfter range validation: {len(master)} fields")

# ═══════════════════════════════════════════════════════════════
# 3. COMPARISON WITH OLD
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("ENDRINGER FRA GAMMEL MODELL")
log("═" * 80)

comparison = master.merge(existing[["field", "api_gravity"]].rename(columns={"api_gravity": "api_old"}), on="field", how="left")
comparison["api_diff"] = comparison["api_gravity"] - comparison["api_old"]

big_changes = comparison.dropna(subset=["api_old"]).copy()
big_changes = big_changes.reindex(big_changes.api_diff.abs().sort_values(ascending=False).index)

log(f"\n{'Field':22s} {'Old':>7s} {'New':>7s} {'Diff':>7s} {'Tier':25s} {'Confidence':>10s}")
log("─" * 100)
for _, r in big_changes.head(30).iterrows():
    diff_marker = "⬆" if r.api_diff > 0 else "⬇"
    log(f"{r.field:22s} {r.api_old:7.1f} {r.api_gravity:7.1f} {r.api_diff:+7.1f} {diff_marker} "
        f"{r.api_quality_tier:25s} {r.api_confidence:>10s}")

n_changed = (big_changes.api_diff.abs() > 1).sum()
n_big_changed = (big_changes.api_diff.abs() > 3).sum()
log(f"\nFelt med endringer > 1°: {n_changed}")
log(f"Felt med endringer > 3°: {n_big_changed}")
log(f"Gjennomsnittlig absolutt endring: {big_changes.api_diff.abs().mean():.1f}°")

# ═══════════════════════════════════════════════════════════════
# 4. SAVE
# ═══════════════════════════════════════════════════════════════
out_cols = ["field", "api_gravity", "api_source", "api_quality_tier", "api_confidence",
            "reservoir_temp_c", "reservoir_pressure_bar", "gor", "formation",
            "reservoir_depth_m", "ooip_msm3", "recovery_factor", "notes"]
master[out_cols].to_csv(DATA / "master_fluid_library.csv", index=False)
log(f"\nSaved: {DATA / 'master_fluid_library.csv'}")
log(f"  → {len(master)} felt med beste tilgjengelige reservoar-API")

with open(RESULTS / "fluid_enrichment_summary.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved: {RESULTS / 'fluid_enrichment_summary.txt'}")
