"""
Analyze Sodir wellbore exploration data for fluid-related signals per NCS field.

Produces field_exploration_summary.csv with:
  - n_exploration_wells
  - content distribution (OIL, GAS, OIL/GAS, etc.)
  - discovery year range
  - formations with HC (unique)
  - heterogeneous_fluids flag (multiple discovery wells with different content)
"""
import pandas as pd
from pathlib import Path

SRC = Path("/Users/emmastrandenskaar/Documents/Claude/Projects/Oljepris/data/raw/sodir/sodir_wellbore_exploration_all.csv")
OUT = Path("/Users/emmastrandenskaar/Documents/Claude/Projects/Oljepris/analyses/decline_quality/data/fluid_enrichment/field_exploration_summary.csv")

df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)

# Normalize string columns
for c in ["wlbField", "wlbContent", "wlbDiscoveryWellbore",
          "wlbFormationWithHc1", "wlbFormationWithHc2", "wlbFormationWithHc3"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Keep only rows with a field assignment
df = df[df["wlbField"].notna() & (df["wlbField"] != "") & (df["wlbField"].str.upper() != "NAN")].copy()

print(f"Total exploration wellbores with field: {len(df)}")
print(f"Unique fields: {df['wlbField'].nunique()}")
print(f"\nGlobal wlbContent distribution:")
print(df["wlbContent"].value_counts(dropna=False))
print(f"\nGlobal wlbDiscoveryWellbore distribution:")
print(df["wlbDiscoveryWellbore"].value_counts(dropna=False).head(10))

# Helper: gather unique non-empty formations across the three HC columns
def collect_formations(sub):
    vals = set()
    for c in ["wlbFormationWithHc1", "wlbFormationWithHc2", "wlbFormationWithHc3"]:
        if c in sub.columns:
            for v in sub[c].dropna().astype(str):
                v2 = v.strip()
                if v2 and v2.lower() != "nan":
                    vals.add(v2)
    return sorted(vals)

# Discovery wellbore mask
df["is_discovery"] = df["wlbDiscoveryWellbore"].str.upper() == "YES"

rows = []
for field, sub in df.groupby("wlbField"):
    n_wells = len(sub)
    content_counts = sub["wlbContent"].value_counts(dropna=False).to_dict()
    content_summary = "; ".join(f"{k}:{v}" for k, v in content_counts.items())

    # Discovery year range (use entry year of discovery well if present, else min entry year)
    disc = sub[sub["is_discovery"]]
    years = pd.to_numeric(sub["wlbEntryYear"], errors="coerce").dropna()
    year_min = int(years.min()) if len(years) else None
    year_max = int(years.max()) if len(years) else None

    # Discovery well contents (only the ones flagged as discovery wellbores)
    disc_contents = sorted(set(disc["wlbContent"].dropna().astype(str).str.strip()))
    disc_contents = [c for c in disc_contents if c and c.lower() != "nan"]

    # Heterogeneous fluids:
    #  - More than one distinct content type among DISCOVERY wells (i.e. distinct discoveries having different fluids), OR
    #  - among all exploration wells with non-trivial counts: presence of both OIL-bearing and GAS-bearing contents
    all_contents = sorted(set(sub["wlbContent"].dropna().astype(str).str.strip()) - {"", "nan", "NaN"})
    oil_like = {c for c in all_contents if "OIL" in c.upper()}
    gas_like = {c for c in all_contents if "GAS" in c.upper()}
    has_oil = bool(oil_like)
    has_gas = bool(gas_like)

    heterogeneous_discovery = len(disc_contents) > 1
    heterogeneous_any = (has_oil and has_gas) or len(all_contents) > 1

    formations = collect_formations(sub)

    rows.append({
        "field": field,
        "n_exploration_wells": n_wells,
        "n_discovery_wells": int(sub["is_discovery"].sum()),
        "content_distribution": content_summary,
        "discovery_well_contents": "; ".join(disc_contents),
        "all_contents": "; ".join(all_contents),
        "has_oil_content": has_oil,
        "has_gas_content": has_gas,
        "heterogeneous_discovery_fluids": heterogeneous_discovery,
        "heterogeneous_any_fluids": heterogeneous_any,
        "year_first_exploration": year_min,
        "year_last_exploration": year_max,
        "formations_with_hc": "; ".join(formations),
        "n_formations_with_hc": len(formations),
    })

summary = pd.DataFrame(rows).sort_values("n_exploration_wells", ascending=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT, index=False)
print(f"\nWrote {OUT}")
print(f"Fields summarized: {len(summary)}")

# Reporting stats
print("\n=== Distribution of fields by number of exploration wells ===")
bins = [0, 1, 2, 3, 5, 10, 20, 50, 1000]
labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51+"]
summary["bin"] = pd.cut(summary["n_exploration_wells"], bins=bins, labels=labels, right=True)
print(summary["bin"].value_counts().sort_index())

print("\n=== Heterogeneity flags ===")
print(f"Fields with multiple distinct DISCOVERY-well contents: {int(summary['heterogeneous_discovery_fluids'].sum())}")
print(f"Fields with both OIL- and GAS-bearing exploration wells: "
      f"{int((summary['has_oil_content'] & summary['has_gas_content']).sum())}")
print(f"Fields with >1 distinct content type among ALL exploration wells: {int(summary['heterogeneous_any_fluids'].sum())}")

print("\nTop 15 fields by exploration well count:")
print(summary[["field", "n_exploration_wells", "n_discovery_wells",
               "content_distribution", "heterogeneous_discovery_fluids"]].head(15).to_string(index=False))

print("\nFields flagged heterogeneous (discovery-level), top 20 by well count:")
het = summary[summary["heterogeneous_discovery_fluids"]].head(20)
print(het[["field", "n_exploration_wells", "n_discovery_wells",
           "discovery_well_contents", "all_contents"]].to_string(index=False))
