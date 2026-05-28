# NCS Crude Oil Quality Model

Quality-adjusted crude oil price differential model for the Norwegian Continental
Shelf (NCS), with field-level decomposition and an interactive map visualisation.

## What it does

Predicts the price differential vs. Dated Brent for individual crude oil grades
based on assay quality (API gravity, sulfur, vacuum residue, Conradson carbon,
middle distillate yield, metals), regional/logistics factors, and market regime
variables (refinery utilisation, crack spreads, geopolitical events).

Validated against:
- **Aker BP** quarterly realised price reports (MAE 1.29 USD/bbl, R²=0.992)
- **Petroleum Price Council** official normpris (MAE 0.68–1.16 USD/bbl per field)

## Models

| Model | Universe | Use case |
|-------|----------|----------|
| **Modell A — Global** | 43 grades worldwide | Cross-regional M&A, global majors |
| **Modell B — Brent-linked** | 32 grades, Dated Brent pricing | NCS companies (Aker BP, Equinor, Vår) |

OLS with HC1 robust standard errors. Out-of-time validation on last 24 months.

## Repository layout

```
scripts/                       # All analysis scripts (numbered chronologically)
  ├── 34_parsimonious_model.py        # Model A (global) training
  ├── 42_akrbp_realized_price...      # AKRBP field-level decomposition
  ├── 43_akrbp_forward_prediction.py  # Forward 2026–2028 scenarios
  ├── 45b_field_comparison...         # Model vs. normpris (presentation chart)
  ├── 47_two_model_system.py          # Trains both Model A and B
  ├── 48_fetch_sodir_geodata.py       # Fetches NCS geometry from Sodir API
  └── 49_interactive_ncs_map.py       # Builds interactive HTML map

data/
  ├── raw/      # External data (gitignored — re-fetch with scripts 23/27/29/31/48)
  └── processed/  # Analysis outputs, model files, charts
```

## Reproducing

```bash
# 1. Fetch raw data
python scripts/48_fetch_sodir_geodata.py

# 2. Train models
python scripts/47_two_model_system.py

# 3. Run AKRBP analyses
python scripts/42_akrbp_realized_price_decomposition.py
python scripts/43_akrbp_forward_prediction.py

# 4. Generate presentation charts
python scripts/45b_field_comparison_presentation.py

# 5. Build interactive map
python scripts/49_interactive_ncs_map.py
# → Output: data/processed/49_ncs_interactive_map.html
```

## Data sources

- **Sodir FactMaps** — NCS field, license, and discovery geometry (REST API)
- **Petroleum Price Council** — official quarterly normpris (tax-reference prices)
- **Equinor Crude Assays** — official lab assays (API, sulfur, distillation cuts)
- **EIA, World Bank, IMF** — market controls (Brent, WTI, crack spreads, inventories)
- **Aker BP Quarterly Reports** — realised oil prices for validation

## Key methodology choices

- **Brent-linked subset**: excludes WTI-anchored grades (Cushing, Edmonton, USGC)
  to avoid contaminating the Brent differential model.
- **Blend-proxy mapping**: fields without standalone assays use the official
  assay of their actual export blend (e.g., Valhall → Ekofisk Blend).
- **Geographical proxy**: small fields without quality data borrow from the
  nearest neighbouring field with assay (same sea area, haversine distance).
- **Time-invariant variables**: features that vary only by grade (e.g., transport
  distance) use grade-clustered standard errors; effective N = number of grades.

## Author

Emma Stranden Skaar — finance/economics student.
Built as part of equity research / asset-valuation work on Norwegian
oil and gas companies.
