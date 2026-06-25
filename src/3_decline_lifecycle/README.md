# NCS Decline Rate Model

A quantitative full-lifecycle production forecasting framework for the Norwegian Continental Shelf. Two complementary engines:

1. **Decline model (V5.1)** — for fields with ≥12 months post-peak history. Physics (Beggs-Robinson viscosity) + a field-specific premium framework.
2. **Pre-peak forecast (V2)** — for new fields from PDO data alone. Predicts peak / ramp / plateau / decline using only ex-ante variables.

## Decline model (V5.1) — for producing fields

```
D_annual = 0.0938 + 0.0106·ln(μ) − 0.0612·P₁₂ + 0.0399·|P₁₂|
```

| Metric | Value |
|--------|-------|
| Nested LOO Cross-validated R² | 0.662 (honest, post-QA) |
| In-sample R² | 0.742 |
| RMSE | 0.042 |
| Aker BP RMSE (n=12) | 0.062 |
| Aker BP hit rate (±0.05) | 83% (Wilson CI: 55–95%) |

## Pre-peak forecast (V2) — for new fields from PDO

Predicts the full production curve from only pre-production variables (recoverable
reserves, planned wells, facility type, operator). Validated with genuine out-of-sample
hold-out (model never saw the test field).

| Phase | Out-of-sample accuracy | Use |
|-------|------------------------|-----|
| **Ramp duration** | median error ±5 months | point estimate |
| **Plateau duration** | median error ±4 months | point estimate |
| **Peak level** | median error ~35% | **range tool / triangulation only** |
| **Decline** | calibrated to recoverable | derived, not free |

**Honest caveat on peak:** log-space CV R²=0.84 (gets order of magnitude right), but
linear median error ~35% out-of-sample. Mega-fields (Johan Sverdrup) and tiny tie-back
fields are hardest. Peak should be used to *triangulate against operator guidance*, not
as a standalone NPV point estimate. A Duan smearing correction removes the systematic
log-retransformation bias (−17% → −11%).

Key methodological discipline: **only variables knowable before first oil are used** —
no post-hoc production data leaks into the forecast.

## Master fluid library

A reservoir-level API gravity database for 110 NCS fields, built from:

- **Sodir DST database**: 1186 wellbore tests with oil density → API for 85 fields
- **Operator direct assays** (Equinor, ExxonMobil, TotalEnergies): 17 fields with high confidence
- **Operator research** (Aker BP, Vår Energi, OKEA, ConocoPhillips): 43 fields with annual report / CMD data
- **Field-specific reservoir temperatures**: 10 fields with measured T used in Beggs-Robinson

## Repository structure

```
analyses/decline_quality/
├── scripts/
│   ├── 13_typecurve_library.py         # Ramp/plateau/decline phase library (69 fields)
│   ├── 14_analog_similarity.py         # Analog matching for new fields
│   ├── 15_build_master_fluid_library.py # Integrate all fluid sources
│   ├── 16_refit_with_enriched_data.py  # Model refit with reservoir-API
│   ├── 17_hybrid_model.py              # Selective enrichment (V3/V4/V5)
│   └── 18_final_v5_figure.py           # Final V5 production figure
├── data/
│   ├── master_fluid_library.csv        # 110 fields × 13 columns
│   ├── typecurve_library.csv           # 69 fields × 31 columns
│   ├── predictions_v5_final.csv        # V5 model predictions
│   └── panel_monthly.csv               # Monthly production data
├── docs/
│   ├── NCS_Decline_Model_Methodology.docx  # Full methodology document
│   └── build_doc_v2.py                 # Document generator
└── results/
    ├── fig_final_v5_model.png          # Main model figure
    ├── fig_hybrid_comparison.png       # V1-V5 comparison
    ├── fig_valhall_forensic.png        # Edge case (platau field)
    └── ...
```

## How to use the model

1. Get reservoir API gravity for the field (Sodir DST or operator data)
2. Compute Beggs-Robinson viscosity: `μ = 10^(x·T^(-1.163)) − 1` where `x = 10^(3.0324 − 0.02023·API)` and T in °F
3. Download last 12 months of post-peak production from Sodir
4. Compute premium: `P₁₂ = mean(ln(actual_i / exp(-D_physics/12 · month_i)))` for last 12 months
5. Apply formula above
6. Forecast: `production_year_T = peak · exp(-D · T)`

## Key methodological choices

- **12-month premium window** (validated against 3-120 month alternatives)
- **Hybrid API**: reservoir API used only when high/medium confidence; blend fallback otherwise
- **Field-specific reservoar T** used in 10 fields where measured (V5 vs V3)
- **Asymmetry correction** via `|premium|` captures U-shaped relationship

## Limitations

- Exponential decline assumption breaks down for plateau fields (e.g., Valhall)
- Model is NCS-specific (UK NSTA validation showed viscosity sign reverses)
- Requires 12 months post-peak data; new fields need analog-based approach

See `docs/NCS_Decline_Model_Methodology.docx` for full methodology.
