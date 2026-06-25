# NCS Crude Oil Analytics Platform

**A full-lifecycle quantitative model of the Norwegian Continental Shelf — turning public data into field-level estimates of realized price, production decline, and valuation inputs that tie out to operator-reported figures at R² = 0.989.**

*Emma Strandenskaar · BI Norwegian Business School · [github.com/EmmaStrandenSkaar/ncs-crude-quality-model](https://github.com/EmmaStrandenSkaar/ncs-crude-quality-model)*

---

## Why this matters for equity research

A field-level NPV is only as good as its two hardest inputs: **what price the barrels actually fetch** and **how fast the volumes decline**. Both are usually hand-waved — analysts assume "Brent flat" and a generic 8–10% decline. This platform replaces those assumptions with data-driven, out-of-sample-validated estimates for the entire NCS, grounded entirely in public data.

Three models chain together to drive a field's revenue line from first principles:

```
Crude quality (assay)      →  Price differential vs Brent  →  Realized price
Reservoir fluid + history  →  Annual decline rate          →  Production profile
PDO variables (ex-ante)    →  Peak / ramp / plateau        →  Pre-production forecast
                                         ↓
                         Field-level revenue & NPV inputs
```

The payoff: a realized price that ties out to operator-reported figures at **R² = 0.989 / MAE 1.56 USD/bbl** (Aker BP, 24 quarters), and a decline rate that lands within ±0.05 **83% of the time** on held-out operator data.

---

## The three models

### 1 · Realized-price model — `src/2_quality_price_model/`

Predicts a crude grade's **price differential to Dated Brent** from assay quality (API gravity, sulphur, vacuum resid, CCR, middle-distillate yield, metals) interacted with the prevailing market regime.

- Two specifications: **Model A** (global, 43 grades) and **Model B** (Brent-linked, 32 grades)
- OLS with cluster-robust standard errors; **out-of-time** validation, not just in-sample
- Differential model: in-sample R² **0.49**, OOT R² **0.38**, RMSE **2.78 USD/bbl**
- Closes the identity `Realized = Brent + Σ(production share × differential)`, validated at field level against **Aker BP** reported realizations: **R² 0.989, MAE 1.56 USD/bbl** across 24 quarters

> *Business relevance:* a 2–3 USD/bbl differential error on a 100 kbbl/d field is ~$75–110m of annual revenue. Getting the grade-level differential right is the difference between a buy and a hold.

### 2 · Decline model V5.1 — `src/3_decline_lifecycle/`

Predicts a producing field's **annual decline rate**:

```
D = 0.094 + 0.011·ln(viscosity) − 0.061·P₁₂ + 0.040·|P₁₂|
```

It blends **physics** (Beggs-Robinson viscosity derived from reservoir API) with a **field-specific premium** `P₁₂` — the trailing 12-month deviation of actual production from the physics baseline, capturing reservoir and operational realities the physics alone misses.

| Metric | Value |
|---|---|
| Nested LOO cross-validated R² | **0.662** (honest, post-QA) |
| Aker BP RMSE | 0.062 |
| Aker BP hit rate (±0.05) | **83%** (Wilson 95% CI: 55–95%) |

Fed by a **108-field master fluid library** with reservoir API enriched from Sodir DST (1,186 wellbore tests) plus operator assays.

### 3 · Pre-peak lifecycle forecast — `src/3_decline_lifecycle/`, `src/4_fluid_and_map/`

For fields **not yet on stream**, forecasts the full curve (peak → ramp → plateau → decline) from **only ex-ante PDO variables** — recoverable reserves, planned wells, facility type, operator. No post-first-oil data leaks in.

| Phase | Out-of-sample accuracy | Intended use |
|---|---|---|
| Ramp duration | median error ±5 months | point estimate |
| Plateau duration | median error ±4 months | point estimate |
| Peak level | log-space CV R² 0.84; linear error ~35% | **range / triangulation tool, not a point NPV input** |
| Decline | calibrated to recoverable reserves | derived, not free |

A **joint bootstrap + recovery constraint** produces physically consistent **P10/P50/P90** bands (the area under the curve cannot exceed recoverable reserves). Case study: **Yggdrasil** (Aker BP NOAKA), modelled as a hub plus five components with triangulated scenarios.

---

## Validation discipline

This is built like research that has to survive a desk review, not a backtest that flatters itself. The standard throughout is *honest out-of-sample performance*, not in-sample fit.

- **Nested LOO and out-of-time cross-validation** — headline numbers are held-out, not in-sample.
- **A QA pass caught and removed premium circularity** in the decline model. The honest CV R² fell from 0.71 → **0.66**, and the lower number is the one reported. Self-correction over self-promotion.
- **Genuine hold-out on new fields** revealed that peak level is a **~35% range tool**, not a point estimate — and this README says so, rather than burying it.
- **Cross-basin stress test:** on UK NSTA data the viscosity coefficient *reverses sign* — so the model is explicitly scoped as **NCS-specific**, not a universal law.
- **Bootstrap CIs** on forecasts and **Wilson CIs** on every hit-rate proportion (small-*n* honesty).

| Model | Headline | Honest validation |
|---|---|---|
| Differential | in-sample R² 0.49 | OOT R² 0.38, RMSE 2.78 USD/bbl |
| Realized price (field) | — | R² 0.989, MAE 1.56 USD/bbl (24 qtrs) |
| Decline V5.1 | — | nested LOO R² 0.662; Aker BP RMSE 0.062 |
| Pre-peak peak | log CV R² 0.84 | OOS linear error ~35% (range tool) |
| Pre-peak ramp/plateau | — | OOS ±5 / ±4 months |

---

## Supporting infrastructure

- **Master fluid library** — 108 fields with reservoir API across **4 provenance tiers** (Sodir DST · operator direct assays · operator research · blended fallback), so any number traces back to a source.
- **Field → quality imputation** (Script 63) — area-median assay characteristics + Sodir-DST API gravity fill quality drivers for the **104 NCS fields** without a published assay, keeping every producing field in the price model rather than dropping it.
- **Interactive NCS map** (Script 49) — **142 NCS fields** with production bar charts (historical + forecast), price differentials, and quality drivers; the full pipeline applied across the entire lifecycle (producing / forward / discovery) in one view.
- **Supporting research** — post-earnings drift, M&A windows, and oil-service lag studies under `src/5_supporting_research/`.

---

## Repository structure

```
src/
├── 1_data_ingestion/        Data fetch — Sodir, EIA, crude assays, normpris
├── 2_quality_price_model/   Quality → differential + Aker BP realized-price validation
├── 3_decline_lifecycle/     V5.1 decline + lifecycle forecast (own scripts/data/results/docs)
├── 4_fluid_and_map/         Fluid imputation, production profiles, interactive map
└── 5_supporting_research/   Earnings, M&A, and oil-service event studies

data/
├── raw/                     Inputs as fetched from public sources
└── processed/               Model-ready panels and derived tables

docs/                        Methodology documents (.docx)
```

**Where to start by interest:**

| If you want to see… | Go to |
|---|---|
| The headline modelling and validation | `src/3_decline_lifecycle/README.md` |
| Quality-to-price economics | `src/2_quality_price_model/` |
| Everything visual, in one place | `src/4_fluid_and_map/` (Script 49, interactive map) |
| How the data is sourced | `src/1_data_ingestion/` |
| The written methodology | `docs/` and `src/3_decline_lifecycle/docs/` |

---

## Reproducibility

- **Layered pipeline** — ingestion → modelling → fluid/map, with numbered, ordered scripts in each stage so the build can be re-run end to end.
- **Versioned artifacts** — master fluid library, type-curve library, and model predictions are written as CSVs under each module's `data/` and `results/`.
- **Provenance-tagged inputs** — every fluid record carries its source tier, so any number traces back to a Sodir DST test or a named operator document.
- **Two methodology documents** (`.docx`) — decline model and realized price — in `src/3_decline_lifecycle/docs/` and `docs/`.

**Decline model, applied step by step:**

1. Obtain reservoir API gravity (Sodir DST or operator data).
2. Compute Beggs-Robinson viscosity from API and reservoir temperature.
3. Pull the last 12 months of post-peak production from Sodir.
4. Compute the premium `P₁₂` as the mean log-deviation from the physics baseline.
5. Apply `D = 0.094 + 0.011·ln(μ) − 0.061·P₁₂ + 0.040·|P₁₂|`.
6. Forecast `production_T = peak · exp(−D · T)`.

---

## Data sources (all public)

- **Sodir** (Sokkeldirektoratet / Norwegian Offshore Directorate) — production, drill-stem tests, geology
- **Petroleumsprisrådet** (Petroleum Price Board) — normpris (norm price)
- **Equinor, ExxonMobil, TotalEnergies** — crude assays
- **EIA** — market fundamentals
- **Aker BP** — quarterly reports (used as an independent validation target)

---

## Limitations & honesty

- The exponential-decline assumption breaks down for plateau fields (e.g. Valhall).
- The decline model is **NCS-specific** — the viscosity coefficient reverses sign on UK NSTA data.
- The decline model requires **≥12 months of post-peak history**; new fields fall back to the analog-based pre-peak forecast.
- Pre-peak **peak level** is a triangulation range (~35% OOS error), not a point NPV input.
- The differential model explains roughly a third of out-of-time variance — useful for systematic quality pricing, not for predicting short-term market noise.

---

*Built to answer a single question end-to-end: given what we can know publicly about an NCS field, what price will its barrels realize, how fast will they decline, and what is that worth? The emphasis throughout is on cross-validated, honestly-bounded estimates — models that are useful precisely because they declare what they can and cannot predict.*