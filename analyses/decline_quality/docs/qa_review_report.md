# QA Review Report — V5 Decline Rate Model & Master Fluid Library

**Scope:** End-to-end audit of the V5 hybrid decline-rate model (CV R² = 0.702, N = 49) and the 110-field master fluid library that feeds it.

---

## 1. Executive Summary

**Overall health: YELLOW — with one RED-flagged structural concern.**

The pipeline is reproducible, the Beggs-Robinson viscosity implementation is mathematically sound, the LOO-CV code is correctly written, and every headline number cited in `build_doc_v2.py` matches script 18's actual output to the rounded digit. Statistical diagnostics show no multicollinearity (VIFs all <1.6) and no heteroscedasticity (BP p = 0.157).

However, the adversarial audit surfaced a **showstopper-grade circularity**: the `premium` regressor is constructed from the same residual it is then regressed against. This inflates the headline CV R² and is not cured by LOO-CV as currently implemented. Until that is resolved, the 0.702 number should not be quoted as honest out-of-sample performance.

Secondary concerns: residuals are non-normal (Shapiro-Wilk p = 0.0001), JOTUN exerts Cook's D = 0.72 (one field dominating the fit), the master fluid library contains four duplicate-field rows, three gas-field GOR values exceed the 10,000 ceiling, and the Aker BP 83% hit-rate has a ±17pp Wilson CI that should be reported.

---

## 2. Data Quality Findings — Master Fluid Library

### 2.1 Confirmed working
- **110 fields built**, tier counts reproduce exactly: 17 `operator_direct`, 31 `dst_robust`, 38 `dst_limited`, 13 `operator_low`, 6 `operator_medium`, 5 `blend_inherited`.
- API values for major fields (Ekofisk, Statfjord, Gullfaks, Oseberg, Heidrun, Johan Sverdrup, Alvheim, Skarv, Valhall, Draugen) match industry consensus ranges.
- `dst_robust` threshold (n ≥ 5) correctly enforced; `dst_limited` correctly carries n < 5.
- Documented operator/DST disagreements (Skarv 43.3 vs 39.0; Grane 27.1 vs 18.9) are intentional and explained in notes.

### 2.2 Issues found

**HIGH severity**
| # | Field(s) | Issue | Action |
|---|---|---|---|
| H1 | TROLL | API = 38.8 logged as `operator_direct`. Troll Blend oil rim is widely reported 28–32. Likely a condensate-blend value contaminating the oil-rim record. | Re-tier as `blend_inherited` or split into oil-rim vs condensate. |
| H2 | KVITEBJORN/KVITEBJØRN, ASGARD/ÅSGARD, JOHAN SVERDRUP (×2), KRAFLA/MUNIN | Four sets of duplicate rows for the same physical field with different API values. | Deduplicate; pick one tier per field. |
| H3 | DUVA (15592), SNØHVIT (15454), ORMEN LANGE (11000) | GOR values exceed the 10,000 Sm³/Sm³ ceiling — these are gas/condensate fields where GOR is not meaningful for the oil model. | Exclude from oil-decline scope or flag explicitly. |
| H4 | DST tier — reference temperature for API conversion | API = 141.5/SG − 131.5 is only valid at 60°F. If Sodir wellbore densities are reported at reservoir T, every DST-tier API is biased high by 2–8°. | Verify Sodir density reference T; apply ASTM D1250 VCF if needed. |
| H5 | MARIA, KNARR, ALVE, VALE, TRESTAKK | Single-DST `dst_limited` entries that override blend fallback. Knarr DST mean 47.9 vs public assay ~38–40 — likely condensate/gas-cap samples misclassified. | Require n ≥ 3 AND std < 5 before accepting DST; flag fields where DST − blend > 10. |
| H6 | ØRN | `reservoir_depth_m` missing despite 667 bar pressure recorded. | Backfill or drop. |

**MEDIUM severity**
- GJØA (API 59.4, std 11.1) and SIGYN (59.5) sit at the 60° ceiling with heterogeneous blends — keep but mark uncertain.
- FENRIS HPHT combination (175°C, 1000 bar, 5000 m) is at upper plausibility; documented but flag.
- "Operator-direct" tier is partly Aker BP marketing material (Edvard Grieg, Skarv, Hod, Ivar Aasen, Valhall) — not independently audited. Consider renaming `operator_self_reported` to be honest about provenance.

---

## 3. Model Audit Findings

### 3.1 Confirmed working
- **Beggs-Robinson** formula present and identical across scripts 16, 17, 18; manual recomputation (API 35, T 90°C → 1.89 cP) matches.
- **Temperature handling**: correct °C→°F conversion (`T_c · 9/5 + 32`); default = 194°F (90°C); script 17 adds a guard that falls back to default when T_c ≤ 30°C (consistent, slightly stricter).
- **LOO-CV** code is correctly structured: `LeaveOneOut().split(X)` correctly excludes test sample, predictions collected before computing Q² (predicted-R² formulation), uses full-y mean for SS_tot.
- **Reproducibility**: script 17 regenerates V5 CV R² = 0.702 and Aker BP RMSE = 0.0614 exactly. Script 18 outputs match `build_doc_v2.py` cited numbers across all 20+ headline statistics and coefficients (intercept +0.0939, ln(visc) +0.0114, premium −0.0611, |premium| +0.0401, t-stats and p-values all match to rounding).

### 3.2 Issues found

**HIGH severity**
- **M1 — Premium window off-by-one.** `mask = t >= t.max() - 12` is inclusive on both ends, selecting **13 months** instead of 12. Effect is modest (premium averaged over 13 obs, slightly biased toward earlier post-peak months) but the spec says 12. Fix: `t > t.max() - 12`.
- **M2 — Non-normal residuals.** Shapiro-Wilk W = 0.871, p = 0.0001. SE-based CIs and parametric p-values (e.g. the t = −11.67 on premium) are unreliable. Driven by EDVARD GRIEG (+0.184 residual). Report bootstrap CIs alongside parametric ones.
- **M3 — JOTUN dominates the fit.** Cook's D = 0.72 (threshold 4/n = 0.082). EDVARD GRIEG D = 0.23 and KNARR D = 0.13 also high-leverage. The fit is being steered by 1–3 fields out of 49.
- **M4 — Failure mode on bad input.** When `api_gravity` is corrupted to non-numeric strings, script 17 crashes deep in pandas with a cryptic `TypeError`. Should validate dtypes at load and fail with a clear error.

**MEDIUM severity**
- The `len(grp) < 12` pre-filter (12 valid post-peak months total) is stricter than the spec's 6-in-window threshold. Document or relax.
- Field-specific T only applies to 10 of 49 fields; the rest use the 90°C default. The "V5 = field-specific T" label oversells what changed from V4.

---

## 4. Adversarial Findings

### 4.1 Most serious concerns

**A1 — RED FLAG: Premium is built from the residual it predicts (circularity).**
Lines 136–149 of script 17 compute `log_premium = ln(actual) − ln(exp(−D_phys/12 · t))` where `D_phys` is the fitted physics-baseline prediction of D itself. Premium is then a regressor for D. **This is regressing D on a scaled version of its own residual.** The β = −0.0611, t = −11.67, p < 0.001 result is largely a mathematical identity, not a discovery. The honest physics-only R² is near zero (Appendix A reports R² = −0.015).

**A2 — LOO-CV does not cure the leakage.** `D_phys_map` is fit on all 49 fields BEFORE the LOO loop. The premium for every held-out fold is computed from a baseline that already saw that field. CV R² = 0.702 is **not genuinely out-of-sample**.

**A3 — V5 vs V3 selection is statistically indistinguishable.** Δ CV R² = 0.011 (V3 = 0.713, V5 = 0.702) is well inside noise for N = 49 LOO-CV (SE ≈ 0.04–0.06). The Aker BP hit-rate jump (75% → 83%) is **one field flipping** in a sample of 12 — Wilson 95% CI is roughly [55%, 95%]. V5 selection is plausibly selection-on-the-test-set dressed as principle.

**A4 — Effective N is smaller than 49.** The Alvheim cluster (Alvheim, Vilje, Volund, Bøyla, Skogul) shares blend and is production-correlated; effective N for Aker BP is closer to 6–8, not 12.

### 4.2 Mitigations
- **For A1/A2:** Refit `D_phys_map` AND recompute premium **inside each LOO fold** (nested CV). Report honest physics-only R² and the genuine premium-augmented R² side by side.
- Alternatively, build premium from a strictly-disjoint held-out time window (e.g. first 24 post-peak months for D_annual fit, months 25–36 for premium calculation).
- **For A3:** Report SE on CV R² via bootstrap; do a Diebold-Mariano test V3 vs V5; report Wilson CIs on hit-rate.
- **For A4:** Cluster-robust CV (leave-cluster-out instead of leave-field-out) for correlated assets.

---

## 5. Reproducibility Status

**GREEN.** Pipeline reproduces end-to-end:
- Script 15 → 110-field `master_fluid_library.csv` regenerates with documented tier counts.
- Script 17 → V5 CV R² = 0.702, Aker BP RMSE = 0.0614 reproduce exactly.
- Script 18 → `predictions_v5_final.csv` plus all coefficients, t-stats, and p-values cited in `build_doc_v2.py` match to the rounded digit.

One robustness gap: scripts crash ungracefully on corrupted API input (M4 above). Not a reproducibility blocker for clean data, but a fragility concern for production use.

---

## 6. Recommended Actions Before Proceeding to ER Use

### MUST-FIX (blocking honest reporting of CV R² = 0.702)
1. **Resolve premium-on-residual circularity (A1, A2).** Implement nested CV: refit physics baseline AND recompute premium inside each LOO fold. Re-report CV R². Expect it to drop materially — possibly into the 0.3–0.5 range.
2. **Deduplicate the master fluid library (H2).** Kvitebjørn, Åsgard, Johan Sverdrup, Krafla/Munin pairs must be resolved to one row each.
3. **Fix the TROLL API value (H1).** 38.8 is wrong for Troll oil rim.
4. **Exclude or flag gas-field GOR outliers (H3).** Duva, Snøhvit, Ormen Lange should not be carried as oil fields.
5. **Off-by-one in premium window (M1).** One-character fix; do it and re-run.

### SHOULD-FIX (required for defensible pitch claims)
6. Report bootstrap CIs alongside parametric t-stats and p-values (M2).
7. Report Wilson 95% CI on Aker BP hit-rate (currently 83% [55%, 95%], N = 12). The bare "83%" is misleading.
8. Re-run with JOTUN excluded as sensitivity (M3). If results move materially, disclose.
9. Verify Sodir DST reference temperature; apply VCF correction if not at 60°F (H4).
10. Tighten DST acceptance criterion to n ≥ 3 AND std < 5; flag DST−blend gaps > 10° (H5).
11. Rename `operator_direct` to `operator_self_reported` where the source is operator marketing.

### NICE-TO-HAVE
12. Add input-validation layer (dtypes, ranges) at CSV load to fail gracefully (M4).
13. Statistical test V3 vs V5 (Diebold-Mariano or bootstrap on Δ CV R²); if indistinguishable, default to V3.
14. Cluster-robust CV for the Alvheim group (A4).
15. Backfill ØRN reservoir depth (H6).

---

## 7. Verdict — Is the Model Ready for ER Use?

**NOT YET.** The model is internally consistent, reproducible, and well-coded — but the headline performance claim (CV R² = 0.702) rests on a circular construction of the premium regressor that LOO-CV does not cure. Quoting that number to ER without disclosure would be misleading.

**Path to GREEN is short:**
- Fix items 1–5 above (estimate 1–2 days of work).
- Re-run V5 with honest nested CV. Whatever the new CV R² is, report it with bootstrap CI.
- If post-fix CV R² remains materially above the physics-only baseline (R² ≈ 0), the framework is genuinely informative and can go to ER with appropriate caveats on N = 49, residual non-normality, and the JOTUN/EDVARD GRIEG leverage.
- If it collapses to near-zero, the honest finding is "API/viscosity does not explain NCS decline-rate variation in this sample" — which is itself a publishable result for ER, just a different story.

The 110-field master fluid library, once deduplicated and with TROLL fixed, is a genuine asset that can stand on its own merits regardless of how the regression story resolves.
