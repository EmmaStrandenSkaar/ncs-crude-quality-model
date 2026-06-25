"""
Script 22: Lifecycle Integration — Komplett produksjonskurve fra PDO-data
═══════════════════════════════════════════════════════════════════════════

Kombinerer 4 sub-modeller til én full prognose:
  1. Peak rate (Script 20b)
  2. Ramp duration (Script 21)
  3. Plateau duration (Script 21)
  4. Ex-ante decline rate (Script 21)

Inkluderer:
  - Logistisk ramp-up
  - Konstant platå
  - Eksponentiell decline
  - Bootstrap-propagering for P10/P50/P90-bånd
  - Sanity-check mot recoverable reserves

Eksportert som modul som Script 23 kan bruke.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).resolve().parents[1] / "data"


def load_submodels():
    """Last alle 4 sub-modeller."""
    models = {}
    for name in ["peak_forecast_simplified", "submodel_ramp", "submodel_plateau", "submodel_decline"]:
        with open(DATA / f"{name}.pkl", "rb") as f:
            artifact = pickle.load(f)
        models[name] = artifact
    return models


def build_feature_vector(field_inputs, feature_cols, facility_cats=None, operator_cats=None):
    """Bygg feature-vektor fra felt-input dict for en gitt modell."""
    vec = {}

    vec["log_recoverable"] = np.log(field_inputs["recoverable_msm3"])
    vec["log_n_wells"] = np.log(max(field_inputs.get("n_wells_planned", 10), 1))
    vec["log_water_depth"] = np.log(max(field_inputs.get("water_depth_m", 100), 1))
    vec["decade_scaled"] = (field_inputs.get("decade", 2020) - 1980) / 10
    vec["api_use"] = field_inputs.get("api_gravity", 37)

    # Facility dummies
    facility = field_inputs.get("facility_type", "FPSO")
    for cat in facility_cats or []:
        col_name = cat
        target_value = cat.replace("fac_", "")
        vec[col_name] = 1 if facility == target_value else 0

    # Operator dummies
    operator = field_inputs.get("operator", "Other")
    for cat in operator_cats or []:
        col_name = cat
        target_value = cat.replace("op_", "")
        vec[col_name] = 1 if operator == target_value else 0

    # Pick the features the model wants
    X = np.array([vec.get(f, 0) for f in feature_cols]).reshape(1, -1)
    return X


def predict_single(model_artifact, field_inputs):
    """Predikter én verdi fra en sub-modell."""
    feature_cols = model_artifact["feature_cols"]
    coefs = np.array([model_artifact["coefficients"][f] for f in feature_cols])
    intercept = model_artifact["intercept"]

    facility_cats = [f for f in feature_cols if f.startswith("fac_")]
    operator_cats = [f for f in feature_cols if f.startswith("op_")]

    X = build_feature_vector(field_inputs, feature_cols, facility_cats, operator_cats)
    pred_log = float(intercept + (X @ coefs)[0])
    return pred_log


def predict_with_uncertainty(model_artifact, field_inputs, n_samples=1000):
    """Sample fra parameter-fordeling for usikkerhet."""
    feature_cols = model_artifact["feature_cols"]
    facility_cats = [f for f in feature_cols if f.startswith("fac_")]
    operator_cats = [f for f in feature_cols if f.startswith("op_")]
    X = build_feature_vector(field_inputs, feature_cols, facility_cats, operator_cats)[0]

    intercept = model_artifact["intercept"]
    coefs = np.array([model_artifact["coefficients"][f] for f in feature_cols])
    ci_low = np.array(model_artifact["ci_low"])
    ci_high = np.array(model_artifact["ci_high"])

    # Approximate normal distribution from CI bounds
    # 95% CI ≈ ±1.96 SD, so SD ≈ (CI_high - CI_low) / 3.92
    all_means = np.concatenate([[intercept], coefs])
    all_sds = (ci_high - ci_low) / 3.92

    # Sample parameters and compute predictions
    np.random.seed(42)
    samples = []
    for _ in range(n_samples):
        sampled = np.random.normal(all_means, all_sds)
        intercept_s = sampled[0]
        coefs_s = sampled[1:]
        pred = intercept_s + X @ coefs_s
        samples.append(pred)
    return np.array(samples)


def predict_lifecycle(field_inputs, n_samples=1000, horizon_months=360):
    """
    Predikter komplett produksjonskurve med usikkerhetsbånd.

    Input (alle ex-ante, fra PDO + discovery):
      - recoverable_msm3   (PDO recoverable oil)
      - n_wells_planned    (PDO antall brønner)
      - facility_type      ("FPSO" / "Fixed" / "Subsea tieback" / "Semi-sub" / "Other")
      - api_gravity        (discovery DST)
      - operator           ("Aker BP ASA" / "Equinor Energy AS" / "Other" / etc.)
      - water_depth_m      (kjent fra lokasjon)
      - decade             (planlagt oppstart, default 2020)
      - first_oil_year     (default 2027)

    Output:
      dict med komponent-prediksjoner + samples + P10/P50/P90 produksjonskurve
    """
    models = load_submodels()
    peak_model = models["peak_forecast_simplified"]
    ramp_model = models["submodel_ramp"]
    plat_model = models["submodel_plateau"]
    dec_model = models["submodel_decline"]

    # Point estimates (medians)
    peak_log_pred = predict_single(peak_model, field_inputs)
    ramp_log_pred = predict_single(ramp_model, field_inputs)
    plat_log_pred = predict_single(plat_model, field_inputs)
    dec_log_pred = predict_single(dec_model, field_inputs)

    point = {
        "peak_msm3_mnd": float(np.exp(peak_log_pred)),
        "ramp_months": float(np.exp(ramp_log_pred) - 1),
        "plateau_months": float(np.exp(plat_log_pred) - 1),
        "decline_rate": float(np.exp(dec_log_pred)),
    }

    # Uncertainty samples
    peak_samples = np.exp(predict_with_uncertainty(peak_model, field_inputs, n_samples))
    ramp_samples = np.maximum(np.exp(predict_with_uncertainty(ramp_model, field_inputs, n_samples)) - 1, 1)
    plat_samples = np.maximum(np.exp(predict_with_uncertainty(plat_model, field_inputs, n_samples)) - 1, 0)
    dec_samples = np.exp(predict_with_uncertainty(dec_model, field_inputs, n_samples))
    # Cap decline samples in plausible range
    dec_samples = np.clip(dec_samples, 0.01, 0.8)

    # Build production curves
    t_months = np.arange(horizon_months)
    curves = np.zeros((n_samples, horizon_months))

    for i in range(n_samples):
        peak = peak_samples[i]
        ramp = ramp_samples[i]
        plat = plat_samples[i]
        D = dec_samples[i]

        for j, m in enumerate(t_months):
            if m < ramp:
                # Logistic ramp-up (S-curve)
                # f(m) = peak / (1 + exp(-k * (m - ramp/2)))
                k = 6 / ramp if ramp > 0 else 1  # steepness so we hit ~peak at m=ramp
                curves[i, j] = peak / (1 + np.exp(-k * (m - ramp / 2)))
            elif m < ramp + plat:
                curves[i, j] = peak
            else:
                years_post = (m - ramp - plat) / 12
                curves[i, j] = peak * np.exp(-D * years_post)

        # Economic shut-in: production drops to 0 when below 5% of peak
        shut_threshold = 0.05 * peak
        below = curves[i] < shut_threshold
        if below.any():
            first_below = np.argmax(below)
            # Only apply if it's after peak (not during ramp)
            if t_months[first_below] > ramp + plat:
                curves[i, first_below:] = 0

    # Percentiles
    p10 = np.percentile(curves, 10, axis=0)
    p50 = np.percentile(curves, 50, axis=0)
    p90 = np.percentile(curves, 90, axis=0)

    # Cumulative production check
    cumulative_p50 = p50.sum()  # MSm³ over horizon
    recovery_check = cumulative_p50 / field_inputs["recoverable_msm3"]

    return {
        "point": point,
        "samples": {
            "peak": peak_samples, "ramp": ramp_samples,
            "plateau": plat_samples, "decline": dec_samples,
        },
        "t_months": t_months,
        "curves": curves,
        "p10": p10, "p50": p50, "p90": p90,
        "cumulative_p50_msm3": float(cumulative_p50),
        "recoverable_msm3": float(field_inputs["recoverable_msm3"]),
        "recovery_check_ratio": float(recovery_check),
        "field_inputs": field_inputs,
    }


if __name__ == "__main__":
    # Quick test
    test_inputs = {
        "recoverable_msm3": 80,
        "n_wells_planned": 70,
        "facility_type": "FPSO",
        "api_gravity": 37,
        "operator": "Aker BP ASA",
        "water_depth_m": 120,
        "decade": 2020,
        "first_oil_year": 2027,
    }

    result = predict_lifecycle(test_inputs)
    print("Test forecast — Yggdrasil-lignende felt:")
    print(f"  Peak:       {result['point']['peak_msm3_mnd']:.3f} MSm³/mnd")
    print(f"  Ramp:       {result['point']['ramp_months']:.0f} mnd")
    print(f"  Plateau:    {result['point']['plateau_months']:.0f} mnd")
    print(f"  Decline:    {result['point']['decline_rate']:.3f}")
    print(f"  Cumulative P50: {result['cumulative_p50_msm3']:.1f} MSm³")
    print(f"  Recovery check: {result['recovery_check_ratio']:.2f} (1.0 = perfekt match)")
