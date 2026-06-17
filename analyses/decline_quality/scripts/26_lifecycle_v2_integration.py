"""
Script 26: Lifecycle V2 Integration
═══════════════════════════════════════════════════════════════════════════

Forbedret integrasjon med:
  - JOINT BOOTSTRAP sampling (bevarer parameter-korrelasjoner)
  - HARD CAPS på sub-modell utganger (basert på empirisk fordeling)
  - RECOVERY CONSTRAINT (kun behold samples med 0.5 < cumulative/recoverable < 1.5)
  - Triangulerte recoverable scenarier (lav/base/høy)

Modul som Script 27 kan bruke.
"""

import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"


def load_v2_models():
    """Last V2 sub-modeller med bootstrap fits."""
    with open(DATA / "lifecycle_v2_models.pkl", "rb") as f:
        return pickle.load(f)


def build_feature_vector(field_inputs, feature_cols, facility_categories, operator_categories,
                          major_operators):
    """Bygg feature-vektor for ett felt."""
    vec = {}
    vec["log_recoverable"] = np.log(field_inputs["recoverable_msm3"])
    vec["log_n_wells"] = np.log(max(field_inputs.get("n_wells_planned", 10), 1))

    # Facility dummies
    facility = field_inputs.get("facility_type", "FPSO")
    for cat in facility_categories:
        target = cat.replace("fac_", "")
        vec[cat] = 1 if facility == target else 0

    # Operator dummies (grouped — same logic as during training)
    operator = field_inputs.get("operator", "Other")
    if operator not in major_operators:
        operator = "Other"
    for cat in operator_categories:
        target = cat.replace("op_", "")
        vec[cat] = 1 if operator == target else 0

    # Pick the features the model wants
    return np.array([vec.get(f, 0.0) for f in feature_cols])


def predict_single_fit(fit, X):
    """Beregn prediksjon fra én bootstrap-fit."""
    intercept = fit["intercept"]
    coefs = np.array(fit["coefs"])
    return intercept + X @ coefs


def build_production_curve(peak, ramp_months, plateau_months, decline_rate,
                            horizon_months=360, shut_in_fraction=0.05):
    """
    Bygg månedlig produksjonskurve fra 4 parametere.
    Logistisk ramp + konstant platå + eksponentiell decline + shut-in.
    """
    t = np.arange(horizon_months)
    curve = np.zeros(horizon_months)

    # Phase 1: Logistic ramp from ~0 to peak
    if ramp_months > 1:
        k = 6 / ramp_months  # gir ~5% av peak ved t=0 og ~95% ved t=ramp
        ramp_mask = t < ramp_months
        curve[ramp_mask] = peak / (1 + np.exp(-k * (t[ramp_mask] - ramp_months / 2)))

    # Phase 2: Plateau
    plat_start = int(np.ceil(ramp_months))
    plat_end = int(np.ceil(ramp_months + plateau_months))
    if plat_end > plat_start:
        curve[plat_start:plat_end] = peak

    # Phase 3: Exponential decline
    decline_start = plat_end
    if decline_start < horizon_months:
        years_post = (t[decline_start:] - decline_start) / 12
        curve[decline_start:] = peak * np.exp(-decline_rate * years_post)

    # Shut-in
    shut_threshold = shut_in_fraction * peak
    below = curve < shut_threshold
    if below.any() and decline_start < horizon_months:
        # Only shut-in AFTER plateau
        first_below_idx = np.argmax(below[decline_start:]) + decline_start
        if first_below_idx > decline_start:
            curve[first_below_idx:] = 0

    return curve


def solve_decline_for_recovery(peak, ramp_months, plat_months, recoverable_msm3,
                                 truncation_factor=0.95):
    """
    Gitt peak, ramp og plat, finn decline-rate som rekvirerer eksakt 'recoverable'.

    Total cumulative = ramp_area + plat_area + decline_area
    ramp_area ≈ peak × ramp/2  (logistisk ≈ trekant)
    plat_area = peak × plat
    decline_area = peak × 12/D × truncation_factor  (decline til 5% av peak)

    Setter total = recoverable, løser for D:
      D = peak × 12 × truncation_factor / (R - peak × ramp/2 - peak × plat)
    """
    pre_decline = peak * (ramp_months / 2 + plat_months)
    remaining = recoverable_msm3 - pre_decline
    if remaining <= 0:
        # Ramp+plat alene rekvirerer allerede mer enn R → kort decline trengs
        return 1.0  # cap-verdi
    D = peak * 12 * truncation_factor / remaining
    return D


def predict_lifecycle_v2(field_inputs, n_samples=3000, horizon_months=360,
                          recovery_min=0.5, recovery_max=1.5, verbose=False,
                          enforce_recovery=True):
    """
    V2 lifecycle forecast med:
      - Joint bootstrap (samme indeks brukt på tvers av sub-modeller)
      - Hard caps på sub-modell utgang
      - enforce_recovery: hvis True, beregn decline analytisk for å sikre cumulative=recoverable

    Returnerer dict med P10/P50/P90 kurver og diagnostikk.
    """
    models = load_v2_models()
    facility_cats = models["facility_categories"]
    operator_cats = models["operator_categories"]
    major_ops = models["major_operators"]

    n_bootstraps = len(models["peak"]["bootstrap_fits"])

    # Build feature vectors for each model (they have different feature sets)
    feat_vecs = {}
    for name in ["peak", "ramp", "plateau", "decline"]:
        feat_vecs[name] = build_feature_vector(
            field_inputs, models[name]["features"], facility_cats, operator_cats, major_ops
        )

    # Sample n_samples bootstrap indices (with replacement)
    rng = np.random.default_rng(42)
    bootstrap_picks = rng.integers(0, n_bootstraps, size=n_samples)

    all_samples = []
    n_filtered = 0
    n_capped = {"ramp": 0, "plateau": 0, "decline": 0}

    for b_idx in bootstrap_picks:
        # Joint prediction — same bootstrap index → correlated params
        peak_log = predict_single_fit(models["peak"]["bootstrap_fits"][b_idx], feat_vecs["peak"])
        ramp_log = predict_single_fit(models["ramp"]["bootstrap_fits"][b_idx], feat_vecs["ramp"])
        plat_log = predict_single_fit(models["plateau"]["bootstrap_fits"][b_idx], feat_vecs["plateau"])
        dec_log = predict_single_fit(models["decline"]["bootstrap_fits"][b_idx], feat_vecs["decline"])

        # Apply Duan smearing to peak (korrigerer log-retransformasjons-bias)
        peak_smear = models["peak"].get("smear_factor", 1.0)
        peak = np.exp(peak_log) * peak_smear
        ramp_raw = np.exp(ramp_log) - 1
        plat_raw = np.exp(plat_log) - 1
        dec_raw = np.exp(dec_log)

        # Apply hard caps
        ramp_cap = models["ramp"]["cap"]
        plat_cap = models["plateau"]["cap"]
        dec_cap = models["decline"]["cap"]

        ramp = float(np.clip(ramp_raw, ramp_cap[0], ramp_cap[1]))
        plat = float(np.clip(plat_raw, plat_cap[0], plat_cap[1]))
        dec_bs = float(np.clip(dec_raw, dec_cap[0], dec_cap[1]))

        if ramp != ramp_raw: n_capped["ramp"] += 1
        if plat != plat_raw: n_capped["plateau"] += 1
        if dec_bs != dec_raw: n_capped["decline"] += 1

        # Optionally enforce recovery=1.0 by solving for decline
        if enforce_recovery:
            dec_implied = solve_decline_for_recovery(peak, ramp, plat,
                                                       field_inputs["recoverable_msm3"])
            # Use implied decline, but cap to plausible range
            dec = float(np.clip(dec_implied, dec_cap[0], dec_cap[1]))
        else:
            dec = dec_bs

        # Build curve
        curve = build_production_curve(peak, ramp, plat, dec, horizon_months)

        # Compute recovery
        cumulative = curve.sum()
        recovery_ratio = cumulative / field_inputs["recoverable_msm3"]

        sample = {
            "peak": peak, "ramp": ramp, "plat": plat, "dec": dec,
            "curve": curve, "cumulative": cumulative, "recovery_ratio": recovery_ratio,
        }

        # Filter by recovery constraint
        if recovery_min <= recovery_ratio <= recovery_max:
            all_samples.append(sample)
        else:
            n_filtered += 1

    if verbose:
        print(f"  Bootstrap samples drawn: {n_samples}")
        print(f"  After recovery filter ({recovery_min}-{recovery_max}): {len(all_samples)} kept")
        print(f"  Filtered out: {n_filtered} ({n_filtered/n_samples*100:.0f}%)")
        for k, v in n_capped.items():
            print(f"  Capped at boundary ({k}): {v} ({v/n_samples*100:.0f}%)")

    if len(all_samples) < 50:
        print(f"  WARNING: only {len(all_samples)} samples passed filter — results unreliable")
        # Relax constraint
        all_samples = []
        for b_idx in bootstrap_picks[:1000]:
            peak_log = predict_single_fit(models["peak"]["bootstrap_fits"][b_idx], feat_vecs["peak"])
            ramp_log = predict_single_fit(models["ramp"]["bootstrap_fits"][b_idx], feat_vecs["ramp"])
            plat_log = predict_single_fit(models["plateau"]["bootstrap_fits"][b_idx], feat_vecs["plateau"])
            dec_log = predict_single_fit(models["decline"]["bootstrap_fits"][b_idx], feat_vecs["decline"])
            peak = np.exp(peak_log) * models["peak"].get("smear_factor", 1.0)
            ramp = float(np.clip(np.exp(ramp_log) - 1, models["ramp"]["cap"][0], models["ramp"]["cap"][1]))
            plat = float(np.clip(np.exp(plat_log) - 1, models["plateau"]["cap"][0], models["plateau"]["cap"][1]))
            dec = float(np.clip(np.exp(dec_log), models["decline"]["cap"][0], models["decline"]["cap"][1]))
            curve = build_production_curve(peak, ramp, plat, dec, horizon_months)
            all_samples.append({
                "peak": peak, "ramp": ramp, "plat": plat, "dec": dec,
                "curve": curve, "cumulative": curve.sum(),
                "recovery_ratio": curve.sum() / field_inputs["recoverable_msm3"],
            })

    # Compute percentiles
    curves = np.array([s["curve"] for s in all_samples])
    p10 = np.percentile(curves, 10, axis=0)
    p50 = np.percentile(curves, 50, axis=0)
    p90 = np.percentile(curves, 90, axis=0)

    # Parameter percentiles
    param_stats = {}
    for param in ["peak", "ramp", "plat", "dec"]:
        vals = np.array([s[param] for s in all_samples])
        param_stats[param] = {
            "p10": float(np.percentile(vals, 10)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "mean": float(vals.mean()),
        }

    recoveries = np.array([s["recovery_ratio"] for s in all_samples])

    return {
        "t_months": np.arange(horizon_months),
        "p10": p10, "p50": p50, "p90": p90,
        "param_stats": param_stats,
        "n_samples": len(all_samples),
        "n_filtered": n_filtered,
        "n_capped": n_capped,
        "recoveries": recoveries,
        "field_inputs": field_inputs,
        "samples": all_samples,
    }


def predict_lifecycle_triangulated(field_inputs_base, recoverable_low, recoverable_high,
                                     n_samples=3000):
    """
    Kjør modellen for 3 recoverable-scenarier (lav/base/høy).
    Returnerer dict med 3 fulle resultater.
    """
    results = {}
    for scenario, R in [("low", recoverable_low),
                        ("base", field_inputs_base["recoverable_msm3"]),
                        ("high", recoverable_high)]:
        inputs = {**field_inputs_base, "recoverable_msm3": R}
        results[scenario] = predict_lifecycle_v2(inputs, n_samples=n_samples)
    return results


if __name__ == "__main__":
    # Quick test
    test_inputs = {
        "recoverable_msm3": 50,  # Justert ned fra 70
        "n_wells_planned": 70,
        "facility_type": "FPSO",
        "operator": "Aker BP ASA",
    }

    print("\n── V2 TEST: Yggdrasil-lignende felt med R=50 MSm³ ──\n")
    result = predict_lifecycle_v2(test_inputs, n_samples=3000, verbose=True)
    print(f"\n  Resultater (P10/P50/P90):")
    for k, v in result["param_stats"].items():
        print(f"    {k:8s}  {v['p10']:.3f} / {v['p50']:.3f} / {v['p90']:.3f}")
    print(f"\n  Peak (kboe/d): {result['param_stats']['peak']['p10']*209.67:.0f} / "
          f"{result['param_stats']['peak']['p50']*209.67:.0f} / "
          f"{result['param_stats']['peak']['p90']*209.67:.0f}")
    print(f"  Recovery (P10-P90): {np.percentile(result['recoveries'], 10):.2f} - "
          f"{np.percentile(result['recoveries'], 90):.2f}")

    print("\n── TRIANGULERING (R=30, 50, 70) ──")
    triangle = predict_lifecycle_triangulated(test_inputs, 30, 70, n_samples=1500)
    for scenario, r in triangle.items():
        ps = r["param_stats"]
        print(f"  {scenario.upper():5s}: peak P50 = {ps['peak']['p50']:.2f} MSm³/mnd "
              f"= {ps['peak']['p50']*209.67:.0f} kboe/d  (n={r['n_samples']})")
