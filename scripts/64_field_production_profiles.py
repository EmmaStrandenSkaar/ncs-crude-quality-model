"""
Script 64 — Felt-produksjonsprofiler for NCS-kartet
====================================================================
Precomputer per-felt produksjonsprofiler som script 49 rendrer i popups:

  PRODUSERENDE felt (har historikk):
    · hist     — faktisk produksjon (% av peak) fra førsteolje
    · pred     — modell-predikert decline-kurve exp(-D·t) fra peak
    · D_pred   — V5.1 modell-predikert decline rate
    · D_actual — observert decline rate
    · stage    — pre-peak / plateau / decline (hvor i livssyklusen feltet er nå)

  FORWARD felt (ikke i produksjon ennå):
    · forecast — full lifecycle-kurve (ramp + platå + decline) fra V2-modellen
    · D_pred   — predikert decline rate (decline-fasen)
    · peak/ramp/plateau parametere

Output: data/processed/64_field_production_profiles.json
"""

import json, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DQ = ROOT / "analyses" / "decline_quality" / "data"
PROC = ROOT / "data" / "processed"
OUT = PROC / "64_field_production_profiles.json"

# ── lifecycle V2 (for forward-felt) ──
spec = importlib.util.spec_from_file_location(
    "lc2", str(ROOT / "analyses" / "decline_quality" / "scripts" / "26_lifecycle_v2_integration.py"))
lc2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc2)

print("=" * 60); print("SCRIPT 64: Felt-produksjonsprofiler"); print("=" * 60)

panel = pd.read_csv(DQ / "panel_monthly.csv", parse_dates=["date"])
preds = pd.read_csv(DQ / "predictions_v51.csv")
D_pred_map = preds.set_index(preds.field.str.upper())["D_pred"].to_dict()
D_act_map = preds.set_index(preds.field.str.upper())["D_annual"].to_dict()

def downsample(xs, ys, step=3):
    """Behold hver `step`-te måned for kompakt JSON."""
    return [[round(float(x), 2), round(float(y), 1)] for x, y in zip(xs[::step], ys[::step])]

profiles = {}

# ── PRODUSERENDE FELT ──
print("\n[1] Produserende felt (historikk + predikert decline)...")
n_prod = 0
for field, g in panel.groupby("field"):
    fu = field.upper().strip()
    g = g.sort_values("date")
    g = g[g.oil_pct_peak > 0]
    if len(g) < 12:
        continue
    # tid siden førsteolje (år)
    t0 = g.months_since_start.min()
    t_years = (g.months_since_start - t0) / 12.0
    pct = g.oil_pct_peak.clip(upper=130).values  # cap visuell outlier
    # peak-punkt (der oil_pct_peak ≈ 100, dvs months_since_peak == 0)
    peak_row = g[g.months_since_peak == 0]
    if len(peak_row) == 0:
        peak_t = t_years.iloc[g.oil_pct_peak.values.argmax()]
    else:
        peak_t = float((peak_row.months_since_start.iloc[0] - t0) / 12.0)

    D_pred = D_pred_map.get(fu)
    D_act = D_act_map.get(fu)

    # predikert decline-kurve fra peak (kun hvis vi har D_pred)
    pred_curve = []
    if D_pred is not None and D_pred > 0:
        t_max = float(t_years.max())
        tp = np.linspace(peak_t, t_max + 2, 40)   # litt forbi siste data
        yp = 100.0 * np.exp(-D_pred * (tp - peak_t))
        pred_curve = [[round(float(x), 2), round(float(y), 1)] for x, y in zip(tp, yp)]

    # livssyklus-stadium: hvor er feltet nå?
    last_pct = float(g.oil_pct_peak.iloc[-1])
    months_post = float(g.months_since_peak.iloc[-1])
    if months_post <= 0:
        stage = "pre-peak / ramp"
    elif last_pct > 85 and months_post < 36:
        stage = "platå"
    else:
        stage = "decline"

    profiles[fu] = {
        "type": "producing",
        "stage": stage,
        "D_pred": round(float(D_pred), 4) if D_pred is not None else None,
        "D_actual": round(float(D_act), 4) if D_act is not None else None,
        "peak_t": round(peak_t, 2),
        "hist": downsample(t_years.values, pct, step=3),
        "pred": pred_curve,
    }
    n_prod += 1
print(f"  {n_prod} produserende felt med profil")

# ── FORWARD FELT (lifecycle V2-forecast) ──
print("\n[2] Forward-felt (lifecycle V2-forecast)...")
# Kuraterte ex-ante input for forward-felt (PDO/CMD-estimater)
FORWARD_INPUTS = {
    "HUGIN":   dict(recoverable_msm3=8,  n_wells_planned=8,  facility_type="Subsea tieback", operator="Aker BP ASA", first_oil=2027),
    "MUNIN":   dict(recoverable_msm3=7,  n_wells_planned=10, facility_type="Subsea tieback", operator="Aker BP ASA", first_oil=2027),
    "FULLA":   dict(recoverable_msm3=11, n_wells_planned=12, facility_type="Subsea tieback", operator="Aker BP ASA", first_oil=2028),
    "SYMRA":   dict(recoverable_msm3=7,  n_wells_planned=8,  facility_type="Subsea tieback", operator="Aker BP ASA", first_oil=2029),
    "TYRVING": dict(recoverable_msm3=5,  n_wells_planned=6,  facility_type="Subsea tieback", operator="Aker BP ASA", first_oil=2025),
    "WISTING": dict(recoverable_msm3=80, n_wells_planned=30, facility_type="FPSO",           operator="Equinor Energy AS", first_oil=2028),
    "DVALIN":  dict(recoverable_msm3=6,  n_wells_planned=6,  facility_type="Subsea tieback", operator="Equinor Energy AS", first_oil=2027),
}
n_fwd = 0
for field, inp in FORWARD_INPUTS.items():
    try:
        r = lc2.predict_lifecycle_v2(
            {k: inp[k] for k in ["recoverable_msm3", "n_wells_planned", "facility_type", "operator"]},
            n_samples=1500)
    except Exception as e:
        print(f"  {field}: forecast feilet ({e})")
        continue
    ps = r["param_stats"]
    # P50-kurve, normalisert til peak (%) — vis ~20 år
    t_months = r["t_months"]
    p50 = r["p50"]
    peak = max(p50.max(), 1e-9)
    horizon = min(len(p50), 240)
    tf = (t_months[:horizon]) / 12.0
    yf = p50[:horizon] / peak * 100.0
    forecast = [[round(float(x), 2), round(float(y), 1)] for x, y in zip(tf[::3], yf[::3])]
    profiles[field] = {
        "type": "forward",
        "stage": "forward (pre-produksjon)",
        "first_oil": inp["first_oil"],
        "D_pred": round(float(ps["dec"]["p50"]), 4),
        "peak_p50_msm3": round(float(ps["peak"]["p50"]), 4),
        "ramp_p50": round(float(ps["ramp"]["p50"]), 1),
        "plateau_p50": round(float(ps["plat"]["p50"]), 1),
        "forecast": forecast,
    }
    n_fwd += 1
    print(f"  {field:10s} → peak {ps['peak']['p50']:.2f} MSm³/mnd, ramp {ps['ramp']['p50']:.0f}mo, "
          f"plat {ps['plat']['p50']:.0f}mo, D {ps['dec']['p50']:.3f}")
print(f"  {n_fwd} forward-felt med forecast")

with open(OUT, "w") as f:
    json.dump(profiles, f, separators=(",", ":"))
print(f"\nLagret: {OUT.name} ({OUT.stat().st_size/1024:.0f} KB, {len(profiles)} felt)")
