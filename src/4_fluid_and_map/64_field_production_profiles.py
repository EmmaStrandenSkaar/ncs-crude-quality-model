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
ROOT = Path(__file__).resolve().parents[2]
DQ = ROOT / "src" / "3_decline_lifecycle" / "data"
PROC = ROOT / "data" / "processed"
OUT = PROC / "64_field_production_profiles.json"

# ── lifecycle V2 (for forward-felt) ──
spec = importlib.util.spec_from_file_location(
    "lc2", str(ROOT / "src" / "3_decline_lifecycle" / "scripts" / "26_lifecycle_v2_integration.py"))
lc2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc2)

print("=" * 60); print("SCRIPT 64: Felt-produksjonsprofiler"); print("=" * 60)

# FULL Sodir månedlig produksjon (alle 132 felt) — ikke det begrensede 52-felts panelet
sodir = pd.read_csv(ROOT / "data" / "raw" / "sodir" / "sodir_field_production_monthly.csv")
sodir["date"] = pd.to_datetime(
    sodir.prfYear.astype(str) + "-" + sodir.prfMonth.astype(str) + "-01", errors="coerce")
sodir = sodir.rename(columns={"prfInformationCarrier": "field", "prfPrdOilNetMillSm3": "oil"})
sodir = sodir.sort_values(["field", "date"])
# normaliser til peak + months_since_start/peak per felt
sodir["field_u"] = sodir.field.str.upper().str.strip()
sodir["peak_oil"] = sodir.groupby("field_u").oil.transform("max")
sodir = sodir[sodir.peak_oil > 0]
sodir["oil_pct_peak"] = sodir.oil / sodir.peak_oil * 100
sodir["mss"] = sodir.groupby("field_u").cumcount()
# months_since_peak: indeks relativt til peak-måned
peak_idx = sodir.loc[sodir.groupby("field_u").oil_pct_peak.idxmax()].set_index("field_u")["mss"].to_dict()
sodir["msp"] = sodir.mss - sodir.field_u.map(peak_idx)

preds = pd.read_csv(DQ / "predictions_v51.csv")
D_pred_map = preds.set_index(preds.field.str.upper())["D_pred"].to_dict()
D_act_map = preds.set_index(preds.field.str.upper())["D_annual"].to_dict()

# typecurve observert decline (fallback for felt utenfor V5.1-treningssettet)
tc = pd.read_csv(DQ / "typecurve_library.csv")
tc_decline_map = tc.dropna(subset=["D_decline_fit"]).set_index(
    tc.dropna(subset=["D_decline_fit"]).field.str.upper())["D_decline_fit"].to_dict()

def annual_bars(t_years, pct):
    """Aggreger månedlig produksjon til årlige søyler: [[år, snitt-%], ...]."""
    df = pd.DataFrame({"yr": np.floor(t_years).astype(int), "pct": pct})
    agg = df.groupby("yr")["pct"].mean()
    return [[int(y), round(float(v), 1)] for y, v in agg.items()]

profiles = {}

# ── PRODUSERENDE FELT (full Sodir-dekning) ──
print("\n[1] Produserende felt (full Sodir-historikk + decline-hierarki)...")
n_prod = n_v51 = n_tc = n_nodecline = 0
FORECAST_YEARS = 12
for fu, g in sodir.groupby("field_u"):
    g = g.sort_values("date")
    g = g[g.oil_pct_peak > 0]
    if len(g) < 12 or g.oil.sum() < 0.3:    # min historikk + min oljevolum (filtrer gass)
        continue
    t_years = (g.mss - g.mss.min()) / 12.0
    pct = g.oil_pct_peak.clip(upper=130).values
    peak_t = float((g.loc[g.oil_pct_peak.idxmax(), "mss"] - g.mss.min()) / 12.0)

    # decline-hierarki: V5.1 → typecurve observert
    D_pred = D_pred_map.get(fu)
    decline_src = "V5.1"
    if D_pred is None:
        D_pred = tc_decline_map.get(fu)
        decline_src = "observert (typecurve)"
    D_act = D_act_map.get(fu)

    hist_bars = annual_bars(t_years.values, pct)
    last_yr, last_pct = hist_bars[-1]

    # livssyklus-stadium
    months_post = float(g.msp.iloc[-1])
    last_actual_pct = float(g.oil_pct_peak.iloc[-1])
    if months_post <= 6:
        stage = "pre-peak / ramp"
    elif last_actual_pct > 85 and months_post < 36:
        stage = "platå"
    else:
        stage = "decline"

    fcst_bars, decline_line = [], []
    if D_pred is not None and D_pred > 0 and months_post > 6:
        for k in range(1, FORECAST_YEARS + 1):
            fcst_bars.append([int(last_yr + k), round(float(last_pct * np.exp(-D_pred * k)), 1)])
        tp = np.linspace(peak_t, last_yr + FORECAST_YEARS, 50)
        yp = 100.0 * np.exp(-D_pred * (tp - peak_t))
        decline_line = [[round(float(x), 2), round(float(y), 1)] for x, y in zip(tp, yp)]
        n_v51 += decline_src == "V5.1"; n_tc += decline_src != "V5.1"
    else:
        n_nodecline += 1
        decline_src = "ennå ikke i decline" if months_post <= 6 else "ingen decline-estimat"

    profiles[fu] = {
        "type": "producing", "stage": stage,
        "D_pred": round(float(D_pred), 4) if (D_pred and fcst_bars) else None,
        "D_actual": round(float(D_act), 4) if D_act is not None else None,
        "decline_src": decline_src,
        "peak_t": round(peak_t, 2),
        "hist_bars": hist_bars, "fcst_bars": fcst_bars, "decline_line": decline_line,
    }
    n_prod += 1
print(f"  {n_prod} produserende felt med profil")
print(f"    decline fra V5.1-modell:        {n_v51}")
print(f"    decline fra typecurve (obs.):   {n_tc}")
print(f"    historikk uten forecast (nye):  {n_nodecline}")

# ── FORWARD FELT (lifecycle V2-forecast) ──
print("\n[2] Forward-felt (lifecycle V2-forecast)...")
# Kuraterte ex-ante input for forward-felt (PDO/CMD-estimater).
# decline-fasen settes fra et navngitt analog-felt (observert NCS-decline), ikke
# fra lifecycle-V2-modellens decline-output. V2-decline traff et tak rundt 0.40
# for nesten alle subsea-tiebacks, noe som er urealistisk bratt. Vi bruker i
# stedet sammenlignbare produserende felt som analoger. Ramp/platå/peak beholdes
# fra V2-modellen; kun decline-raten overstyres med analog-raten.
#   D_analog forankret i observert decline (predictions_v51.csv):
#     Ivar Aasen ~18-22%, Vilje ~22%, Volund ~24%, Skarv ~11%, Goliat ~13%, Maria ~8%
FORWARD_INPUTS = {
    "HUGIN":   dict(recoverable_msm3=8,  n_wells_planned=8,  facility_type="Subsea tieback", operator="Aker BP ASA",       first_oil=2027, analog="Ivar Aasen",   D_analog=0.18),
    "MUNIN":   dict(recoverable_msm3=7,  n_wells_planned=10, facility_type="Subsea tieback", operator="Aker BP ASA",       first_oil=2027, analog="Ivar Aasen",   D_analog=0.18),
    "FULLA":   dict(recoverable_msm3=11, n_wells_planned=12, facility_type="Subsea tieback", operator="Aker BP ASA",       first_oil=2028, analog="Skarv (kondensat)", D_analog=0.15),
    "SYMRA":   dict(recoverable_msm3=7,  n_wells_planned=8,  facility_type="Subsea tieback", operator="Aker BP ASA",       first_oil=2029, analog="Vilje",        D_analog=0.20),
    "TYRVING": dict(recoverable_msm3=5,  n_wells_planned=6,  facility_type="Subsea tieback", operator="Aker BP ASA",       first_oil=2025, analog="Volund",       D_analog=0.24),
    "WISTING": dict(recoverable_msm3=80, n_wells_planned=30, facility_type="FPSO",           operator="Equinor Energy AS", first_oil=2028, analog="Goliat",       D_analog=0.13),
    "DVALIN":  dict(recoverable_msm3=6,  n_wells_planned=6,  facility_type="Subsea tieback", operator="Equinor Energy AS", first_oil=2027, analog="Maria",        D_analog=0.16),
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
    t_months = np.asarray(r["t_months"], dtype=float)
    p50 = np.asarray(r["p50"], dtype=float).copy()
    peak = max(float(p50.max()), 1e-9)

    # Behold V2-modellens ramp + platå, men overstyr decline-halen med analog-
    # raten (V2-decline traff et urealistisk tak ~0.40 for subsea-tiebacks).
    D = float(inp["D_analog"])
    peak_i = int(p50.argmax())
    plat_end = peak_i
    for i in range(peak_i, len(p50)):
        if p50[i] >= 0.98 * peak:
            plat_end = i
        else:
            break
    for i in range(plat_end + 1, len(p50)):
        dt_yr = (t_months[i] - t_months[plat_end]) / 12.0
        p50[i] = peak * np.exp(-D * dt_yr)

    horizon = min(len(p50), 240)   # ~20 år
    tf = (t_months[:horizon]) / 12.0
    yf = p50[:horizon] / peak * 100.0
    # årlige forecast-søyler (alle "fcst" → lys farge)
    fcst_bars = annual_bars(tf, yf)
    # stiplet linje = jevn lifecycle-kurve
    decline_line = [[round(float(x), 2), round(float(y), 1)] for x, y in zip(tf[::3], yf[::3])]
    profiles[field] = {
        "type": "forward",
        "stage": "forward (pre-produksjon)",
        "first_oil": inp["first_oil"],
        "D_pred": round(D, 4),
        "analog": inp["analog"],
        "peak_p50_msm3": round(float(ps["peak"]["p50"]), 4),
        "ramp_p50": round(float(ps["ramp"]["p50"]), 1),
        "plateau_p50": round(float(ps["plat"]["p50"]), 1),
        "fcst_bars": fcst_bars,
        "decline_line": decline_line,
    }
    n_fwd += 1
    print(f"  {field:10s} → peak {ps['peak']['p50']:.2f} MSm³/mnd, ramp {ps['ramp']['p50']:.0f}mo, "
          f"plat {ps['plat']['p50']:.0f}mo, D {D:.3f} (analog: {inp['analog']})")
print(f"  {n_fwd} forward-felt med forecast")

with open(OUT, "w") as f:
    json.dump(profiles, f, separators=(",", ":"))
print(f"\nLagret: {OUT.name} ({OUT.stat().st_size/1024:.0f} KB, {len(profiles)} felt)")
