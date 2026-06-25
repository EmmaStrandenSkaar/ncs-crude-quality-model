"""
Script 23: Yggdrasil Full Lifecycle Forecast
═══════════════════════════════════════════════════════════════════════════

Anvender den integrerte modellen (Script 22) på Aker BPs Yggdrasil-prosjekt.

Forecaster:
  1. Hub-nivå (aggregert Yggdrasil-system)
  2. Komponenter (Krafla, Fulla, Frøy, Munin) hvor data finnes
  3. Med P10/P50/P90 usikkerhetsbånd
  4. Sammenligning med operatør-guidance (120 kboe/d)
  5. NPV-relevante metrikker (cumulative, rate-over-tid)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Import Script 22 as module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "lifecycle", str(Path(__file__).resolve().parent / "22_lifecycle_integration.py"))
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

log("═" * 80)
log("SCRIPT 23: YGGDRASIL FULL LIFECYCLE FORECAST")
log("═" * 80)

# ═══════════════════════════════════════════════════════════════
# YGGDRASIL INPUTS (fra Aker BP CMD og offentlig informasjon)
# ═══════════════════════════════════════════════════════════════
# Source: Aker BP Capital Markets Day 2023, PDO documents

# Hub-nivå (samlet Yggdrasil-system):
#   - Gross recoverable: ~650 mboe ≈ 100 MSm³ olje-ekvivalenter
#   - Olje-andel ~70%, så ~70 MSm³ olje (rest er gass og kondensat)
#   - Peak gross: ~120 kboe/d ≈ 2.0 MSm³/mnd (alle hydrokarboner)
#   - Olje peak: ~85 kboe/d ≈ 0.4 MSm³/mnd
#   - First oil: 2027
#   - Facility: Krafla FPSO/platform
#   - Aker BP operator share: 47.6%

yggdrasil_hub = {
    "name": "Yggdrasil (samlet)",
    "recoverable_msm3": 70,           # Olje-andel av 650 mboe
    "n_wells_planned": 70,            # Aker BP CMD
    "facility_type": "FPSO",
    "api_gravity": 37,                # NOAKA blend estimate
    "operator": "Aker BP ASA",
    "water_depth_m": 120,
    "decade": 2020,
    "first_oil_year": 2027,
}

# Komponenter (basert på Aker BP investor materials og Sodir oppdagelsesdata)
components = {
    "Krafla": {
        "name": "Krafla (hovedreservoar)",
        "recoverable_msm3": 25,
        "n_wells_planned": 25,
        "facility_type": "FPSO",
        "api_gravity": 38,
        "operator": "Aker BP ASA",
        "water_depth_m": 120,
        "decade": 2020,
        "first_oil_year": 2027,
    },
    "Fulla": {
        "name": "Fulla (subsea tieback)",
        "recoverable_msm3": 15,
        "n_wells_planned": 12,
        "facility_type": "Subsea tieback",
        "api_gravity": 36,
        "operator": "Aker BP ASA",
        "water_depth_m": 115,
        "decade": 2020,
        "first_oil_year": 2028,
    },
    "Frøy": {
        "name": "Frøy (re-utbygging)",
        "recoverable_msm3": 12,
        "n_wells_planned": 15,
        "facility_type": "FPSO",
        "api_gravity": 40,
        "operator": "Aker BP ASA",
        "water_depth_m": 119,
        "decade": 2020,
        "first_oil_year": 2027,
    },
    "Munin/Symra": {
        "name": "Munin + Symra (satellitter)",
        "recoverable_msm3": 10,
        "n_wells_planned": 10,
        "facility_type": "Subsea tieback",
        "api_gravity": 42,
        "operator": "Aker BP ASA",
        "water_depth_m": 125,
        "decade": 2020,
        "first_oil_year": 2029,
    },
    "Hugin/Fenris": {
        "name": "Hugin + Fenris (satellitter)",
        "recoverable_msm3": 8,
        "n_wells_planned": 8,
        "facility_type": "Subsea tieback",
        "api_gravity": 37,
        "operator": "Aker BP ASA",
        "water_depth_m": 130,
        "decade": 2020,
        "first_oil_year": 2029,
    },
}

# ═══════════════════════════════════════════════════════════════
# HUB-NIVÅ PROGNOSE
# ═══════════════════════════════════════════════════════════════
log("\n── HUB-NIVÅ FORECAST ──")
log(f"  Input: {yggdrasil_hub['recoverable_msm3']} MSm³ olje (av ~100 MSm³ total OE)")
log(f"         {yggdrasil_hub['n_wells_planned']} brønner planlagt, FPSO, API={yggdrasil_hub['api_gravity']}°")

hub_result = lifecycle.predict_lifecycle(yggdrasil_hub, n_samples=2000, horizon_months=360)

log(f"\n  Point estimate:")
log(f"    Peak rate:    {hub_result['point']['peak_msm3_mnd']:.3f} MSm³/mnd  "
    f"= {hub_result['point']['peak_msm3_mnd']*6.29*1000/30:.0f} bbl/d  "
    f"= {hub_result['point']['peak_msm3_mnd']*6.29*1000/30:.0f} kboe/d olje")
log(f"    Ramp:         {hub_result['point']['ramp_months']:.0f} måneder")
log(f"    Plateau:      {hub_result['point']['plateau_months']:.0f} måneder")
log(f"    Decline rate: {hub_result['point']['decline_rate']:.3f}/år")
log(f"    Cumulative P50: {hub_result['cumulative_p50_msm3']:.1f} MSm³  "
    f"(rec. check {hub_result['recovery_check_ratio']:.2f})")

# Sample percentiles
peak_p10, peak_p50, peak_p90 = np.percentile(hub_result["samples"]["peak"], [10, 50, 90])
ramp_p10, ramp_p50, ramp_p90 = np.percentile(hub_result["samples"]["ramp"], [10, 50, 90])
plat_p10, plat_p50, plat_p90 = np.percentile(hub_result["samples"]["plateau"], [10, 50, 90])
dec_p10, dec_p50, dec_p90 = np.percentile(hub_result["samples"]["decline"], [10, 50, 90])

log(f"\n  P10 / P50 / P90:")
log(f"    Peak (MSm³/mnd):   {peak_p10:.2f} / {peak_p50:.2f} / {peak_p90:.2f}")
log(f"    Peak (kboe/d):     {peak_p10*6.29*1000/30:.0f} / {peak_p50*6.29*1000/30:.0f} / {peak_p90*6.29*1000/30:.0f}")
log(f"    Ramp (mnd):        {ramp_p10:.0f} / {ramp_p50:.0f} / {ramp_p90:.0f}")
log(f"    Plateau (mnd):     {plat_p10:.0f} / {plat_p50:.0f} / {plat_p90:.0f}")
log(f"    Decline (/år):     {dec_p10:.3f} / {dec_p50:.3f} / {dec_p90:.3f}")

# Aker BP guidance comparison
log(f"\n  ── Sammenligning med Aker BP guidance ──")
log(f"    Operatør guidance peak (olje):     ~85 kboe/d = ~0.4 MSm³/mnd")
log(f"    Vår P50 peak:                       {peak_p50*6.29*1000/30:.0f} kboe/d = {peak_p50:.2f} MSm³/mnd")
log(f"    Vår P50 vs guidance:                {peak_p50/0.4*100:.0f}% av guidance")
log(f"    P10-P90 range:                      {peak_p10*6.29*1000/30:.0f}-{peak_p90*6.29*1000/30:.0f} kboe/d")

# ═══════════════════════════════════════════════════════════════
# KOMPONENT-NIVÅ
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("KOMPONENT-NIVÅ FORECAST")
log("═" * 80)

component_results = {}
for name, inputs in components.items():
    r = lifecycle.predict_lifecycle(inputs, n_samples=1000, horizon_months=360)
    component_results[name] = r
    log(f"\n  {inputs['name']:30s}  "
        f"Peak={r['point']['peak_msm3_mnd']:.3f}  "
        f"Ramp={r['point']['ramp_months']:.0f}mo  "
        f"Plat={r['point']['plateau_months']:.0f}mo  "
        f"D={r['point']['decline_rate']:.3f}")

# Sum of component curves (with staggered first oil)
log(f"\n── AGGREGERT HUB-PROGNOSE (sum av komponenter, staggered) ──")
hub_start_year = 2027
hub_horizon_months = 360
t_calendar = np.arange(hub_horizon_months)  # months since hub start

agg_p50 = np.zeros(hub_horizon_months)
agg_p10 = np.zeros(hub_horizon_months)
agg_p90 = np.zeros(hub_horizon_months)

for name, r in component_results.items():
    inputs = components[name]
    delay_months = (inputs["first_oil_year"] - hub_start_year) * 12
    # Shift the curve by delay
    p50 = r["p50"]
    p10 = r["p10"]
    p90 = r["p90"]
    if delay_months >= 0:
        end = min(hub_horizon_months, delay_months + len(p50))
        n_use = end - delay_months
        agg_p50[delay_months:end] += p50[:n_use]
        agg_p10[delay_months:end] += p10[:n_use]
        agg_p90[delay_months:end] += p90[:n_use]

log(f"  Aggregert peak (P50):  {agg_p50.max():.2f} MSm³/mnd = {agg_p50.max()*6.29*1000/30:.0f} kboe/d")
log(f"  Aggregert peak (P10):  {agg_p10.max():.2f}")
log(f"  Aggregert peak (P90):  {agg_p90.max():.2f}")
log(f"  Total cumulative P50:  {agg_p50.sum():.1f} MSm³")
log(f"  Sum av components recoverable:  {sum(c['recoverable_msm3'] for c in components.values())} MSm³")

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 13))
fig.suptitle("Yggdrasil Full Lifecycle Forecast — Hub + Komponenter med P10/P50/P90",
             fontsize=15, fontweight="bold", y=1.0)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Panel 1: Hub forecast with bands
ax = fig.add_subplot(gs[0, :2])
years = t_calendar / 12 + hub_start_year
# Convert MSm³/mnd to kboe/d for readability
to_kboed = 6.29 * 1000 / 30  # MSm³/mnd → kboe/d: ×6290 boe/MSm³ ÷ 30 d/mnd ÷ 1000 = 209.67
ax.fill_between(years, hub_result["p10"] * to_kboed, hub_result["p90"] * to_kboed,
                alpha=0.2, color="#1565C0", label="P10-P90")
ax.plot(years, hub_result["p50"] * to_kboed, color="#1565C0", lw=2, label="P50 (median)")
ax.axhline(0.085 * to_kboed, color="red", ls="--", lw=1.5, alpha=0.7,
           label="Aker BP guidance (~85 kboe/d olje)")
ax.set_xlim(2027, 2055)
ax.set_xlabel("År"); ax.set_ylabel("Olje-produksjon (kboe/d)")
ax.set_title(f"Yggdrasil hub-nivå (treat som ett system)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

# Panel 2: Aggregated components (staggered)
ax = fig.add_subplot(gs[0, 2])
ax.fill_between(years, agg_p10 * to_kboed, agg_p90 * to_kboed,
                alpha=0.2, color="#2E7D32", label="P10-P90 (sum)")
ax.plot(years, agg_p50 * to_kboed, color="#2E7D32", lw=2, label="P50 aggregert")
ax.plot(years, hub_result["p50"] * to_kboed, color="#1565C0", lw=1.5, ls="--", alpha=0.7,
        label="Hub-nivå P50 (vs)")
ax.set_xlim(2027, 2055)
ax.set_xlabel("År"); ax.set_ylabel("Olje (kboe/d)")
ax.set_title("Komponent-aggregat (staggered)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3-7: Individual component curves
comp_axes = []
for idx, (name, r) in enumerate(component_results.items()):
    row = 1 + idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    inputs = components[name]
    delay = (inputs["first_oil_year"] - hub_start_year) * 12
    comp_years = np.arange(len(r["p50"])) / 12 + inputs["first_oil_year"]
    ax.fill_between(comp_years, r["p10"] * to_kboed, r["p90"] * to_kboed,
                    alpha=0.25, color="#9C27B0")
    ax.plot(comp_years, r["p50"] * to_kboed, color="#9C27B0", lw=1.8)
    ax.set_xlim(inputs["first_oil_year"] - 1, 2050)
    ax.set_title(f"{inputs['name']}\nP50 peak: {r['point']['peak_msm3_mnd']*6.29*1000/30:.0f} kboe/d",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("År", fontsize=9); ax.set_ylabel("kboe/d", fontsize=9)
    ax.grid(alpha=0.3)

# Last panel: Summary
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
summary = f"""YGGDRASIL FORECAST OPPSUMMERING

── HUB-NIVÅ (P10/P50/P90) ──
Peak:      {peak_p10*6.29*1000/30:.0f}/{peak_p50*6.29*1000/30:.0f}/{peak_p90*6.29*1000/30:.0f} kboe/d
Ramp:      {ramp_p10:.0f}/{ramp_p50:.0f}/{ramp_p90:.0f} mnd
Platå:     {plat_p10:.0f}/{plat_p50:.0f}/{plat_p90:.0f} mnd
Decline:   {dec_p10:.3f}/{dec_p50:.3f}/{dec_p90:.3f}

── AGGREGERT KOMPONENTER ──
Sum peak:  {agg_p50.max()*6.29*1000/30:.0f} kboe/d
Recovery:  {agg_p50.sum():.0f} MSm³

── VS AKER BP CMD ──
Guidance:  85 kboe/d (olje)
Vår P50:   {peak_p50*6.29*1000/30:.0f} kboe/d
Forhold:   {peak_p50*6.29*1000/30/85*100:.0f}%

── NB ──
Vide bånd reflekterer
ekte usikkerhet i sub-
modellene (ramp R²=0.10,
plat R²=0.19, dec R²=0.29).
Brukes som sanity-check
mot operatør-guidance,
ikke som single point."""

ax.text(0.05, 0.97, summary, transform=ax.transAxes, fontsize=9,
        fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="#E8F5E9", ec="#2E7D32", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_yggdrasil_forecast.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_yggdrasil_forecast.png")

# Save raw curves for further analysis
forecast_df = pd.DataFrame({
    "year": years,
    "hub_p10_msm3_mnd": hub_result["p10"],
    "hub_p50_msm3_mnd": hub_result["p50"],
    "hub_p90_msm3_mnd": hub_result["p90"],
    "agg_p10_msm3_mnd": agg_p10,
    "agg_p50_msm3_mnd": agg_p50,
    "agg_p90_msm3_mnd": agg_p90,
})
forecast_df.to_csv(DATA / "yggdrasil_forecast.csv", index=False)
log(f"Saved: yggdrasil_forecast.csv")

with open(RESULTS / "yggdrasil_forecast.txt", "w") as f:
    f.write("\n".join(lines))
