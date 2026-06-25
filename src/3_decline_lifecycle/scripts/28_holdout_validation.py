"""
Script 28: Ekte Out-of-Sample Hold-out Validering
═══════════════════════════════════════════════════════════════════════════

For hvert testfelt:
  1. RE-TREN alle sub-modeller på de ANDRE feltene (ekskluder testfeltet helt)
  2. Forecast peak/ramp/platå KUN fra ex-ante input (R, wells, facility, operator)
  3. Sammenlign med faktisk realisert produksjon

Dette er genuint blindt — modellen har ALDRI sett testfeltet under trening.

Tester spesielt nylige felt der vi kjenner fasiten:
  Johan Sverdrup (2019), Edvard Grieg (2015), Martin Linge (2021),
  Goliat (2016), Ivar Aasen (2016), Gina Krog (2017), Maria (2018), m.fl.
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "sodir"

TO_KBOED = 6.29 * 1000 / 30

lines = []
def log(msg=""):
    print(msg); lines.append(msg)

log("═" * 80)
log("SCRIPT 28: EKTE OUT-OF-SAMPLE HOLD-OUT VALIDERING")
log("═" * 80)

# Load & build dataset (same as Script 25)
typecurve = pd.read_csv(DATA / "typecurve_library.csv")
master = pd.read_csv(DATA / "master_fluid_library_v51.csv")
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")
if "DatesyncNPD" in reserves.columns:
    reserves = reserves.sort_values("DatesyncNPD")
res = (reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]]
       .rename(columns={"fldName": "field", "fldRecoverableOil": "recoverable_msm3"}))
typecurve = typecurve.merge(res, on="field", how="left")
api_map = master.set_index("field")["api_gravity"].to_dict()
typecurve["api_v51"] = typecurve.field.map(api_map).fillna(typecurve.get("api_gravity"))

df = typecurve.dropna(subset=["recoverable_msm3", "facility_type", "peak_oil_msm3"]).copy()
df = df[df.recoverable_msm3 > 0.5]
df = df[df.n_wells_total > 0]

# Features
df["log_peak"] = np.log(df.peak_oil_msm3)
df["log_recoverable"] = np.log(df.recoverable_msm3)
df["log_n_wells"] = np.log(df.n_wells_total.clip(lower=1))
df["log_ramp_p1"] = np.log(df.ramp_length_months + 1)
df["log_plat_p1"] = np.log(df.plateau_length_months.clip(lower=0) + 1)

fac_d = pd.get_dummies(df.facility_type, prefix="fac", drop_first=True).astype(int)
df = pd.concat([df, fac_d], axis=1)
facility_cols = list(fac_d.columns)

log(f"\nDatasett: {len(df)} felt")

# Model feature specs (matching Script 25 simplified)
peak_features = ["log_recoverable", "log_n_wells"] + facility_cols
ramp_features = ["log_recoverable"]
plat_features = ["log_recoverable"]

# ═══════════════════════════════════════════════════════════════
# DEFINE TEST FIELDS — nylige felt der vi kjenner fasiten
# ═══════════════════════════════════════════════════════════════
test_field_names = [
    "JOHAN SVERDRUP", "EDVARD GRIEG", "MARTIN LINGE", "GOLIAT",
    "IVAR AASEN", "GINA KROG", "MARIA", "GUDRUN", "KNARR", "GJØA",
    "ALVHEIM", "SKARV", "VOLUND", "BØYLA", "YME", "FENJA", "DUVA",
]

test_fields = df[df.field.isin(test_field_names)].copy()
log(f"\nTestfelt tilgjengelig: {len(test_fields)} av {len(test_field_names)} ønskede")

# ═══════════════════════════════════════════════════════════════
# HOLD-OUT TEST: re-tren uten testfelt, predikter blind
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("HOLD-OUT RESULTATER — PEAK (viktigst)")
log("═" * 80)
log(f"\n{'Felt':18s} {'År':>5s} {'R':>6s} {'Faktisk':>9s} {'Predikert':>10s} {'%-feil':>8s} {'Status':>8s}")
log("─" * 75)

results = []
for _, test_row in test_fields.sort_values("first_year", ascending=False).iterrows():
    # Train on ALL OTHER fields
    train = df[df.field != test_row.field]

    # Peak
    Xp_tr = train[peak_features].fillna(0).values
    yp_tr = train["log_peak"].values
    lr_peak = LinearRegression().fit(Xp_tr, yp_tr)
    Xp_te = test_row[peak_features].fillna(0).values.reshape(1, -1)
    peak_pred = np.exp(lr_peak.predict(Xp_te)[0])
    peak_actual = test_row.peak_oil_msm3
    peak_err = (peak_pred - peak_actual) / peak_actual * 100

    # Ramp
    Xr_tr = train[ramp_features].fillna(0).values
    yr_tr = train["log_ramp_p1"].values
    lr_ramp = LinearRegression().fit(Xr_tr, yr_tr)
    Xr_te = test_row[ramp_features].fillna(0).values.reshape(1, -1)
    ramp_pred = np.exp(lr_ramp.predict(Xr_te)[0]) - 1
    ramp_pred = np.clip(ramp_pred, 3, 96)
    ramp_actual = test_row.ramp_length_months

    # Plateau
    Xpl_tr = train[plat_features].fillna(0).values
    ypl_tr = train["log_plat_p1"].values
    lr_plat = LinearRegression().fit(Xpl_tr, ypl_tr)
    Xpl_te = test_row[plat_features].fillna(0).values.reshape(1, -1)
    plat_pred = np.exp(lr_plat.predict(Xpl_te)[0]) - 1
    plat_pred = np.clip(plat_pred, 0, 84)
    plat_actual = test_row.plateau_length_months

    status = "✓" if abs(peak_err) < 35 else "🚩"
    results.append({
        "field": test_row.field, "year": int(test_row.first_year),
        "recoverable": test_row.recoverable_msm3,
        "peak_actual": peak_actual, "peak_pred": peak_pred, "peak_err": peak_err,
        "ramp_actual": ramp_actual, "ramp_pred": ramp_pred,
        "plat_actual": plat_actual, "plat_pred": plat_pred,
    })
    log(f"  {test_row.field:16s} {int(test_row.first_year):>5d} {test_row.recoverable_msm3:6.1f} "
        f"{peak_actual:9.3f} {peak_pred:10.3f} {peak_err:+7.0f}% {status:>8s}")

res = pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════
# SAMMENDRAG-STATISTIKK
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("SAMMENDRAG")
log("═" * 80)

peak_mape = res.peak_err.abs().median()
peak_bias = res.peak_err.median()
within_25 = (res.peak_err.abs() < 25).mean() * 100
within_35 = (res.peak_err.abs() < 35).mean() * 100

log(f"\n  PEAK:")
log(f"    Median absolutt %-feil:  {peak_mape:.0f}%")
log(f"    Median bias:             {peak_bias:+.0f}%")
log(f"    Innenfor ±25%:           {within_25:.0f}% av felt")
log(f"    Innenfor ±35%:           {within_35:.0f}% av felt")

# Ramp / plateau
res_ramp = res.dropna(subset=["ramp_actual"])
ramp_mae = (res_ramp.ramp_pred - res_ramp.ramp_actual).abs().median()
plat_mae = (res.plat_pred - res.plat_actual).abs().median()
log(f"\n  RAMP:    median absolutt feil = {ramp_mae:.0f} mnd")
log(f"  PLATEAU: median absolutt feil = {plat_mae:.0f} mnd")

# Bias-analyse
log(f"\n  BIAS-ANALYSE (er modellen systematisk skjev?):")
log(f"    Felt overpredikert (>0):  {(res.peak_err > 0).sum()}/{len(res)}")
log(f"    Felt underpredikert (<0): {(res.peak_err < 0).sum()}/{len(res)}")
if abs(peak_bias) < 10:
    log(f"    → Median bias {peak_bias:+.0f}% er liten — modellen er rimelig balansert")
else:
    direction = "over" if peak_bias > 0 else "under"
    log(f"    → Median bias {peak_bias:+.0f}% — modellen {direction}predikerer systematisk")

# ═══════════════════════════════════════════════════════════════
# KONTEKST: Johan Sverdrup spesifikt (det viktigste testfeltet)
# ═══════════════════════════════════════════════════════════════
js = res[res.field == "JOHAN SVERDRUP"]
if len(js) > 0:
    js = js.iloc[0]
    log(f"\n  ── JOHAN SVERDRUP (gull-standard test) ──")
    log(f"    Recoverable: {js.recoverable:.0f} MSm³")
    log(f"    Faktisk peak: {js.peak_actual:.3f} MSm³/mnd = {js.peak_actual*TO_KBOED:.0f} kboe/d")
    log(f"    Predikert:    {js.peak_pred:.3f} MSm³/mnd = {js.peak_pred*TO_KBOED:.0f} kboe/d")
    log(f"    Feil:         {js.peak_err:+.0f}%")
    log(f"    NB: JS er et atypisk mega-felt (lavt trykk, høy RF) — vanskelig case")

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Out-of-Sample Hold-out Validering — modellen har ALDRI sett testfeltet",
             fontsize=14, fontweight="bold", y=1.02)

# Panel 1: Peak predicted vs actual
ax = axes[0]
ax.scatter(res.peak_pred, res.peak_actual, s=70, alpha=0.75, c="#1565C0",
           edgecolors="white", lw=0.5)
for _, r in res.iterrows():
    ax.annotate(r.field[:10], (r.peak_pred, r.peak_actual), fontsize=6, alpha=0.8,
                xytext=(4, 3), textcoords="offset points")
lims = [0, max(res.peak_actual.max(), res.peak_pred.max()) * 1.1]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="Perfekt")
ax.fill_between(lims, [l*0.65 for l in lims], [l*1.35 for l in lims],
                alpha=0.1, color="green", label="±35%")
ax.set_xlabel("Predikert peak (blind, MSm³/mnd)")
ax.set_ylabel("Faktisk peak (MSm³/mnd)")
ax.set_title(f"PEAK: median |feil| = {peak_mape:.0f}%, {within_35:.0f}% innen ±35%",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: Error by recoverable size
ax = axes[1]
colors_err = ["#2E7D32" if abs(e) < 35 else "#C62828" for e in res.peak_err]
ax.scatter(res.recoverable, res.peak_err, s=70, alpha=0.75, c=colors_err,
           edgecolors="white", lw=0.5)
for _, r in res.iterrows():
    ax.annotate(r.field[:10], (r.recoverable, r.peak_err), fontsize=6, alpha=0.8,
                xytext=(4, 3), textcoords="offset points")
ax.axhline(0, color="black", lw=1)
ax.axhline(35, color="gray", ls=":", lw=0.8)
ax.axhline(-35, color="gray", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("Recoverable (MSm³, log)")
ax.set_ylabel("Peak %-feil")
ax.set_title("Feil vs feltstørrelse", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 3: Error distribution
ax = axes[2]
ax.hist(res.peak_err, bins=10, color="#1565C0", alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.axvline(peak_bias, color="red", ls="--", lw=1.5, label=f"Median: {peak_bias:+.0f}%")
ax.axvspan(-35, 35, alpha=0.1, color="green")
ax.set_xlabel("Peak %-feil")
ax.set_ylabel("Antall felt")
ax.set_title("Feilfordeling", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(RESULTS / "fig_holdout_validation.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_holdout_validation.png")

res.to_csv(DATA / "holdout_validation.csv", index=False)
log(f"Saved: holdout_validation.csv")

# ═══════════════════════════════════════════════════════════════
# VERDIKT
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("VERDIKT")
log("═" * 80)
if peak_mape < 30 and within_35 >= 65:
    log(f"""
  ✅ PEAK-MODELLEN BESTÅR out-of-sample test
     Median |feil| {peak_mape:.0f}%, {within_35:.0f}% innen ±35%
     → Klar for ER-bruk som point-estimate med ±35% usikkerhet""")
elif peak_mape < 40:
    log(f"""
  🟠 PEAK-MODELLEN ER BRUKBAR men med stor usikkerhet
     Median |feil| {peak_mape:.0f}%, {within_35:.0f}% innen ±35%
     → Bruk kun som sanity-check mot operatør-guidance, ikke standalone""")
else:
    log(f"""
  🚩 PEAK-MODELLEN BOMMER FOR MYE for standalone bruk
     Median |feil| {peak_mape:.0f}%
     → Trenger flere variabler eller bedre input-data""")

with open(RESULTS / "holdout_validation.txt", "w") as f:
    f.write("\n".join(lines))
