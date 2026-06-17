"""
Script 29: Peak Model V2 — Systematisk forbedring
═══════════════════════════════════════════════════════════════════════════

Tester kandidat-formuleringer for å redusere 38% out-of-sample feil.
Alle evalueres med EKTE LOO (re-tren uten testfelt) + lineær %-feil.

Kandidater (alle ex-ante):
  M0: Baseline           ln(peak) ~ ln(R) + ln(wells) + facility
  M1: + Smearing         M0 + Duan retransformasjons-korreksjon
  M2: + wells/R density  M0 + ln(wells per recoverable)
  M3: + kvadratisk ln(R) M0 + ln(R)²  (mega-felt-bøyning)
  M4: Offtake framing    ln(peak/R) ~ facility + ln(wells/R)
  M5: Beste kombo + smearing

Metrikk: lineær median |%-feil| via LOO på alle 63 felt + 13 nylige felt.
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
log("SCRIPT 29: PEAK MODEL V2 — SYSTEMATISK FORBEDRING")
log("═" * 80)

# Load
typecurve = pd.read_csv(DATA / "typecurve_library.csv")
master = pd.read_csv(DATA / "master_fluid_library_v51.csv")
reserves = pd.read_csv(RAW / "sodir_field_reserves.csv")
if "DatesyncNPD" in reserves.columns:
    reserves = reserves.sort_values("DatesyncNPD")
res = (reserves.groupby("fldName").tail(1)[["fldName", "fldRecoverableOil"]]
       .rename(columns={"fldName": "field", "fldRecoverableOil": "recoverable_msm3"}))
typecurve = typecurve.merge(res, on="field", how="left")

df = typecurve.dropna(subset=["recoverable_msm3", "facility_type", "peak_oil_msm3"]).copy()
df = df[df.recoverable_msm3 > 0.5]
df = df[df.n_wells_total > 0].reset_index(drop=True)

# Features
df["log_peak"] = np.log(df.peak_oil_msm3)
df["log_R"] = np.log(df.recoverable_msm3)
df["log_R_sq"] = df.log_R ** 2
df["log_wells"] = np.log(df.n_wells_total.clip(lower=1))
df["wells_per_R"] = df.n_wells_total / df.recoverable_msm3
df["log_wells_per_R"] = np.log(df.wells_per_R.clip(lower=0.001))
df["log_peak_per_R"] = np.log(df.peak_oil_msm3 / df.recoverable_msm3)

fac_d = pd.get_dummies(df.facility_type, prefix="fac", drop_first=True).astype(int)
df = pd.concat([df, fac_d], axis=1)
fac_cols = list(fac_d.columns)

recent_fields = ["JOHAN SVERDRUP", "EDVARD GRIEG", "MARTIN LINGE", "GOLIAT",
                 "IVAR AASEN", "GINA KROG", "MARIA", "GUDRUN", "KNARR", "GJØA",
                 "ALVHEIM", "SKARV", "VOLUND", "BØYLA", "YME", "FENJA", "DUVA"]
df["is_recent"] = df.field.isin(recent_fields)

log(f"\nDatasett: {len(df)} felt, {df.is_recent.sum()} nylige test-felt")

# ═══════════════════════════════════════════════════════════════
# LOO EVALUATOR — re-tren uten testfelt, predikter lineær peak
# ═══════════════════════════════════════════════════════════════
def evaluate_model(df, feature_cols, target_col, use_smearing=False,
                    is_offtake=False):
    """
    LOO: for hvert felt, tren på resten, predikter testfelt.
    Returnerer faktisk peak + predikert peak (lineær MSm³/mnd).
    """
    n = len(df)
    peak_actual = df.peak_oil_msm3.values
    peak_pred = np.zeros(n)

    for i in range(n):
        train = df.drop(df.index[i])
        test = df.iloc[[i]]

        X_tr = train[feature_cols].fillna(0).values
        y_tr = train[target_col].values
        lr = LinearRegression().fit(X_tr, y_tr)

        X_te = test[feature_cols].fillna(0).values
        pred_log = lr.predict(X_te)[0]

        # Smearing correction (Duan 1983)
        if use_smearing:
            resids = y_tr - lr.predict(X_tr)
            smear = np.mean(np.exp(resids))
        else:
            smear = 1.0

        if is_offtake:
            # target is ln(peak/R), so peak = exp(pred) * R * smear
            peak_pred[i] = np.exp(pred_log) * test.recoverable_msm3.values[0] * smear
        else:
            # target is ln(peak), so peak = exp(pred) * smear
            peak_pred[i] = np.exp(pred_log) * smear

    return peak_actual, peak_pred

def metrics(actual, pred, mask=None):
    if mask is not None:
        actual, pred = actual[mask], pred[mask]
    pct_err = (pred - actual) / actual * 100
    log_actual, log_pred = np.log(actual), np.log(pred)
    log_r2 = 1 - ((log_actual - log_pred)**2).sum() / ((log_actual - log_actual.mean())**2).sum()
    return {
        "median_ape": np.median(np.abs(pct_err)),
        "bias": np.median(pct_err),
        "within_25": (np.abs(pct_err) < 25).mean() * 100,
        "within_35": (np.abs(pct_err) < 35).mean() * 100,
        "log_r2": log_r2,
    }

# ═══════════════════════════════════════════════════════════════
# CANDIDATE MODELS
# ═══════════════════════════════════════════════════════════════
candidates = {
    "M0 Baseline": dict(
        feature_cols=["log_R", "log_wells"] + fac_cols, target_col="log_peak"),
    "M1 +Smearing": dict(
        feature_cols=["log_R", "log_wells"] + fac_cols, target_col="log_peak",
        use_smearing=True),
    "M2 +wells/R": dict(
        feature_cols=["log_R", "log_wells_per_R"] + fac_cols, target_col="log_peak",
        use_smearing=True),
    "M3 +kvadratisk R": dict(
        feature_cols=["log_R", "log_R_sq", "log_wells"] + fac_cols, target_col="log_peak",
        use_smearing=True),
    "M4 Offtake framing": dict(
        feature_cols=["log_R", "log_wells_per_R"] + fac_cols, target_col="log_peak_per_R",
        use_smearing=True, is_offtake=True),
    "M5 Offtake+kvadR": dict(
        feature_cols=["log_R", "log_R_sq", "log_wells_per_R"] + fac_cols,
        target_col="log_peak_per_R", use_smearing=True, is_offtake=True),
}

log("\n" + "═" * 80)
log("LOO-EVALUERING (alle 63 felt)")
log("═" * 80)
log(f"\n{'Model':22s} {'med|%feil|':>10s} {'bias':>7s} {'±25%':>6s} {'±35%':>6s} {'logR²':>7s}")
log("─" * 70)

all_metrics = {}
all_preds = {}
for name, spec in candidates.items():
    actual, pred = evaluate_model(df, **spec)
    m = metrics(actual, pred)
    all_metrics[name] = m
    all_preds[name] = (actual, pred)
    marker = ""
    log(f"  {name:22s} {m['median_ape']:9.0f}% {m['bias']:+6.0f}% "
        f"{m['within_25']:5.0f}% {m['within_35']:5.0f}% {m['log_r2']:7.3f}")

log("\n" + "═" * 80)
log("SAMME MODELLER PÅ KUN 13 NYLIGE FELT (hardere test)")
log("═" * 80)
log(f"\n{'Model':22s} {'med|%feil|':>10s} {'bias':>7s} {'±35%':>6s}")
log("─" * 55)
recent_metrics = {}
for name, spec in candidates.items():
    actual, pred = all_preds[name]
    mask = df.is_recent.values
    m = metrics(actual, pred, mask=mask)
    recent_metrics[name] = m
    log(f"  {name:22s} {m['median_ape']:9.0f}% {m['bias']:+6.0f}% {m['within_35']:5.0f}%")

# ═══════════════════════════════════════════════════════════════
# VELG BESTE MODELL
# ═══════════════════════════════════════════════════════════════
# Rank by recent-fields median APE (the relevant test)
best_name = min(recent_metrics, key=lambda k: recent_metrics[k]["median_ape"])
log("\n" + "═" * 80)
log(f"BESTE MODELL (lavest feil på nylige felt): {best_name}")
log("═" * 80)

best_full = all_metrics[best_name]
best_recent = recent_metrics[best_name]
baseline_recent = recent_metrics["M0 Baseline"]

log(f"\n  Forbedring vs M0 baseline (nylige felt):")
log(f"    Median |%-feil|:  {baseline_recent['median_ape']:.0f}% → {best_recent['median_ape']:.0f}%")
log(f"    Bias:             {baseline_recent['bias']:+.0f}% → {best_recent['bias']:+.0f}%")
log(f"    Innen ±35%:       {baseline_recent['within_35']:.0f}% → {best_recent['within_35']:.0f}%")

# Detail per-field for best model
log(f"\n  ── {best_name}: per-felt (nylige) ──")
log(f"  {'Felt':18s} {'Faktisk':>9s} {'Predikert':>10s} {'%-feil':>8s}")
actual, pred = all_preds[best_name]
df_pred = df.copy()
df_pred["pred"] = pred
df_pred["pct_err"] = (pred - actual) / actual * 100
for _, r in df_pred[df_pred.is_recent].sort_values("recoverable_msm3", ascending=False).iterrows():
    log(f"  {r.field:18s} {r.peak_oil_msm3:9.3f} {r.pred:10.3f} {r.pct_err:+7.0f}%")

# ═══════════════════════════════════════════════════════════════
# FIT FINAL MODEL ON ALL DATA + SAVE
# ═══════════════════════════════════════════════════════════════
import pickle
spec = candidates[best_name]
X_all = df[spec["feature_cols"]].fillna(0).values
y_all = df[spec["target_col"]].values
lr_final = LinearRegression().fit(X_all, y_all)
resids = y_all - lr_final.predict(X_all)
smear_factor = float(np.mean(np.exp(resids))) if spec.get("use_smearing") else 1.0

artifact = {
    "model_name": best_name,
    "feature_cols": spec["feature_cols"],
    "target_col": spec["target_col"],
    "is_offtake": spec.get("is_offtake", False),
    "smear_factor": smear_factor,
    "intercept": float(lr_final.intercept_),
    "coefs": lr_final.coef_.tolist(),
    "facility_categories": fac_cols,
    "loo_median_ape_all": float(best_full["median_ape"]),
    "loo_median_ape_recent": float(best_recent["median_ape"]),
    "loo_log_r2": float(best_full["log_r2"]),
}
with open(DATA / "peak_forecast_v2.pkl", "wb") as f:
    pickle.dump(artifact, f)
log(f"\n  Smearing factor: {smear_factor:.3f}")
log(f"  Saved: peak_forecast_v2.pkl")

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f"Peak Model V2 — kandidat-sammenligning (beste: {best_name})",
             fontsize=14, fontweight="bold", y=1.02)

# Panel 1: model comparison bars
ax = axes[0]
names = list(candidates.keys())
recent_apes = [recent_metrics[n]["median_ape"] for n in names]
all_apes = [all_metrics[n]["median_ape"] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, all_apes, 0.4, color="#90CAF9", alpha=0.85, label="Alle 63 felt")
ax.bar(x + 0.2, recent_apes, 0.4, color="#1565C0", alpha=0.85, label="13 nylige")
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7, rotation=0)
ax.set_ylabel("Median |%-feil|")
ax.set_title("Kandidat-modeller (LOO)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.axhline(35, color="green", ls=":", lw=1, alpha=0.6)

# Panel 2: best model predicted vs actual (recent)
ax = axes[1]
actual, pred = all_preds[best_name]
mask = df.is_recent.values
ax.scatter(pred[mask], actual[mask], s=70, alpha=0.75, c="#2E7D32",
           edgecolors="white", lw=0.5)
for i in np.where(mask)[0]:
    ax.annotate(df.iloc[i].field[:10], (pred[i], actual[i]), fontsize=6, alpha=0.8,
                xytext=(4, 3), textcoords="offset points")
lims = [0, max(actual[mask].max(), pred[mask].max()) * 1.1]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
ax.fill_between(lims, [l*0.65 for l in lims], [l*1.35 for l in lims], alpha=0.1, color="green")
ax.set_xlabel("Predikert (blind)"); ax.set_ylabel("Faktisk")
ax.set_title(f"{best_name}: nylige felt\nmedian |feil| = {best_recent['median_ape']:.0f}%",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 3: bias comparison
ax = axes[2]
recent_bias = [recent_metrics[n]["bias"] for n in names]
colors_b = ["#2E7D32" if abs(b) < 10 else "#FF9800" if abs(b) < 20 else "#C62828" for b in recent_bias]
ax.barh(x, recent_bias, color=colors_b, alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels(names, fontsize=8)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Median bias (%)")
ax.set_title("Bias (nylige felt)\ngrønn=balansert", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(RESULTS / "fig_peak_model_v2.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_peak_model_v2.png")

with open(RESULTS / "peak_model_v2.txt", "w") as f:
    f.write("\n".join(lines))
