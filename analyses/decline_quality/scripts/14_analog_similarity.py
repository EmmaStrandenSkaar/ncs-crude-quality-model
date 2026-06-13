"""
Script 14: Analog Similarity Scoring
═══════════════════════════════════════════════════════════════════════════

For et nytt felt, finn de mest sammenlignbare NCS-analogene basert på:
  - API gravity (drives decline)
  - Recoverable reserves (drives platå-lengde)
  - Facility type (drives ramp-shape)
  - Operator (track record)
  - Main area (basin/region)
  - Water depth (utbyggings-kompleksitet)
  - Decade (teknologi-epoke)

Output:
  - Funksjoner: similarity_score(), get_analogs(), predict_phases()
  - Validering: LOO-test mot eksisterende felt
  - Eksempel: Full Yggdrasil-prediksjon
  - Figur: fig_analog_similarity.png
  - Tabell: analog_similarity.txt
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

log("═" * 80)
log("SCRIPT 14: ANALOG SIMILARITY SCORING")
log("═" * 80)

# ═══════════════════════════════════════════════════════════════
# LOAD LIBRARY
# ═══════════════════════════════════════════════════════════════
lib = pd.read_csv(DATA / "typecurve_library.csv")
panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
log(f"\nBibliotek: {len(lib)} felt, {len(lib.columns)} kolonner")

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
lib = lib.copy()
lib["decade"] = (lib.first_year // 10 * 10).astype(int)
lib["log_peak"] = np.log10(lib.peak_oil_msm3.clip(lower=0.001))
lib["log_total"] = np.log10(lib.total_oil_msm3.clip(lower=0.001))

# Fill numeric NaN with median (for feature space only)
NUMERIC_FEATURES = ["api_gravity", "log_total", "log_peak", "water_depth", "decade"]
fill_vals = {f: lib[f].median() for f in NUMERIC_FEATURES if f in lib.columns}
for f, v in fill_vals.items():
    lib[f + "_filled"] = lib[f].fillna(v)

CATEGORICAL_FEATURES = ["facility_type", "operator", "main_area"]

# ═══════════════════════════════════════════════════════════════
# DEFAULT WEIGHTS
# ═══════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS = {
    "api_gravity":   2.0,   # Sterkt driver for decline + ramp-tek
    "log_total":     2.0,   # Driver platå-lengde + ramp-tid
    "log_peak":      1.0,   # Komplementær til total
    "water_depth":   0.5,   # Mindre viktig
    "decade":        1.5,   # Tek-epoke
    "facility_type": 3.0,   # Sterkt driver for ramp/platå-shape
    "operator":      2.0,   # Track record / utbyggings-filosofi
    "main_area":     1.0,   # Geologisk likhet
}

# ═══════════════════════════════════════════════════════════════
# SIMILARITY FUNCTION
# ═══════════════════════════════════════════════════════════════
def normalize_numeric(values, ref_values):
    """Z-score normalization using reference distribution."""
    mu = ref_values.mean()
    sd = ref_values.std()
    return (values - mu) / sd if sd > 0 else values * 0

def similarity_score(target_field, candidates, weights=None):
    """
    Compute similarity scores (0-1) between target and each candidate.

    target_field: dict with field characteristics
    candidates: DataFrame with filled features

    Returns: candidates with 'similarity' column (1.0 = identical, 0.0 = very different)
    """
    weights = weights or DEFAULT_WEIGHTS

    candidates = candidates.copy()
    distances = np.zeros(len(candidates))
    weight_sum = 0

    # Numeric features (Euclidean in normalized space)
    for feat in NUMERIC_FEATURES:
        if feat not in target_field or feat + "_filled" not in candidates.columns:
            continue
        target_val = target_field[feat]
        if pd.isna(target_val):
            continue
        cand_vals = candidates[feat + "_filled"].values
        sd = candidates[feat + "_filled"].std()
        if sd == 0:
            continue
        diff = (target_val - cand_vals) / sd
        w = weights.get(feat, 1.0)
        distances += w * diff ** 2
        weight_sum += w

    # Categorical features (1 if match, 0 if not)
    for feat in CATEGORICAL_FEATURES:
        if feat not in target_field or feat not in candidates.columns:
            continue
        target_val = target_field[feat]
        if pd.isna(target_val) or target_val == "":
            continue
        # Distance = 0 if match, 1 if no match
        matches = candidates[feat].astype(str).str.strip() == str(target_val).strip()
        diff = np.where(matches, 0.0, 1.0)
        w = weights.get(feat, 1.0)
        distances += w * diff ** 2
        weight_sum += w

    # Normalize distances to similarity score
    if weight_sum > 0:
        distances = np.sqrt(distances / weight_sum)
    # Convert distance to similarity: similarity = exp(-distance)
    candidates["similarity"] = np.exp(-distances)
    candidates["distance"] = distances
    return candidates.sort_values("similarity", ascending=False)

# ═══════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════
def predict_phases(target_field, library, top_n=5, weights=None, exclude_field=None):
    """
    Predict ramp/plateau/decline parameters by weighted average of top-N analogs.
    """
    candidates = library.copy()
    if exclude_field:
        candidates = candidates[candidates.field != exclude_field]

    scored = similarity_score(target_field, candidates, weights)
    top = scored.head(top_n).copy()

    # Weighted by similarity
    weights_arr = top.similarity.values
    weights_arr = weights_arr / weights_arr.sum()

    predictions = {
        "ramp_length_months": np.nansum(weights_arr * top.ramp_length_months.values),
        "plateau_length_months": np.nansum(weights_arr * top.plateau_length_months.values),
        "D_decline_fit": np.nansum(weights_arr * top.D_decline_fit.fillna(top.D_decline_fit.median()).values),
        "D_12": np.nansum(weights_arr * top.D_12.fillna(top.D_12.median()).values),
    }

    # Weighted median for robustness
    def weighted_median(vals, ws):
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            return np.nan
        vals = vals[valid]
        ws = ws[valid]
        order = np.argsort(vals)
        vals = vals[order]
        ws = ws[order]
        cumw = np.cumsum(ws) / ws.sum()
        return vals[np.searchsorted(cumw, 0.5)]

    predictions["ramp_median"] = weighted_median(top.ramp_length_months.values, weights_arr)
    predictions["plateau_median"] = weighted_median(top.plateau_length_months.values, weights_arr)
    predictions["D_median"] = weighted_median(top.D_decline_fit.values, weights_arr)

    return predictions, top

# ═══════════════════════════════════════════════════════════════
# VALIDATION: Leave-One-Out on existing fields
# ═══════════════════════════════════════════════════════════════
log("\n" + "═" * 80)
log("VALIDERING: Leave-One-Out test")
log("═" * 80)

valid_lib = lib.dropna(subset=["ramp_length_months", "plateau_length_months"]).copy()

loo_results = []
for _, row in valid_lib.iterrows():
    target = {
        "api_gravity": row.api_gravity,
        "log_total": row.log_total,
        "log_peak": row.log_peak,
        "water_depth": row.water_depth,
        "decade": row.decade,
        "facility_type": row.facility_type,
        "operator": row.operator,
        "main_area": row.main_area,
    }
    preds, top = predict_phases(target, valid_lib, top_n=5, exclude_field=row.field)
    loo_results.append({
        "field": row.field,
        "actual_ramp": row.ramp_length_months,
        "pred_ramp": preds["ramp_length_months"],
        "pred_ramp_med": preds["ramp_median"],
        "actual_plateau": row.plateau_length_months,
        "pred_plateau": preds["plateau_length_months"],
        "pred_plateau_med": preds["plateau_median"],
        "actual_D": row.D_decline_fit,
        "pred_D": preds["D_decline_fit"],
        "pred_D_med": preds["D_median"],
    })

loo = pd.DataFrame(loo_results)

# Performance metrics
def metrics(actual, predicted):
    mask = ~(pd.isna(actual) | pd.isna(predicted))
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan
    a, p = actual[mask], predicted[mask]
    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2

log(f"\n{'Variabel':25s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'n':>5s}")
log("─" * 60)

for var, lbl in [("ramp", "Ramp (mnd)"), ("plateau", "Platå (mnd)"), ("D", "Decline (D)")]:
    for method, suffix in [("Vektet snitt", ""), ("Vektet median", "_med")]:
        actual = loo[f"actual_{var}"]
        pred = loo[f"pred_{var}{suffix}"]
        mae, rmse, r2 = metrics(actual, pred)
        n = (~(pd.isna(actual) | pd.isna(pred))).sum()
        log(f"  {lbl + ' — ' + method:35s} {mae:>6.1f}  {rmse:>6.1f}  {r2:>6.3f}  {n:>4d}")
    log("")

# ═══════════════════════════════════════════════════════════════
# YGGDRASIL PREDICTION (Aker BP NOAKA hub)
# ═══════════════════════════════════════════════════════════════
log("═" * 80)
log("EKSEMPEL: YGGDRASIL (Aker BP, planlagt 2027)")
log("═" * 80)

# Yggdrasil-spesifikasjoner (offentlig kjent)
yggdrasil = {
    "api_gravity": 37.0,           # NOAKA blend (Krafla, Fulla, Frøy)
    "log_total": np.log10(100.0),  # ~650 mboe = ~100 MSm³ olje (recoverable)
    "log_peak": np.log10(2.0),     # ~120 kboe/d peak = ~2 MSm³/mnd
    "water_depth": 120.0,          # Nord-Nordsjøen
    "decade": 2020,                # Online 2027
    "facility_type": "FPSO",       # Floating production unit
    "operator": "Aker BP ASA",
    "main_area": "North sea",
}

log(f"\nYggdrasil-input:")
log(f"  API gravity:        {yggdrasil['api_gravity']}°")
log(f"  Recoverable oil:    ~{10**yggdrasil['log_total']:.0f} MSm³ (≈650 mboe)")
log(f"  Peak production:    ~{10**yggdrasil['log_peak']:.1f} MSm³/mnd (≈120 kboe/d)")
log(f"  Water depth:        {yggdrasil['water_depth']:.0f} m")
log(f"  Facility:           {yggdrasil['facility_type']}")
log(f"  Operator:           {yggdrasil['operator']}")
log(f"  Main area:          {yggdrasil['main_area']}")
log(f"  Online:             2027")

preds, top_analogs = predict_phases(yggdrasil, valid_lib, top_n=5)

log(f"\n── TOPP 5 ANALOGER ──")
log(f"\n{'Field':18s} {'Similarity':>11s} {'Operator':22s} {'Facility':15s} {'API':>5s} {'Ramp':>5s} {'Platå':>6s} {'D':>6s}")
log("─" * 105)
for _, r in top_analogs.iterrows():
    sim = f"{r.similarity:.3f}"
    op = str(r.operator)[:22]
    fac = str(r.facility_type)[:15]
    api = f"{r.api_gravity:.1f}" if not pd.isna(r.api_gravity) else "—"
    ramp = f"{r.ramp_length_months:.0f}"
    plat = f"{r.plateau_length_months:.0f}"
    d = f"{r.D_decline_fit:.3f}" if not pd.isna(r.D_decline_fit) else "—"
    log(f"{r.field:18s} {sim:>11s} {op:22s} {fac:15s} {api:>5s} {ramp:>5s} {plat:>6s} {d:>6s}")

log(f"\n── PREDIKSJON ──")
log(f"\n  Ramp-up (vektet snitt):    {preds['ramp_length_months']:>5.0f} mnd  ({preds['ramp_length_months']/12:.1f} år)")
log(f"  Ramp-up (vektet median):   {preds['ramp_median']:>5.0f} mnd  ({preds['ramp_median']/12:.1f} år)")
log(f"  Platå (vektet snitt):      {preds['plateau_length_months']:>5.0f} mnd  ({preds['plateau_length_months']/12:.1f} år)")
log(f"  Platå (vektet median):     {preds['plateau_median']:>5.0f} mnd  ({preds['plateau_median']/12:.1f} år)")
log(f"  Decline D (vektet snitt):  {preds['D_decline_fit']:>5.3f}")
log(f"  Decline D (vektet median): {preds['D_median']:>5.3f}")

# Tidslinje
start_year = 2027
ramp_end = start_year + preds['ramp_median'] / 12
plateau_end = ramp_end + preds['plateau_median'] / 12
log(f"\n── TIDSLINJE ──")
log(f"  Første olje:          {start_year}")
log(f"  Peak (ramp slutt):    {ramp_end:.1f}")
log(f"  Platå slutt:          {plateau_end:.1f}")
log(f"  Ved 50% av peak:      {plateau_end + (np.log(2) / preds['D_median']):.1f}")

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 13))
fig.suptitle("Analog Similarity Scoring — Validering & Yggdrasil-prediksjon",
             fontsize=15, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.32)

# ── Panel 1: LOO Ramp ──
ax = fig.add_subplot(gs[0, 0])
plot_data = loo.dropna(subset=["actual_ramp", "pred_ramp_med"])
ax.scatter(plot_data.actual_ramp, plot_data.pred_ramp_med, alpha=0.6, s=50, c="#1565C0")
lims = [0, max(plot_data.actual_ramp.max(), plot_data.pred_ramp_med.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
mae_r, rmse_r, r2_r = metrics(plot_data.actual_ramp, plot_data.pred_ramp_med)
ax.set_xlabel("Faktisk ramp (mnd)")
ax.set_ylabel("Predikert ramp (mnd)")
ax.set_title(f"LOO Validering — Ramp\nR²={r2_r:.3f}  MAE={mae_r:.1f} mnd", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# ── Panel 2: LOO Plateau ──
ax = fig.add_subplot(gs[0, 1])
plot_data = loo.dropna(subset=["actual_plateau", "pred_plateau_med"])
ax.scatter(plot_data.actual_plateau, plot_data.pred_plateau_med, alpha=0.6, s=50, c="#2E7D32")
lims = [0, max(plot_data.actual_plateau.max(), plot_data.pred_plateau_med.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
mae_p, rmse_p, r2_p = metrics(plot_data.actual_plateau, plot_data.pred_plateau_med)
ax.set_xlabel("Faktisk platå (mnd)")
ax.set_ylabel("Predikert platå (mnd)")
ax.set_title(f"LOO Validering — Platå\nR²={r2_p:.3f}  MAE={mae_p:.1f} mnd", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# ── Panel 3: LOO Decline ──
ax = fig.add_subplot(gs[0, 2])
plot_data = loo.dropna(subset=["actual_D", "pred_D_med"])
ax.scatter(plot_data.actual_D, plot_data.pred_D_med, alpha=0.6, s=50, c="#E91E63")
lims = [0, max(plot_data.actual_D.max(), plot_data.pred_D_med.max()) * 1.05]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
mae_d, rmse_d, r2_d = metrics(plot_data.actual_D, plot_data.pred_D_med)
ax.set_xlabel("Faktisk D")
ax.set_ylabel("Predikert D")
ax.set_title(f"LOO Validering — Decline\nR²={r2_d:.3f}  MAE={mae_d:.3f}", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# ── Panel 4: Yggdrasil analoger ──
ax = fig.add_subplot(gs[1, 0])
y_pos = np.arange(len(top_analogs))
ax.barh(y_pos, top_analogs.similarity.values, color="#1565C0", alpha=0.85, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_analogs.field.tolist(), fontsize=10)
ax.set_xlabel("Similarity score")
ax.set_title("Yggdrasil: Topp 5 NCS-analoger", fontsize=11, fontweight="bold")
for i, (_, r) in enumerate(top_analogs.iterrows()):
    ax.text(r.similarity + 0.005, i, f"{r.similarity:.3f}", va="center", fontsize=9)
ax.invert_yaxis()

# ── Panel 5: Yggdrasil predicted production profile ──
ax = fig.add_subplot(gs[1, 1])

# Plot top 5 analogers as overlays (normalized to peak)
colors_a = plt.cm.tab10(np.linspace(0, 0.9, len(top_analogs)))
for i, (_, a) in enumerate(top_analogs.iterrows()):
    fd = panel[panel.field == a.field].sort_values("date").copy()
    fd_norm = fd.oil_pct_peak.values / 100.0
    first_oil_arr = np.where(fd_norm > 0.05)[0]
    if len(first_oil_arr) == 0:
        continue
    first = first_oil_arr[0]
    fd_plot = fd_norm[first:]
    t_yr = np.arange(len(fd_plot)) / 12
    ax.plot(t_yr, fd_plot, color=colors_a[i], alpha=0.4, lw=1, label=a.field)

# Build Yggdrasil predicted profile
ramp_mo = preds["ramp_median"]
plateau_mo = preds["plateau_median"]
D = preds["D_median"]

total_mo = 360
t_mo = np.arange(total_mo)
profile = np.zeros(total_mo)
for i, m in enumerate(t_mo):
    if m < ramp_mo:
        profile[i] = 1 / (1 + np.exp(-0.15 * (m - ramp_mo / 2)))
    elif m < ramp_mo + plateau_mo:
        profile[i] = 1.0
    else:
        years_post = (m - ramp_mo - plateau_mo) / 12
        profile[i] = np.exp(-D * years_post)

ax.plot(t_mo / 12, profile, color="black", lw=3, label="Yggdrasil prediksjon", zorder=10)
ax.fill_between(t_mo / 12, profile, alpha=0.1, color="black")

ax.set_xlim(0, 25)
ax.set_ylim(0, 1.2)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("År siden første olje")
ax.set_ylabel("Produksjon / peak")
ax.set_title("Yggdrasil: Vektet prediksjon vs. analoger", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3)

# ── Panel 6: Yggdrasil prediction summary ──
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")

summary_text = f"""YGGDRASIL PREDIKSJON
Aker BP NOAKA hub, planlagt 2027

── Input ──
  API:        37°
  Reserver:   ~100 MSm³ olje
  Peak:       ~120 kboe/d
  Facility:   FPSO
  Vanndybde:  120 m

── Topp 5 analoger ──"""
for _, r in top_analogs.iterrows():
    summary_text += f"\n  {r.field:14s}  sim={r.similarity:.2f}"

summary_text += f"""

── Prediksjon (vektet median) ──
  Ramp-up:    {preds['ramp_median']:.0f} mnd ({preds['ramp_median']/12:.1f} år)
  Platå:      {preds['plateau_median']:.0f} mnd ({preds['plateau_median']/12:.1f} år)
  Decline D:  {preds['D_median']:.3f}

── Tidslinje ──
  Første olje:    2027
  Peak (ramp):    {ramp_end:.0f}
  Platå slutt:    {plateau_end:.0f}
  Halvert prod:   {plateau_end + np.log(2)/preds['D_median']:.0f}

── Validering (LOO) ──
  Ramp R²:    {r2_r:.2f}  MAE={mae_r:.0f} mnd
  Platå R²:   {r2_p:.2f}  MAE={mae_p:.0f} mnd
  Decline R²: {r2_d:.2f}  MAE={mae_d:.3f}"""

ax.text(0.05, 0.97, summary_text, transform=ax.transAxes, fontsize=9,
        fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="#E8F5E9", ec="#2E7D32", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS / "fig_analog_similarity.png", dpi=160, bbox_inches="tight")
log(f"\nSaved: fig_analog_similarity.png")

with open(RESULTS / "analog_similarity.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved: analog_similarity.txt")
