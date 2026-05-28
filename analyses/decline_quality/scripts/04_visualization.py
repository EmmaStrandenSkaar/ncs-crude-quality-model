"""
Script 04: Publication-quality figures for the decline-quality analysis.

Uses peak-normalized D as the decline metric.

Outputs (in results/):
  - fig_main_scatter.png        — API & sulfur vs. D (2-panel)
  - fig_decay_curves.png        — normalized decay curves by quality class
  - fig_regression_summary.png  — coefficient plot + model comparison
  - fig_size_confound.png       — partial regression / size confound visual
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")
df = summary.dropna(subset=["D_annual"]).copy()

SAVEKW = dict(bbox_inches="tight")

COLORS = {
    "heavy": "#C62828", "medium": "#F57C00", "light": "#2E7D32",
    "north": "#1565C0", "norwegian": "#EF6C00", "barents": "#2E7D32",
    "fit": "#455A64", "grid": "#E0E0E0",
}

def style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=11, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


area_map = {
    "North sea": ("North Sea", COLORS["north"], "o"),
    "Norwegian sea": ("Norwegian Sea", COLORS["norwegian"], "s"),
    "Barents sea": ("Barents Sea", COLORS["barents"], "D"),
}

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main scatter — API & Sulfur vs. D (2-panel)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (x_col, x_label) in enumerate([
    ("api_gravity", "API Gravity (°)"),
    ("sulfur_pct", "Sulfur Content (wt%)"),
]):
    ax = axes[ax_idx]

    for area, (label, color, marker) in area_map.items():
        mask = df.main_area == area
        d = df[mask]
        ax.scatter(
            d[x_col], d.D_annual,
            c=color, marker=marker, s=np.clip(d.n_post_peak / 3, 30, 200),
            alpha=0.7, edgecolors="white", linewidths=0.5, label=label, zorder=3,
        )

    x = df[x_col].values
    y = df.D_annual.values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = intercept + slope * x_fit

    n = len(x)
    x_mean = x.mean()
    s_res = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (n - 2))
    se_fit = s_res * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / np.sum((x - x_mean) ** 2))

    ax.plot(x_fit, y_fit, color=COLORS["fit"], linewidth=2, zorder=2)
    ax.fill_between(x_fit, y_fit - 1.96 * se_fit, y_fit + 1.96 * se_fit,
                     color=COLORS["fit"], alpha=0.12, zorder=1)

    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.03, 0.97,
            f"r = {r:+.3f} ({sig})\nslope = {slope:.5f}\nn = {n}",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDBDBD", alpha=0.9))

    style_ax(ax, x_label, "Annual Decline Constant D (yr⁻¹)")

    for _, row in pd.concat([df.nsmallest(2, "D_annual"), df.nlargest(2, "D_annual")]).drop_duplicates().iterrows():
        ax.annotate(row.field, (row[x_col], row.D_annual),
                    fontsize=7, alpha=0.6, xytext=(6, 4), textcoords="offset points")

axes[0].legend(fontsize=8, title="Region", title_fontsize=9, loc="upper right")

fig.suptitle("Oil Quality vs. Exponential Decline Rate — Norwegian Continental Shelf",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_main_scatter.png", **SAVEKW)
plt.close()
print("Saved fig_main_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Peak-normalized decay curves by quality class
# ═══════════════════════════════════════════════════════════════════════════

panel["api_class"] = pd.cut(
    panel.api_gravity,
    bins=[0, 30, 40, 100],
    labels=["Heavy (<30°)", "Medium (30-40°)", "Light (>40°)"],
)

post_peak = panel[panel.is_post_peak].copy()

fig, ax = plt.subplots(figsize=(12, 6))
class_colors = {
    "Heavy (<30°)": COLORS["heavy"],
    "Medium (30-40°)": COLORS["medium"],
    "Light (>40°)": COLORS["light"],
}

for field, grp in post_peak.groupby("field"):
    cls = grp.api_class.iloc[0]
    if pd.isna(cls):
        continue
    ax.plot(grp.months_since_peak, grp.oil_pct_peak,
            color=class_colors[cls], alpha=0.06, linewidth=0.5, zorder=1)

for cls, color in class_colors.items():
    d = post_peak[post_peak.api_class == cls]
    if len(d) == 0:
        continue
    monthly_med = d.groupby("months_since_peak")["oil_pct_peak"].median()
    smoothed = monthly_med.rolling(6, min_periods=3, center=True).median()
    n_fields = d.field.nunique()

    cls_fields = df[df.field.isin(d.field.unique())]
    med_D = cls_fields.D_annual.median()

    ax.plot(smoothed.index, smoothed.values, color=color, linewidth=2.5,
            label=f"{cls} (n={n_fields}, D={med_D:.3f})", zorder=3)

ax.axhline(100, color="#9E9E9E", linewidth=0.8, linestyle=":", zorder=0)
ax.axhline(50, color="#9E9E9E", linewidth=0.5, linestyle=":", alpha=0.5, zorder=0)
ax.text(2, 52, "50% of peak", fontsize=7, alpha=0.4)

ax.set_xlim(0, 350)
ax.set_ylim(0, 120)
style_ax(ax, "Months Since Peak Production", "Production (% of Peak)",
         "Post-Peak Production Decline by Oil Quality Class")
ax.legend(fontsize=9, title="API Gravity Class", title_fontsize=10, loc="upper right")

ax.text(0.98, 0.40,
        "Faint lines = individual fields\nBold lines = class median (6-month smoothed)",
        transform=ax.transAxes, fontsize=8, va="top", ha="right", alpha=0.5, style="italic")

fig.tight_layout()
fig.savefig(RESULTS / "fig_decay_curves.png", **SAVEKW)
plt.close()
print("Saved fig_decay_curves.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Regression summary — coefficients + model comparison
# ═══════════════════════════════════════════════════════════════════════════

QUALITY = ["api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct"]
df_reg = df.dropna(subset=QUALITY).copy()
y = df_reg["D_annual"]

models = {}
specs = {
    "Quality only": QUALITY,
    "+ Field size": QUALITY + ["oil_mean"],
    "+ Size + Age": QUALITY + ["oil_mean", "field_age_mean"],
}

for name, xcols in specs.items():
    X = sm.add_constant(df_reg[xcols])
    m = sm.OLS(y, X).fit(cov_type="HC1")
    models[name] = m

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

# Left: coefficients across specs
ax = axes[0]
var_labels = {"api_gravity": "API Gravity", "sulfur_pct": "Sulfur %",
              "pour_point_c": "Pour Point", "vacuum_resid_pct": "Vacuum Resid %",
              "oil_mean": "Avg Production"}

y_positions = np.arange(len(var_labels))
offsets = [-0.2, 0, 0.2]
spec_colors = ["#1565C0", "#E65100", "#2E7D32"]

for i, (spec_name, model) in enumerate(models.items()):
    for j, (var, label) in enumerate(var_labels.items()):
        if var not in model.params.index:
            continue
        c = model.params[var]
        lo, hi = model.conf_int().loc[var]
        p = model.pvalues[var]
        marker = "o" if p < 0.05 else "s" if p < 0.1 else "D"
        alpha = 1.0 if p < 0.1 else 0.4
        ax.plot(c, y_positions[j] + offsets[i], marker=marker, color=spec_colors[i],
                markersize=8, alpha=alpha, zorder=3)
        ax.plot([lo, hi], [y_positions[j] + offsets[i]] * 2,
                color=spec_colors[i], linewidth=2, alpha=alpha * 0.7, zorder=2)

    ax.plot([], [], color=spec_colors[i], marker="o", linewidth=2, label=spec_name, markersize=6)

ax.axvline(0, color="black", linewidth=1, zorder=0)
ax.set_yticks(y_positions)
ax.set_yticklabels(list(var_labels.values()))
ax.legend(fontsize=8, title="Specification", title_fontsize=9, loc="lower right")
style_ax(ax, "Coefficient (effect on D, yr⁻¹)", "",
         "Quality Coefficients Across Specifications")
ax.invert_yaxis()

# Right: model fit
ax = axes[1]
spec_names = list(models.keys())
r2 = [m.rsquared for m in models.values()]
adj_r2 = [m.rsquared_adj for m in models.values()]

x = np.arange(len(spec_names))
w = 0.35
ax.bar(x - w / 2, r2, w, color="#78909C", alpha=0.8, label="R²", edgecolor="white")
ax.bar(x + w / 2, adj_r2, w, color="#B0BEC5", alpha=0.8, label="Adj R²", edgecolor="white")

for i, (r, ar) in enumerate(zip(r2, adj_r2)):
    ax.text(i - w / 2, r + 0.005, f"{r:.3f}", ha="center", fontsize=8)
    ax.text(i + w / 2, ar + 0.005, f"{ar:.3f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(spec_names, fontsize=8, rotation=15)
ax.legend(fontsize=8)
style_ax(ax, "", "R²", "Model Fit")
ax.set_ylim(0, max(r2) * 1.3)

fig.suptitle("Cross-Sectional Regression: Quality → Decline Rate D (N=51 NCS fields)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "fig_regression_summary.png", **SAVEKW)
plt.close()
print("Saved fig_regression_summary.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Size confound — why API effect weakens when controlling for size
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: API vs D (raw)
ax = axes[0]
ax.scatter(df_reg.api_gravity, df_reg.D_annual, c=COLORS["north"], alpha=0.6, edgecolors="white", s=50)
slope, intercept, r, p, _ = stats.linregress(df_reg.api_gravity, df_reg.D_annual)
x_fit = np.linspace(df_reg.api_gravity.min(), df_reg.api_gravity.max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, color=COLORS["fit"], linewidth=2)
sig = "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
style_ax(ax, "API Gravity", "D (yr⁻¹)", "API vs. Decline (raw)")

# Panel 2: Field size vs D
ax = axes[1]
ax.scatter(df_reg.oil_mean, df_reg.D_annual, c=COLORS["norwegian"], alpha=0.6, edgecolors="white", s=50)
slope, intercept, r, p, _ = stats.linregress(df_reg.oil_mean, df_reg.D_annual)
x_fit = np.linspace(df_reg.oil_mean.min(), df_reg.oil_mean.max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, color=COLORS["fit"], linewidth=2)
sig = "***" if p < 0.001 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
style_ax(ax, "Avg Production (Mill Sm³/mnd)", "D (yr⁻¹)", "Field Size vs. Decline")

# Panel 3: Partial regression — API|size vs D|size
ax = axes[2]
X_size = sm.add_constant(df_reg[["oil_mean"]])
resid_api = sm.OLS(df_reg.api_gravity, X_size).fit().resid
resid_D = sm.OLS(df_reg.D_annual, X_size).fit().resid

ax.scatter(resid_api, resid_D, c=COLORS["light"], alpha=0.6, edgecolors="white", s=50)
slope, intercept, r, p, _ = stats.linregress(resid_api, resid_D)
x_fit = np.linspace(resid_api.min(), resid_api.max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, color=COLORS["fit"], linewidth=2)
sig = "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
ax.text(0.03, 0.97, f"r = {r:+.3f} ({sig})", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
style_ax(ax, "API (residual, net of size)", "D (residual, net of size)",
         "Partial: API vs. Decline | Size")

fig.suptitle("Decomposing the Quality–Decline Relationship: The Field Size Confound",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(RESULTS / "fig_size_confound.png", **SAVEKW)
plt.close()
print("Saved fig_size_confound.png")

print("\nAll figures saved to results/")
