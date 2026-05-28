"""
Script 02: Exploratory analysis — quality vs. decline relationships.

Uses peak-normalized production and fitted exponential decline constant D.

Outputs (in results/):
  - scatter_api_D.png
  - scatter_sulfur_D.png
  - correlation_heatmap.png
  - decline_distribution.png
  - decline_by_area.png
  - peak_normalized_curves.png
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})
SAVEKW = dict(bbox_inches="tight")

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

panel = pd.read_csv(DATA / "panel_monthly.csv", parse_dates=["date"])
summary = pd.read_csv(DATA / "field_summary.csv")

# Drop fields without a valid D fit
summary_valid = summary.dropna(subset=["D_annual"]).copy()

print(f"Panel: {len(panel):,} obs, {panel.field.nunique()} fields")
print(f"Summary (with valid D): {len(summary_valid)} fields")

# ── Helper ──────────────────────────────────────────────────────────────────

area_colors = {"North sea": "#2196F3", "Norwegian sea": "#FF9800", "Barents sea": "#4CAF50"}
area_markers = {"North sea": "o", "Norwegian sea": "s", "Barents sea": "D"}


def scatter_quality_vs_D(df, x_col, x_label, filename):
    """Scatter plot of quality feature vs. annual decline constant D."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for area, color in area_colors.items():
        mask = df.main_area == area
        marker = area_markers[area]
        ax.scatter(
            df.loc[mask, x_col], df.loc[mask, "D_annual"],
            c=color, marker=marker, label=area,
            s=np.clip(df.loc[mask, "n_post_peak"] / 3, 25, 180),
            alpha=0.7, edgecolors="white", linewidths=0.5,
        )

    # OLS fit
    x, y = df[x_col].values, df.D_annual.values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, "k--", alpha=0.5, linewidth=1)

    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.text(0.02, 0.98, f"r={r:.3f} (p={p:.3f}) {sig}\nslope={slope:.5f}",
            transform=ax.transAxes, va="top", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # Label extremes
    for _, row in pd.concat([df.nsmallest(2, "D_annual"), df.nlargest(2, "D_annual")]).drop_duplicates().iterrows():
        ax.annotate(row.field, (row[x_col], row.D_annual),
                    fontsize=6.5, alpha=0.65, xytext=(5, 5), textcoords="offset points")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Annual Decline Constant D (yr⁻¹)")
    ax.set_title(f"{x_label} vs. Exponential Decline Rate — NCS Fields")
    ax.legend(title="Main area", fontsize=8)
    fig.savefig(RESULTS / filename, **SAVEKW)
    plt.close()
    print(f"Saved {filename}")


# ═══════════════════════════════════════════════════════════════════════════
# 1 & 2. Scatter: API and Sulfur vs. D
# ═══════════════════════════════════════════════════════════════════════════

scatter_quality_vs_D(summary_valid, "api_gravity", "API Gravity", "scatter_api_D.png")
scatter_quality_vs_D(summary_valid, "sulfur_pct", "Sulfur Content (%)", "scatter_sulfur_D.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════

quality_cols = [
    "api_gravity", "sulfur_pct", "pour_point_c", "vacuum_resid_pct",
    "naphtha_pct", "middle_distillate_pct", "bottom_of_barrel_pct", "wax_pct",
]
decline_cols = ["D_annual", "D_r2", "half_life_months", "decline_median", "decline_std"]
field_cols = ["field_age_mean", "oil_mean", "n_months", "water_cut_mean", "gor_mean"]

all_cols = quality_cols + decline_cols + field_cols
corr_data = summary_valid[all_cols].dropna(axis=1, how="all")
corr = corr_data.corr()

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

labels = [c.replace("_", " ").replace("pct", "%").title() for c in corr.columns]
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(labels, fontsize=7.5)

for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)

fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Correlation Matrix: Oil Quality × Decline Metrics (D) × Field Characteristics")
fig.savefig(RESULTS / "correlation_heatmap.png", **SAVEKW)
plt.close()
print("Saved correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. D distribution
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# D_annual distribution
ax = axes[0]
ax.hist(summary_valid.D_annual, bins=20, color="#607D8B", alpha=0.8, edgecolor="white", linewidth=0.3)
med_D = summary_valid.D_annual.median()
ax.axvline(med_D, color="orange", linewidth=1.5, linestyle="--", label=f"Median: {med_D:.3f} yr⁻¹")
ax.set_xlabel("Annual Decline Constant D (yr⁻¹)")
ax.set_ylabel("Frequency")
ax.set_title(f"Exponential Decline Rate Distribution (N={len(summary_valid)})")
ax.legend(fontsize=8)

skew = summary_valid.D_annual.skew()
ax.text(0.98, 0.98, f"Skew: {skew:.2f}\nMean: {summary_valid.D_annual.mean():.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Half-life distribution
ax = axes[1]
hl = summary_valid.half_life_months.dropna()
hl_clipped = hl.clip(upper=300)
ax.hist(hl_clipped, bins=20, color="#795548", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(hl.median(), color="orange", linewidth=1.5, linestyle="--",
           label=f"Median: {hl.median():.0f} mnd ({hl.median()/12:.1f} yr)")
ax.set_xlabel("Half-Life (months)")
ax.set_ylabel("Frequency")
ax.set_title("Time to 50% of Peak Production")
ax.legend(fontsize=8)

fig.suptitle("Decline Rate Distributions — NCS Fields (Peak-Normalized)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "decline_distribution.png", **SAVEKW)
plt.close()
print("Saved decline_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5. D by main area
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

areas = ["North sea", "Norwegian sea", "Barents sea"]
area_data = [summary_valid.loc[summary_valid.main_area == a, "D_annual"].dropna() for a in areas]
area_data = [d for d in area_data if len(d) > 0]
area_labels = [f"{a}\n(n={len(d)})" for a, d in zip(areas, area_data)]

bp = ax.boxplot(area_data, labels=area_labels, patch_artist=True, widths=0.5)
colors = ["#2196F3", "#FF9800", "#4CAF50"]
for patch, color in zip(bp["boxes"], colors[:len(area_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel("Annual Decline Constant D (yr⁻¹)")
ax.set_title("Exponential Decline Rate by Main Area")
fig.savefig(RESULTS / "decline_by_area.png", **SAVEKW)
plt.close()
print("Saved decline_by_area.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Peak-normalized production curves by quality class
# ═══════════════════════════════════════════════════════════════════════════

panel["api_class"] = pd.cut(
    panel.api_gravity,
    bins=[0, 30, 40, 100],
    labels=["Heavy (<30 API)", "Medium (30-40)", "Light (>40 API)"],
)

# Post-peak only
post_peak = panel[panel.is_post_peak].copy()

fig, ax = plt.subplots(figsize=(12, 6))

class_colors = {"Heavy (<30 API)": "#D32F2F", "Medium (30-40)": "#FF9800", "Light (>40 API)": "#4CAF50"}

# Individual traces (faint)
for field, grp in post_peak.groupby("field"):
    cls = grp.api_class.iloc[0]
    if pd.isna(cls):
        continue
    ax.plot(grp.months_since_peak, grp.oil_pct_peak,
            color=class_colors[cls], alpha=0.06, linewidth=0.5, zorder=1)

# Class medians
for cls, color in class_colors.items():
    d = post_peak[post_peak.api_class == cls]
    if len(d) == 0:
        continue
    monthly_med = d.groupby("months_since_peak")["oil_pct_peak"].median()
    smoothed = monthly_med.rolling(6, min_periods=3, center=True).median()
    n_fields = d.field.nunique()

    # Median D for this class
    cls_fields = summary_valid[summary_valid.field.isin(d.field.unique())]
    med_D = cls_fields.D_annual.median()

    ax.plot(smoothed.index, smoothed.values, color=color, linewidth=2.5,
            label=f"{cls} (n={n_fields}, D={med_D:.3f})", zorder=3)

ax.axhline(100, color="#9E9E9E", linewidth=0.8, linestyle=":", zorder=0)
ax.axhline(50, color="#9E9E9E", linewidth=0.5, linestyle=":", alpha=0.5, zorder=0)
ax.set_xlim(0, 350)
ax.set_ylim(0, 120)
ax.set_xlabel("Months Since Peak Production")
ax.set_ylabel("Production (% of Peak)")
ax.set_title("Post-Peak Production Decline by Oil Quality Class (normalized to peak = 100%)")
ax.legend(fontsize=9, title="API Gravity Class", title_fontsize=10, loc="upper right")

ax.text(0.98, 0.35,
        "Faint lines = individual fields\nBold lines = class median (6-month smoothed)",
        transform=ax.transAxes, fontsize=8, va="top", ha="right", alpha=0.5, style="italic")

fig.tight_layout()
fig.savefig(RESULTS / "peak_normalized_curves.png", **SAVEKW)
plt.close()
print("Saved peak_normalized_curves.png")

# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Key correlations with D_annual ──")
for col in ["api_gravity", "sulfur_pct", "vacuum_resid_pct", "pour_point_c",
            "field_age_mean", "water_cut_mean", "oil_mean"]:
    valid = summary_valid[[col, "D_annual"]].dropna()
    if len(valid) < 5:
        continue
    r, p = stats.pearsonr(valid[col], valid.D_annual)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {col:25s} r={r:+.3f} (p={p:.3f}) {sig}")

print(f"\n  Overall: median D={summary_valid.D_annual.median():.4f}/yr, "
      f"half-life={summary_valid.half_life_months.median():.0f} months")
