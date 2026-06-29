"""
Generate the static figures embedded in the README.

Outputs (committed as PNGs in figures/):
  1. decline_curves.png    Post-peak oil decline for six NCS fields, from Sodir monthly data.
  2. price_vs_quality.png   Mean realised differential vs Dated Brent against crude API gravity.

Run:  python figures/make_figures.py
Data: src/3_decline_lifecycle/data/panel_monthly.csv  (real Sodir monthly production)
      data/processed/quality_vs_differential.csv      (field differentials vs Brent)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# ---- consistent, clean house style ------------------------------------------
rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.frameon": False,
})
INK = "#1b1b1b"
ACCENT = "#0b6e8f"   # muted petroleum teal
FIT = "#c44e34"      # muted brick for fitted lines


def fig_decline_curves():
    panel = pd.read_csv(ROOT / "src/3_decline_lifecycle/data/panel_monthly.csv")
    pred = pd.read_csv(ROOT / "src/3_decline_lifecycle/data/predictions_v51.csv")
    dmap = dict(zip(pred.field, pred.D_annual))
    amap = dict(zip(pred.field, pred.api_v51))

    # Aggregate to full calendar years to remove maintenance-shutdown noise.
    g = (panel.groupby(["field", "year"])
               .agg(oil=("oil_msm3", "sum"), nmonths=("oil_msm3", "size"))
               .reset_index())
    g = g[g.nmonths == 12]  # complete years only, so partial first/last years do not distort

    # six fields spanning the observed decline range (slow -> fast)
    fields = ["EKOFISK", "ALVHEIM", "HEIDRUN", "GULLFAKS", "NORNE", "EDVARD GRIEG"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey=True)
    for ax, fld in zip(axes.ravel(), fields):
        d = g[g.field == fld].sort_values("year")
        peak_year = d.loc[d.oil.idxmax(), "year"]
        d = d[d.year >= peak_year]
        peak_oil = d.oil.iloc[0]
        t_yr = (d.year - peak_year).values
        q_pct = 100.0 * d.oil.values / peak_oil

        # real annual production, normalised to peak year
        ax.plot(t_yr, q_pct, color=ACCENT, lw=1.6, marker="o", ms=4,
                label="Sodir annual")
        # exponential decline fitted at the field's annual rate
        D = dmap[fld]
        tt = np.linspace(0, t_yr.max(), 100)
        ax.plot(tt, 100.0 * np.exp(-D * tt), color=FIT, lw=1.8, ls="--",
                label=f"exp. fit  {D*100:.0f}%/yr")

        ax.set_title(f"{fld.title()}   ({amap[fld]:.0f}° API)", color=INK)
        ax.set_ylim(0, 112)
        ax.set_xlim(0, max(6, t_yr.max()))
        ax.legend(loc="upper right", fontsize=8.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Years since peak production")
    for ax in axes[:, 0]:
        ax.set_ylabel("Oil rate, % of peak")

    fig.suptitle("Post-peak oil decline across NCS fields (Sodir annual production)",
                 fontsize=14, fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.008,
             "Real annual production normalised to each field's peak year, with an "
             "exponential decline fitted at the field's annual rate.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(OUT / "decline_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "decline_curves.png")


def fig_price_vs_quality():
    q = pd.read_csv(ROOT / "data/processed/quality_vs_differential.csv").dropna(
        subset=["api_gravity", "mean_diff_usd"])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(q.api_gravity, q.mean_diff_usd, c=q.sulfur_pct,
                    cmap="viridis_r", s=70, edgecolor="white", linewidth=0.8,
                    zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("Sulphur content (wt %)")
    cb.outline.set_visible(False)

    # OLS fit, reported honestly with R^2 (relationship is weak / non-monotonic)
    x, y = q.api_gravity.values, q.mean_diff_usd.values
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(x.min() - 1, x.max() + 1, 100)
    yhat = b0 + b1 * x
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ax.plot(xx, b0 + b1 * xx, color=FIT, lw=1.8, ls="--", zorder=2,
            label=f"OLS fit  (R² = {r2:.2f})")
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)

    # label the informative extremes, not every point
    for _, r in q.iterrows():
        if r.field in {"ALVHEIM", "TROLL", "OSEBERG", "GRANE", "JOHAN SVERDRUP",
                       "SKARV", "BALDER", "GUDRUN", "ASGARD", "ÅSGARD"}:
            ax.annotate(r.field.title(), (r.api_gravity, r.mean_diff_usd),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color="#333333")

    ax.set_xlabel("Crude quality  —  API gravity (°)")
    ax.set_ylabel("Mean realised differential vs Dated Brent (USD/bbl)")
    ax.set_title("Realised price differential against crude quality, NCS grades",
                 fontsize=14, fontweight="bold", color=INK)
    ax.legend(loc="lower left")
    fig.text(0.5, 0.01,
             "Each point is one field's 2012–2025 mean differential vs Brent. API gravity alone "
             "barely prices the crude: the relationship is weak and non-monotonic.",
             ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "price_vs_quality.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "price_vs_quality.png")


if __name__ == "__main__":
    fig_decline_curves()
    fig_price_vs_quality()
