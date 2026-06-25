"""
Steg 5: Slå sammen kvalitetsdata og differensialer, og se på sammenhengen
mellom oljekvalitet (API-grad, svovel) og pris-premium/rabatt mot Brent.

Pipeline:
  1. Last kvalitets-CSV (én rad per felt: API, svovel).
  2. Last differensial-CSV (mange rader per felt: en differensial per måned).
  3. Aggreger differensialer til snitt + spredning per felt.
  4. Slå sammen på felt-navn.
  5. Beregn korrelasjon mellom kvalitet og snitt-differensial.
  6. Plott: API vs differensial, svovel vs differensial, med trendlinje.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALITY_CSV = PROJECT_ROOT / "data" / "raw" / "crude_quality.csv"
DIFF_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def aggregate_differentials(diff: pd.DataFrame) -> pd.DataFrame:
    """Reduser månedlige differensialer til ett tall per felt:
    snitt, std-avvik, og antall observasjoner."""
    agg = diff.groupby("field").agg(
        mean_diff_usd=("differential_usd", "mean"),
        std_diff_usd=("differential_usd", "std"),
        n_obs=("differential_usd", "size"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).reset_index()
    return agg


def add_trendline(ax, x, y, color="black"):
    """Tegn en enkel lineær regresjons-linje (least squares).
    Returner (slope, intercept, korrelasjon r)."""
    # numpy.polyfit med degree=1 gir [slope, intercept].
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ys = slope * xs + intercept
    ax.plot(xs, ys, color=color, linestyle="--", alpha=0.6, linewidth=1.5,
            label=f"Trend: y = {slope:+.3f}·x {intercept:+.2f}")
    # Pearson r — np.corrcoef gir 2x2-matrise, vi tar [0,1].
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, intercept, r


def scatter_with_labels(ax, df, xcol, ycol, title, xlabel):
    """Scatter-plott der hvert punkt er ett felt, med navn ved siden av.
    Punktstørrelse skalerer med antall observasjoner."""
    # Marker-størrelse: minst 50, opp til 350 — jo flere observasjoner jo større.
    sizes = 50 + (df["n_obs"] / df["n_obs"].max()) * 300
    ax.scatter(df[xcol], df[ycol], s=sizes, alpha=0.65, color="steelblue",
               edgecolor="navy", linewidth=0.7)

    for _, row in df.iterrows():
        ax.annotate(
            row["field"].title(),
            (row[xcol], row[ycol]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8,
        )

    # Trendlinje + korrelasjon (kun for felt der vi har begge tall).
    valid = df[[xcol, ycol]].dropna()
    if len(valid) >= 3:
        slope, intercept, r = add_trendline(ax, valid[xcol].values, valid[ycol].values)
        ax.text(0.02, 0.98, f"Pearson r = {r:+.2f}\nN = {len(valid)} felt",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Snitt-differensial mot Brent (USD/fat)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    quality = pd.read_csv(QUALITY_CSV)
    diff = pd.read_csv(DIFF_CSV)

    quality["field"] = quality["field"].str.upper().str.strip()
    diff["field"] = diff["field"].str.upper().str.strip()

    agg = aggregate_differentials(diff)

    # Inner join: kun felt som finnes i begge datasett.
    merged = quality.merge(agg, on="field", how="inner")
    merged = merged.sort_values("mean_diff_usd", ascending=False).reset_index(drop=True)

    # Skriv ut samlet tabell.
    print("=== Sammenslått: kvalitet + snitt-differensial per felt ===")
    cols = ["field", "api_gravity", "sulfur_pct", "mean_diff_usd",
            "std_diff_usd", "n_obs", "first_year", "last_year"]
    print(merged[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    out_csv = OUT_DIR / "quality_vs_differential.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\nSamlet tabell lagret: {out_csv}")

    # Sjekk hvilke felt som finnes i bare én av tabellene.
    quality_only = set(quality["field"]) - set(agg["field"])
    diff_only = set(agg["field"]) - set(quality["field"])
    if quality_only:
        print(f"\nFelt med kvalitet men ingen differensial-data: {sorted(quality_only)}")
    if diff_only:
        print(f"Felt med differensial men ingen kvalitet-data: {sorted(diff_only)}")

    # === Plott ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    scatter_with_labels(
        axes[0], merged,
        xcol="api_gravity", ycol="mean_diff_usd",
        title="Lettere olje (høyere API) → premium?",
        xlabel="API-grad (høyere = lettere)",
    )
    scatter_with_labels(
        axes[1], merged,
        xcol="sulfur_pct", ycol="mean_diff_usd",
        title="Mer svovel (surere) → rabatt?",
        xlabel="Svovel (%) — lavere = søtere",
    )

    fig.suptitle("Norske oljefelt: kvalitet vs. pris-differensial mot Brent\n"
                 "(snitt over hele perioden, punktstørrelse = antall obs.)",
                 fontsize=12)
    fig.tight_layout()

    out_png = OUT_DIR / "04_quality_vs_differential.png"
    fig.savefig(out_png, dpi=130)
    print(f"Plott lagret: {out_png}")


if __name__ == "__main__":
    main()
