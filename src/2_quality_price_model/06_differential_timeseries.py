"""
Steg 7: Tidsserie-analyse av differensialer per felt.

Spørsmål vi prøver å svare på:
  - Er differensialene stabile over tid, eller drifter de?
  - Endret Trolls differensial seg etter at det ble del av BFOET i 2018?
  - Hvordan har Johan Sverdrup blitt priset siden oppstart i 2019?
  - Er det fellestrekk i hvordan kondensater (Åsgard, Gudrun) handles?

Vi lager:
  1. Linjeplott av differensial over tid for utvalgte 'highlight'-felt.
  2. Rullerende 6-måneders snitt for å glatte støy.
  3. Heatmap (felt × år) for å se mønstre på tvers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

HIGHLIGHT_FIELDS = [
    "EKOFISK", "OSEBERG", "TROLL",        # BFOET
    "JOHAN SVERDRUP",                      # ny + medium sour
    "ÅSGARD", "GUDRUN",                    # kondensater
    "HEIDRUN", "GRANE",                    # tunge / høy TAN
]


def main() -> None:
    diff = pd.read_csv(DIFF_CSV)
    diff["field"] = diff["field"].str.upper().str.strip()

    # Bygg en ekte dato-kolonne — første dag i den måneden differensialen gjelder.
    diff["date"] = pd.to_datetime(
        dict(year=diff["year"], month=diff["month"], day=1),
        errors="coerce",
    )
    diff = diff.dropna(subset=["date"]).sort_values(["field", "date"]).reset_index(drop=True)

    print(f"Datoer: {diff['date'].min().date()} til {diff['date'].max().date()}")
    print(f"Felt: {diff['field'].nunique()}")

    # === Plott 1: linjer for utvalgte felt + 6m rullerende ===
    fig, ax = plt.subplots(figsize=(13, 7))
    palette = plt.cm.tab10.colors

    for i, field in enumerate(HIGHLIGHT_FIELDS):
        sub = diff[diff["field"] == field].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("date")
        # 6-måneders rullerende snitt for å glatte ut sesong/støy.
        sub["roll6"] = sub["differential_usd"].rolling(window=6, min_periods=3).mean()

        color = palette[i % len(palette)]
        # Tynn rå-linje + tjukk rullerende snitt.
        ax.plot(sub["date"], sub["differential_usd"],
                color=color, alpha=0.25, linewidth=0.8)
        ax.plot(sub["date"], sub["roll6"],
                color=color, linewidth=2.2, label=field.title())

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    # 2018 = BFOET utvidet med Troll
    ax.axvline(pd.Timestamp("2018-01-01"), color="gray", linestyle=":", alpha=0.6)
    ax.text(pd.Timestamp("2018-02-01"), ax.get_ylim()[1] * 0.92,
            "Troll inn i BFOET", fontsize=9, color="gray")
    # 2019 oktober = Sverdrup oppstart
    ax.axvline(pd.Timestamp("2019-10-01"), color="gray", linestyle=":", alpha=0.6)
    ax.text(pd.Timestamp("2019-11-01"), ax.get_ylim()[1] * 0.85,
            "Sverdrup oppstart", fontsize=9, color="gray")

    ax.set_xlabel("Dato")
    ax.set_ylabel("Differensial mot Brent (USD/fat)")
    ax.set_title("Normpris-differensialer 2012–2025\n"
                 "tynn linje = månedlig, tjukk = 6m rullerende snitt")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out1 = OUT_DIR / "06_differential_timeseries.png"
    fig.savefig(out1, dpi=130)
    print(f"Plott 1 lagret: {out1}")

    # === Plott 2: heatmap (felt × år) ===
    annual = (
        diff.groupby(["field", "year"])["differential_usd"]
        .mean()
        .unstack("year")
    )
    # Sorter felt etter snitt over hele perioden — stiger fra rabatt til premium.
    annual = annual.loc[annual.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(13, 8))
    vmax = max(abs(annual.min().min()), abs(annual.max().max()))
    im = ax.imshow(annual.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(annual.index)))
    ax.set_yticklabels([f.title() for f in annual.index])
    ax.set_xticks(range(len(annual.columns)))
    ax.set_xticklabels(annual.columns)
    ax.set_xlabel("År")
    ax.set_title("Årlig snitt-differensial mot Brent (USD/fat)\n"
                 "rødt = rabatt, grønt = premium, blankt = ingen data")

    # Skriv tall i hver celle.
    for i in range(annual.shape[0]):
        for j in range(annual.shape[1]):
            v = annual.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > vmax * 0.55 else "black")

    fig.colorbar(im, ax=ax, label="USD/fat")
    fig.tight_layout()
    out2 = OUT_DIR / "06_differential_heatmap.png"
    fig.savefig(out2, dpi=130)
    print(f"Plott 2 lagret: {out2}")

    # === Stabilitets-sjekk: standardavvik over tid per felt ===
    stability = (
        diff.groupby("field")
        .agg(mean_diff=("differential_usd", "mean"),
             std_diff=("differential_usd", "std"),
             n_obs=("differential_usd", "size"))
        .sort_values("std_diff", ascending=False)
        .round(2)
    )
    print("\n=== Stabilitet (høyt std-avvik = volatil differensial) ===")
    print(stability.to_string())


if __name__ == "__main__":
    main()
