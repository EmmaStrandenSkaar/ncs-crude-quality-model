"""
Script 52 — Heatmap: per-grade sensitivitet til produkt-marginer (presentasjon)

Visualiserer hvordan endringer i raffinerings-marginer (crack spreads)
påvirker prisen på hver enkelt crude-grade.

LAYOUT:
  Vertikal akse: 24 Brent-linkede grades (gruppert etter region)
  Horisontal akse: 3 produkt-drivere (diesel crack, gasoline crack, diesel-gas spread)
  Celleinnhold: USD/bbl effekt per typisk crack-move
  Farge: rød = grade taper, grønn = grade vinner

Designet for IB/ER-presentasjon.

OUTPUT:
  data/processed/52_product_sensitivity_heatmap.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).parent.parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
SENS_CSV     = PROC_DIR / "51_grade_product_sensitivity.csv"
OUT_PNG      = PROC_DIR / "52_product_sensitivity_heatmap.png"

# Region-mapping for sortering
REGION_MAP = {
    # NCS
    "Alvheim": "NCS", "Johan Sverdrup": "NCS", "Ekofisk": "NCS",
    "Grane": "NCS", "Skarv": "NCS", "Heidrun": "NCS", "Oseberg": "NCS",
    "Statfjord": "NCS", "Troll": "NCS", "Gullfaks": "NCS", "Norne": "NCS",
    "Asgard": "NCS", "Gudrun": "NCS", "Goliat": "NCS", "Gina Krog": "NCS",
    "Jotun": "NCS", "Knarr": "NCS", "Martin Linge": "NCS", "Njord": "NCS",
    "Draugen": "NCS", "Balder": "NCS",
    # Midtøsten
    "Dubai Fateh": "Middle East", "Arab Light": "Middle East",
    "Arab Medium": "Middle East", "Arab Extra Light": "Middle East",
    "Basrah Light": "Middle East",
    # Vest-Afrika
    "Bonny Light": "West Africa", "Forcados": "West Africa",
    "Cabinda": "West Africa", "Qua Iboe": "West Africa", "Rabi Light": "West Africa",
    # Nord-Afrika
    "Saharan Blend": "North Africa",
}

REGION_COLORS = {
    "NCS":         "#1A5276",
    "Middle East": "#C0392B",
    "West Africa": "#D4720A",
    "North Africa": "#7E5109",
}


def make_diverging_cmap():
    """Tilpasset rød-grønn divergerende fargeskala (rabatt/premium)."""
    return LinearSegmentedColormap.from_list(
        "redgreen",
        ["#A93226", "#E67E22", "#F8F8F8", "#52BE80", "#1E8449"],
        N=256,
    )


def main():
    df = pd.read_csv(SENS_CSV)

    # Legg til region og sorter
    df["region"] = df["grade"].map(REGION_MAP).fillna("Other")
    region_order = ["NCS", "West Africa", "North Africa", "Middle East", "Other"]
    df["region_rank"] = df["region"].apply(
        lambda r: region_order.index(r) if r in region_order else 99
    )
    df = df.sort_values(["region_rank", "impact_diesel_per_5usd"],
                         ascending=[True, False]).reset_index(drop=True)

    # ── Bygg sensitivity-matrise ────────────────────────────────────────────
    # Kolonner: tre crack-drivere, normalisert til typisk månedlig svingning
    impact_cols = {
        "Diesel crack\n(+5 USD/bbl)":       df["impact_diesel_per_5usd"].values,
        "Gasoline crack\n(+4 USD/bbl)":     df["impact_gasoline_per_4usd"].values,
        "Diesel–gasoline\nspread (+3 USD)": df["sens_dg_spread"].values * 3.0,
    }
    matrix = np.column_stack(list(impact_cols.values()))
    grades = df["grade"].tolist()
    regions = df["region"].tolist()

    # Symmetrisk fargeskala basert på maks abs-verdi
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    vmax = round(vmax * 2) / 2  # rund opp til nærmeste 0.5

    # ── Plot ────────────────────────────────────────────────────────────────
    n_rows = len(grades)
    n_cols = len(impact_cols)
    fig_h  = max(8, n_rows * 0.32)
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor="white")

    cmap = make_diverging_cmap()
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax,
                    aspect="auto", interpolation="none")

    # Annoter hver celle med dollar-verdi
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            txt_color = "white" if abs(v) > vmax * 0.55 else "#222"
            sign = "+" if v > 0 else ""
            ax.text(j, i, f"{sign}{v:.2f}",
                     ha="center", va="center", fontsize=9,
                     color=txt_color, family="monospace", fontweight="bold")

    # X-akse: produkt-drivere
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(list(impact_cols.keys()), fontsize=10, fontweight="bold")
    ax.xaxis.tick_top()

    # Y-akse: grades med region-fargete labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(grades, fontsize=9)
    for tick, region in zip(ax.get_yticklabels(), regions):
        tick.set_color(REGION_COLORS.get(region, "#333"))

    # Region-grenser (horisontale linjer)
    prev_region = None
    for i, r in enumerate(regions):
        if prev_region is not None and r != prev_region:
            ax.axhline(i - 0.5, color="#222", lw=1.2, alpha=0.6)
        prev_region = r

    # Spines og grid
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, shrink=0.5)
    cbar.set_label("USD/bbl effekt på differensial vs. Brent",
                    fontsize=9, color="#444")
    cbar.ax.tick_params(labelsize=8, colors="#444")

    # Tittel
    fig.suptitle(
        "Crude-grade sensitivitet til produkt-etterspørsel\n"
        "Forventet prisendring per typisk svingning i raffinerings-marginer",
        fontsize=13, fontweight="bold", color="#1A1A2E", y=0.995
    )

    # Region-legend nederst
    region_patches = [
        mpatches.Patch(facecolor=c, label=r)
        for r, c in REGION_COLORS.items()
        if r in regions
    ]
    fig.legend(
        handles=region_patches,
        loc="lower center", bbox_to_anchor=(0.5, -0.005),
        ncol=4, fontsize=9, framealpha=0.95,
        edgecolor="#cccccc", title="Region (farge på grade-label)",
        title_fontsize=9,
    )

    # Fotnote
    fig.text(
        0.02, 0.018,
        "Metode: per-grade sensitivitet beregnet fra OLS-koeffisientene (standalone crack + yield × crack-interaksjoner) "
        "i Brent-linked modellen (32 grades, OOT R²=0.33).\n"
        "Tolkning: 'Diesel crack +5 USD' viser hvor mye en grade endrer pris hvis diesel-marginen øker 5 USD/bbl "
        "(typisk månedlig svingning).\n"
        "Eksempel: Bonny Light vinner +1.24 USD/bbl per +5 USD diesel crack pga. høy middle distillate-yield (54.9%). "
        "Gudrun mister −0.87 USD/bbl per +4 USD gasoline crack pga. høy naphtha (52.9%).",
        fontsize=7.5, color="#777", style="italic", wrap=True
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figur lagret: {OUT_PNG}")

    # ── Print interpretasjon ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  NØKKELFUNN — direkte til presentasjonen")
    print(f"{'='*70}")

    print("\n▶ DIESEL DEMAND-VINNERE (krudere med høyt mid-distillate-yield):")
    top_diesel = df.nlargest(5, "impact_diesel_per_5usd")
    for _, r in top_diesel.iterrows():
        print(f"  • {r['grade']:<20} ({r['region']:<12}) "
              f"middle-dist {r['middle_distillate_pct']:.1f}% → "
              f"+{r['impact_diesel_per_5usd']:.2f} USD/bbl per +5 USD diesel crack")

    print("\n▶ GASOLINE DEMAND-TAPERE (lette crudes med høyt naphtha-yield):")
    bot_gas = df.nsmallest(5, "impact_gasoline_per_4usd")
    for _, r in bot_gas.iterrows():
        print(f"  • {r['grade']:<20} ({r['region']:<12}) "
              f"naphtha {r['naphtha_pct']:.1f}% → "
              f"{r['impact_gasoline_per_4usd']:.2f} USD/bbl per +4 USD gasoline crack")

    # AKRBP-spesifikk lesning
    print("\n▶ AKER BP PORTFOLIO-IMPLIKASJONER:")
    akrbp_grades = ["Alvheim", "Johan Sverdrup", "Skarv", "Ekofisk", "Grane"]
    print(f"  Hvis IEA hever diesel/gasoil-prognosen sin (typisk +5 USD/bbl crack-move),")
    print(f"  hvilke AKRBP-relevante grades vinner mest?")
    for g in akrbp_grades:
        sub = df[df["grade"] == g]
        if len(sub):
            r = sub.iloc[0]
            print(f"    {g:<18} Δ diesel +5$: {r['impact_diesel_per_5usd']:+.2f} | "
                  f"Δ gasoline +4$: {r['impact_gasoline_per_4usd']:+.2f}")


if __name__ == "__main__":
    main()
