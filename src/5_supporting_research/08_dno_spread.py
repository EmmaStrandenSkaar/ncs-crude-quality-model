"""
DNO-feltanalyse: API vs svovel-spread for hele porteføljen.

Formål:
  Visuelt oppsummere DNOs eksponering mot oljekvalitet og identifisere
  hvilke deler av porteføljen som gir premium vs rabatt mot Brent —
  nyttig for å vurdere realiseringspris inn mot Q2-rapport.

Vi plotter:
  1. Scatter: API vs svovel, med punkt-størrelse = estimert netto-produksjon,
     fargekode = region (Kurdistan / Nordsjøen / Vest-Afrika).
  2. Stacked bar: estimert produksjon vektet etter kvalitets-segmenter.
  3. Tabell: predikert differensial per felt fra vår modell E.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DNO_CSV = PROJECT_ROOT / "data" / "raw" / "dno_fields.csv"
MODEL_JSON = PROJECT_ROOT / "data" / "model" / "model_E_bfoet.json"
DIFF_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

REGION_COLORS = {
    "Kurdistan": "#c0392b",    # rød
    "Nordsjoen": "#2980b9",    # blå
    "VestAfrika": "#27ae60",   # grønn
}

SWEET_SOUR_BOUNDARY = 0.50  # % svovel


def predict(row: pd.Series, coefs: dict) -> float:
    """Prediker differensial mot Brent basert på modell E."""
    api = row["api_gravity"]
    return (
        coefs["const"]
        + coefs["api_gravity"] * api
        + coefs["api2"] * api ** 2
        + coefs["sulfur_pct"] * row["sulfur_pct"]
        + coefs["is_bfoet"] * (1 if row.get("normpris_field") in ("EKOFISK", "OSEBERG", "TROLL") else 0)
    )


def add_actual_differentials(dno: pd.DataFrame, diff: pd.DataFrame) -> pd.DataFrame:
    """Hent faktisk snitt-differensial fra normpris-datasettet der vi har treff."""
    agg = diff.groupby("field")["differential_usd"].mean().reset_index()
    agg.columns = ["normpris_field", "actual_diff_usd"]
    return dno.merge(agg, on="normpris_field", how="left")


def main() -> None:
    dno = pd.read_csv(DNO_CSV)
    diff = pd.read_csv(DIFF_CSV)
    diff["field"] = diff["field"].str.upper().str.strip()

    model = json.loads((MODEL_JSON).read_text(encoding="utf-8"))
    coefs = model["coefficients"]

    dno["predicted_diff_usd"] = dno.apply(lambda r: predict(r, coefs), axis=1)
    dno = add_actual_differentials(dno, diff)

    # VIKTIG: modell E er trent på nordsjøfelt med svovel 0.04–0.80 %.
    # For Kurdistan (svovel 3–4 %) ekstrapolerer modellen galt.
    # Vi bruker markedsestimat for Kurdistan istedenfor modellprediksjoner.
    KURDISTAN_DISCOUNT = -8.0  # typisk $6–12 rabatt til Brent
    dno.loc[dno["region"] == "Kurdistan", "predicted_diff_usd"] = KURDISTAN_DISCOUNT

    # Velg "best" differensial — faktisk der vi har det, ellers predikert/manuell.
    dno["best_diff_usd"] = dno["actual_diff_usd"].fillna(dno["predicted_diff_usd"])
    dno["diff_source"] = np.where(
        dno["actual_diff_usd"].notna(), "normpris",
        np.where(dno["region"] == "Kurdistan", "markedsestimat", "modell E"),
    )

    # === Tabell ===
    print("=" * 90)
    print("DNO PORTEFØLJE — KVALITET OG ESTIMERT DIFFERENSIAL")
    print("=" * 90)
    cols = ["field", "region", "api_gravity", "sulfur_pct",
            "est_net_kbpd", "best_diff_usd", "diff_source", "confidence"]
    print(dno[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # Produksjonsvektet snitt-differensial per region.
    print("\n=== Produksjonsvektet snitt-differensial per region ===")
    for region in dno["region"].unique():
        sub = dno[dno["region"] == region]
        wtd_diff = np.average(sub["best_diff_usd"], weights=sub["est_net_kbpd"])
        total_kbpd = sub["est_net_kbpd"].sum()
        print(f"  {region:<15} {total_kbpd:>6.1f} kbpd   snitt diff: {wtd_diff:+.2f} USD/fat")
    # Samlet
    wtd_total = np.average(dno["best_diff_usd"], weights=dno["est_net_kbpd"])
    total = dno["est_net_kbpd"].sum()
    print(f"  {'TOTALT':<15} {total:>6.1f} kbpd   snitt diff: {wtd_total:+.2f} USD/fat")

    # === PLOTT 1: API vs svovel-spread ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for region, color in REGION_COLORS.items():
        sub = dno[dno["region"] == region]
        sizes = 40 + sub["est_net_kbpd"] * 8
        ax.scatter(sub["api_gravity"], sub["sulfur_pct"],
                   s=sizes, c=color, alpha=0.75, edgecolor="black", linewidth=0.6,
                   label=region, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(
                row["field"],
                (row["api_gravity"], row["sulfur_pct"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8,
            )

    # Referanselinjer.
    ax.axhline(SWEET_SOUR_BOUNDARY, color="gray", linestyle="--", alpha=0.5)
    ax.text(50, SWEET_SOUR_BOUNDARY + 0.08, "sweet/sour grense", fontsize=8, color="gray")

    # Skraver "sweet spot" for raffinerier (API 30-42, S < 0.5%).
    sweet_spot = plt.Rectangle((30, 0), 12, SWEET_SOUR_BOUNDARY,
                                alpha=0.08, color="green", zorder=1)
    ax.add_patch(sweet_spot)
    ax.text(31, 0.03, "Raffineri sweet spot", fontsize=8, color="darkgreen", alpha=0.7)

    ax.set_xlabel("API-grad (høyere = lettere)")
    ax.set_ylabel("Svovelinnhold (%)")
    ax.set_title("DNO: oljekvalitet per felt\n(størrelse = netto produksjon)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 58)
    ax.set_ylim(-0.05, 4.2)

    # === PLOTT 2: produksjonsfordeling per kvalitetssegment ===
    ax = axes[1]
    dno["quality_segment"] = pd.cut(
        dno["best_diff_usd"],
        bins=[-20, -2, 0, 2, 20],
        labels=["Stor rabatt\n(< -$2)", "Liten rabatt\n(-$2 til $0)",
                "Liten premium\n($0 til +$2)", "Stor premium\n(> +$2)"],
    )
    seg_prod = dno.groupby(["quality_segment", "region"])["est_net_kbpd"].sum().unstack(fill_value=0)
    seg_prod = seg_prod.reindex(columns=REGION_COLORS.keys(), fill_value=0)
    seg_prod.plot.barh(
        ax=ax, stacked=True,
        color=[REGION_COLORS[r] for r in seg_prod.columns],
        edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    ax.set_xlabel("Netto produksjon (kbpd)")
    ax.set_title("DNO: produksjon fordelt på pris-segment\n(basert på modell E / normpris)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.suptitle("DNO ASA — Kvalitet & Pris-eksponering (Q2 2026)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "08_dno_quality_spread.png"
    fig.savefig(out, dpi=130)
    print(f"\nPlott lagret: {out}")

    # Lagre tabell.
    out_csv = OUT_DIR / "08_dno_field_analysis.csv"
    dno.to_csv(out_csv, index=False)
    print(f"CSV lagret: {out_csv}")


if __name__ == "__main__":
    main()
