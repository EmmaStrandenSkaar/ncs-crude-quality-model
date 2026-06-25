"""
Script 51 — Per-grade sensitivitetsanalyse: hvor mye påvirker produkt-marginer
            (crack spreads) prisen på hver enkelt crude?

METODE:
  For en crude med yield-profil (naphtha, middle distillate, vac.resid etc.),
  beregner vi den totale sensitiviteten til en +1 USD/bbl-endring i hver
  crack-spread. Dette inkluderer både:
    · Standalone-effekten av crack-spreaden
    · Interaksjons-effekten (yield × crack)

  Eksempel: for diesel crack
    sensitivity = β_diesel + β_md_x_diesel × middle_dist
                           + β_vr_x_diesel × vac_resid
                           + β_dg_x_diesel × diesel_gasoil

  Tolkning: hvis diesel crack stiger med 5 USD/bbl, endrer grade-prisen seg
  med sensitivity × 5 USD/bbl.

OUTPUT:
  data/processed/51_grade_product_sensitivity.csv
  Tabell per Brent-linked grade × hver produkt-driver, inkludert dollar-effekt
  per +5 USD/bbl crack-move (typisk månedlig svingning).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
MODEL_JSON   = PROC_DIR / "34b_brent_model.json"
PANEL_CSV    = PROC_DIR / "regression_panel.csv"
OUT_CSV      = PROC_DIR / "51_grade_product_sensitivity.csv"


# Typisk svingning per crack-spread (én standardavvik over 2020–2026)
TYPICAL_MOVE_USD = {
    "diesel_crack":              5.0,
    "gasoline_crack":            4.0,
    "diesel_minus_gasoline":     3.0,
}


def coef(model: dict, name: str) -> float:
    """Hent koeffisient (0 hvis feature ikke er i modellen)."""
    return model["coefficients"].get(name, 0.0)


def compute_sensitivity(grade_row: pd.Series, model: dict) -> dict:
    """
    Beregn USD/bbl-endring i grade-differensialet per +1 USD/bbl-endring
    i hver crack-spread, gitt grade-ens yield-profil.

    Totaleffekt = standalone + sum(yield × interaction)
    """
    yields = {
        "naphtha":          grade_row.get("naphtha_pct", 0),
        "kerosene":         grade_row.get("kerosene_pct", 0),
        "diesel_gasoil":    grade_row.get("diesel_gasoil_pct", 0),
        "middle_distillate":grade_row.get("middle_distillate_pct", 0),
        "vacuum_resid":     grade_row.get("vacuum_resid_pct", 0),
    }

    # ── Sensitivitet til DIESEL CRACK ───────────────────────────────────────
    # standalone + middle_dist × diesel + vac_resid × diesel + diesel_gasoil × diesel
    sens_diesel = (
        coef(model, "diesel_crack_brent")
        + coef(model, "middle_dist_x_diesel_crack")  * yields["middle_distillate"]
        + coef(model, "vacuum_resid_x_diesel_crack") * yields["vacuum_resid"]
        + coef(model, "diesel_x_diesel_crack")       * yields["diesel_gasoil"]
    )

    # ── Sensitivitet til GASOLINE CRACK ─────────────────────────────────────
    # standalone + naphtha × gasoline
    sens_gasoline = (
        coef(model, "gasoline_crack_brent")
        + coef(model, "naphtha_x_gasoline_crack") * yields["naphtha"]
    )

    # ── Sensitivitet til DIESEL–GASOLINE SPREAD ─────────────────────────────
    # Bare standalone — fanger preferanse for distillates vs gasoline
    sens_dg_spread = coef(model, "diesel_minus_gasoline_crack")

    # ── Sensitivitet til JET CRACK ──────────────────────────────────────────
    # kerosene × jet_crack ble eliminert, men vi tester jet-effekten via
    # diesel-corr (jet og diesel cracks korrelerer ~0.85)
    sens_jet = (
        coef(model, "jet_crack_brent")
        + coef(model, "kerosene_x_jet_crack") * yields["kerosene"]
    )

    return {
        "grade":              grade_row["grade"],
        "api_gravity":        grade_row.get("api_gravity"),
        "sulfur_pct":         grade_row.get("sulfur_pct"),
        "naphtha_pct":        yields["naphtha"],
        "kerosene_pct":       yields["kerosene"],
        "middle_distillate_pct": yields["middle_distillate"],
        "vacuum_resid_pct":   yields["vacuum_resid"],
        # Sensitiviteter per +1 USD/bbl crack-endring
        "sens_diesel":        sens_diesel,
        "sens_gasoline":      sens_gasoline,
        "sens_jet":           sens_jet,
        "sens_dg_spread":     sens_dg_spread,
        # Per typisk månedlig svingning
        "impact_diesel_per_5usd":   sens_diesel   * TYPICAL_MOVE_USD["diesel_crack"],
        "impact_gasoline_per_4usd": sens_gasoline * TYPICAL_MOVE_USD["gasoline_crack"],
    }


def main() -> None:
    print("=" * 75)
    print("  SCRIPT 51: Per-grade sensitivitet til produkt-marginer")
    print("=" * 75)

    model = json.loads(MODEL_JSON.read_text())
    print(f"\nModell: {model['model_name']}")
    print(f"OOT R²: {model['metrics']['r2_oot']:.3f}, RMSE: {model['metrics']['rmse']:.2f}")

    # Last unike grades med yield-data
    panel = pd.read_csv(PANEL_CSV)
    # Bruk siste tilgjengelige observasjon per grade (yields er statiske)
    last_per_grade = (panel.sort_values("date_str")
                           .groupby("grade")
                           .last()
                           .reset_index())

    # Filtrer til Brent-linkede grades (modellens treningsunivers)
    brent_grades = set(model["grades"])
    last_per_grade = last_per_grade[last_per_grade["grade"].isin(brent_grades)]
    print(f"\nGrades med yield-data: {len(last_per_grade)} (Brent-linked)")

    # Beregn sensitiviteter
    print("\n[1] Beregner sensitiviteter per grade...")
    results = [compute_sensitivity(row, model) for _, row in last_per_grade.iterrows()]
    df = pd.DataFrame(results).sort_values("impact_diesel_per_5usd", ascending=False)

    # ── Skriv ut topp/bunn per produkt-driver ───────────────────────────────
    print(f"\n{'='*75}")
    print(f"  DIESEL CRACK-sensitivitet (+5 USD/bbl crack-move → grade-pris-endring)")
    print(f"{'='*75}")
    print(f"  {'Grade':<22} {'mid_dist':>8} {'vac_resid':>9} "
          f"{'sens/+1$':>9} {'Δ per +5$':>10}")
    print(f"  {'-'*70}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['grade']:<22} {r['middle_distillate_pct']:>7.1f}% "
              f"{r['vacuum_resid_pct']:>8.1f}% "
              f"{r['sens_diesel']:>+9.4f} {r['impact_diesel_per_5usd']:>+10.2f}")
    print(f"  ... ({len(df) - 20} mellom-grades)")
    for _, r in df.tail(10).iterrows():
        print(f"  {r['grade']:<22} {r['middle_distillate_pct']:>7.1f}% "
              f"{r['vacuum_resid_pct']:>8.1f}% "
              f"{r['sens_diesel']:>+9.4f} {r['impact_diesel_per_5usd']:>+10.2f}")

    print(f"\n{'='*75}")
    print(f"  GASOLINE CRACK-sensitivitet (+4 USD/bbl crack-move)")
    print(f"{'='*75}")
    df_g = df.sort_values("impact_gasoline_per_4usd", ascending=False)
    print(f"  {'Grade':<22} {'naphtha':>8} {'sens/+1$':>9} {'Δ per +4$':>10}")
    print(f"  {'-'*55}")
    for _, r in df_g.head(8).iterrows():
        print(f"  {r['grade']:<22} {r['naphtha_pct']:>7.1f}% "
              f"{r['sens_gasoline']:>+9.4f} {r['impact_gasoline_per_4usd']:>+10.2f}")
    print(f"  ...")
    for _, r in df_g.tail(5).iterrows():
        print(f"  {r['grade']:<22} {r['naphtha_pct']:>7.1f}% "
              f"{r['sens_gasoline']:>+9.4f} {r['impact_gasoline_per_4usd']:>+10.2f}")

    # ── Konkrete scenarioer for NCS-feltene ─────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  NCS-EKSEMPLER (relevant for AKRBP/Equinor)")
    print(f"{'='*75}")
    ncs_examples = ["Alvheim", "Johan Sverdrup", "Ekofisk", "Grane", "Skarv",
                     "Heidrun", "Oseberg", "Statfjord"]
    print(f"  {'Grade':<22} {'API':>5} {'mid_d':>6} {'naph':>6} "
          f"{'Δ +5$ diesel':>13} {'Δ +4$ gasolin':>14}")
    print(f"  {'-'*72}")
    for g in ncs_examples:
        sub = df[df["grade"] == g]
        if len(sub):
            r = sub.iloc[0]
            print(f"  {r['grade']:<22} {r['api_gravity']:>5.1f} "
                  f"{r['middle_distillate_pct']:>5.1f}% "
                  f"{r['naphtha_pct']:>5.1f}% "
                  f"{r['impact_diesel_per_5usd']:>+13.2f} "
                  f"{r['impact_gasoline_per_4usd']:>+14.2f}")

    # ── Lagre ────────────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  ✓ Lagret: {OUT_CSV.name}")
    print(f"  Bruk i script 52 for heatmap-visualisering.")


if __name__ == "__main__":
    main()
