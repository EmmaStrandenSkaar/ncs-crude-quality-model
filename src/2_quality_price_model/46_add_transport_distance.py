"""
Script 46 — Legg til kontinuerlig transportavstandsvariabel i regresjons-panelet.

Problemet med eksisterende modell: bruker kun grove kategorier (short/medium/long)
og kategoriserer WTI, Maya, Merey som "short" — samme kategori som NCS-olje!

Løsning: Kontinuerlig avstand (km) fra lastehavn til Rotterdam (verdens største
raffineringshub og Brent-referansepunkt). Beregnes med haversine-formel.

Rotterdam: 51.9°N, 4.5°E

Lastehavner basert på:
  - NCS: offisielle FPSO-posisjoner og terminal-koordinater fra Equinor/Sodir
  - Midtøsten: Ras Tanura, Basrah (Al Faw), Fateh terminal
  - Vest-Afrika: offisielle eksportterminaler per felt
  - Amerika: lastehavner for eksport til Rotterdam-markedet
  - Nord-Afrika: Bejaia (Saharan Blend)

Merk: Haversine = stor-sirkel-avstand (rett linje). Faktisk sjøruteAvstand
er høyere (Suez, Panama, Kapp det gode håp), men haversine gir riktig
relativ rangering og korrelerer sterkt med faktiske fraktrater.
"""

from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_CSV    = PROJECT_ROOT / "data" / "processed" / "regression_panel.csv"

# Rotterdam: referansepunkt
ROTTERDAM_LAT, ROTTERDAM_LON = 51.9, 4.5


def haversine_km(lat1: float, lon1: float,
                 lat2: float = ROTTERDAM_LAT,
                 lon2: float = ROTTERDAM_LON) -> float:
    """Stor-sirkel-avstand (km) mellom to koordinater."""
    R = 6_371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# Lastehavn-koordinater per grade
# Kilde: FPSO-posisjoner (Sodir/Equinor), terminalpublikasjoner, ACSA, Platts
# ---------------------------------------------------------------------------
LOADING_PORTS: dict[str, tuple[float, float, str]] = {
    # ── NCS (laster fra FPSO eller fastlandsterminal) ──────────────────────
    "Alvheim":       (57.04,  2.51,  "Alvheim FPSO"),
    "Asgard":        (59.30,  5.24,  "Kårstø terminal"),          # pipeline til Kårstø
    "Balder":        (59.14,  2.28,  "Balder FPSO"),
    "Draugen":       (64.36,  7.78,  "Draugen FPSO"),
    "Ekofisk":       (56.53,  3.22,  "Ekofisk SBM / Teesside"),
    "Gina Krog":     (58.26,  1.91,  "Gina Krog FPSO"),
    "Goliat":        (71.28, 22.52,  "Goliat FPSO"),              # Barentshavet
    "Grane":         (60.53,  4.79,  "Sture terminal"),
    "Gudrun":        (58.19,  1.81,  "Sleipner A (til Kårstø)"),
    "Gullfaks":      (60.82,  5.03,  "Mongstad"),
    "Heidrun":       (65.32,  7.32,  "Heidrun TLP"),
    "Johan Sverdrup": (59.50, 4.79,  "Sture terminal"),
    "Jotun":         (58.73,  1.89,  "Jotun FPSO"),
    "Knarr":         (60.76,  2.79,  "Knarr FPSO"),
    "Martin Linge":  (60.50,  2.10,  "Martin Linge FPSO"),
    "Njord":         (64.73,  8.16,  "Njord A FPSO"),
    "Norne":         (66.02,  8.10,  "Norne FPSO"),
    "Oseberg":       (60.53,  4.79,  "Sture terminal"),
    "Skarv":         (65.72,  7.05,  "Skarv FPSO"),
    "Statfjord":     (61.25,  1.85,  "Statfjord A/B/C SBM"),
    "Troll":         (60.82,  5.03,  "Mongstad"),
    # ── Midtøsten ──────────────────────────────────────────────────────────
    "Arab Extra Light": (26.64, 50.16, "Ras Tanura"),
    "Arab Light":       (26.64, 50.16, "Ras Tanura"),
    "Arab Medium":      (26.64, 50.16, "Ras Tanura"),
    "Basrah Light":     (29.78, 48.57, "Al Faw terminal"),
    "Dubai Fateh":      (25.01, 55.22, "Fateh terminal"),
    # ── Vest-Afrika ────────────────────────────────────────────────────────
    "Bonny Light":   ( 4.44,   7.15,  "Bonny terminal, Nigeria"),
    "Forcados":      ( 5.36,   5.32,  "Forcados terminal, Nigeria"),
    "Cabinda":       (-5.58,  12.19,  "Malongo terminal, Angola"),
    "Qua Iboe":      ( 4.59,   7.95,  "Qua Iboe terminal, Nigeria"),
    "Rabi Light":    (-1.92,  10.13,  "Cap Lopez, Gabon"),
    "Leona":         ( 4.44,   7.15,  "Bonny terminal, Nigeria"),   # eksporteres via Nigeria
    # ── Nord-Afrika ────────────────────────────────────────────────────────
    "Saharan Blend": (36.75,   5.09,  "Bejaia, Algeria"),
    # ── Latin-Amerika ──────────────────────────────────────────────────────
    # Venezuela og Ecuador: eksport til USGC og Europa via Atlanterhavet
    "Merey":         (10.15, -64.69,  "Jose terminal, Venezuela"),
    "Napo":          (-0.07, -78.58,  "Esmeraldas, Ecuador"),
    "Oriente":       (-0.07, -78.58,  "Esmeraldas, Ecuador"),
    "Marlim":        (-22.00,-40.00,  "Campos basin offshore, Brasil"),
    # ── Nord-Amerika ───────────────────────────────────────────────────────
    # WTI og kanadiske grades: laster fra US Gulf Coast for eksport til Europa
    "WTI":                  (29.73, -93.33, "Beaumont/Port Arthur TX"),
    "Bow River Heavy":      (29.73, -93.33, "US Gulf Coast (pipeline via Enbridge)"),
    "Canadian Light Sour":  (29.73, -93.33, "US Gulf Coast (pipeline via Enbridge)"),
    "Lloydminster":         (29.73, -93.33, "US Gulf Coast (pipeline via Enbridge)"),
    # ── Mexico ─────────────────────────────────────────────────────────────
    "Maya":   (18.14, -94.41, "Pajaritos / Dos Bocas, Mexico"),
    "Olmeca": (18.14, -94.41, "Pajaritos / Dos Bocas, Mexico"),
}


def main():
    df = pd.read_csv(PANEL_CSV)
    print(f"Panel lastet: {df.shape[0]} rader, {df['grade'].nunique()} grades")

    # Sjekk alle grades er dekket
    grades = df["grade"].unique()
    missing = [g for g in grades if g not in LOADING_PORTS]
    if missing:
        print(f"\n⚠️  Mangler lastehavn for: {missing}")
        return

    # Beregn avstand til Rotterdam
    port_data = {
        g: {
            "lat": LOADING_PORTS[g][0],
            "lon": LOADING_PORTS[g][1],
            "port_name": LOADING_PORTS[g][2],
            "dist_rotterdam_km": haversine_km(LOADING_PORTS[g][0], LOADING_PORTS[g][1]),
        }
        for g in LOADING_PORTS
        if g in grades
    }

    # Print sammenligning
    print("\n=== Avstand fra lastehavn til Rotterdam (km) ===")
    print(f"{'Grade':<25} {'Havn':<40} {'km':>6}  Gammel band")
    print("-" * 80)
    port_df = pd.DataFrame(port_data).T.sort_values("dist_rotterdam_km")
    grade_bands = df.groupby("grade")["distance_band"].first()
    for g, row in port_df.iterrows():
        band = grade_bands.get(g, "?")
        print(f"  {g:<23} {row['port_name']:<40} {row['dist_rotterdam_km']:>6.0f}  [{band}]")

    # Legg til i panel
    df["dist_rotterdam_km"] = df["grade"].map(
        {g: d["dist_rotterdam_km"] for g, d in port_data.items()}
    )
    # Log-versjon (bedre for lineær regresjon siden fraktrater er ikke-lineære)
    df["log_dist_rotterdam"] = np.log(df["dist_rotterdam_km"])

    # Normaliser (z-score)
    mu    = df["dist_rotterdam_km"].mean()
    sigma = df["dist_rotterdam_km"].std()
    df["dist_rotterdam_zscore"] = (df["dist_rotterdam_km"] - mu) / sigma

    print(f"\nStatistikk:")
    print(f"  Min:    {df['dist_rotterdam_km'].min():.0f} km  ({df.loc[df['dist_rotterdam_km'].idxmin(),'grade']})")
    print(f"  Max:    {df['dist_rotterdam_km'].max():.0f} km  ({df.loc[df['dist_rotterdam_km'].idxmax(),'grade']})")
    print(f"  Gj.sn.: {mu:.0f} km")
    print(f"  Std:    {sigma:.0f} km")

    df.to_csv(PANEL_CSV, index=False)
    print(f"\n✓ Panel oppdatert: {PANEL_CSV}")
    print(f"  Nye kolonner: dist_rotterdam_km, log_dist_rotterdam, dist_rotterdam_zscore")


if __name__ == "__main__":
    main()
