"""
Steg 1: Les inn kvalitetsdata for de fire nordsjøfeltene og se hva vi har.

Hva scriptet gjør:
  1. Leser CSV-fila med kvalitetsdata.
  2. Skriver ut tabellen.
  3. Lager en enkel scatter-plott: API-grad mot svovelinnhold, ett punkt per felt.
  4. Lagrer plottet som PNG i data/processed/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Path(__file__) er denne fila. .parent gir mappa fila ligger i (scripts/).
# .parent en gang til gir prosjektroten. Slik finner vi data/ uavhengig
# av hvor scriptet kjøres fra.
PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_CSV = PROJECT_ROOT / "data" / "raw" / "crude_quality.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# pandas leser CSV inn i en "DataFrame" — tenk på det som et regneark i minnet.
quality = pd.read_csv(QUALITY_CSV)

print("=== Kvalitetsdata ===")
print(quality.to_string(index=False))
print()
print(f"Antall felt: {len(quality)}")
print(f"API-grad spenn: {quality['api_gravity'].min()} - {quality['api_gravity'].max()}")
print(f"Svovel spenn:   {quality['sulfur_pct'].min()}% - {quality['sulfur_pct'].max()}%")

# Enkelt scatter-plott. Hvert felt blir ett punkt.
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(quality["api_gravity"], quality["sulfur_pct"], s=120, color="steelblue")

# Skriv navnet på feltet ved siden av hvert punkt.
for _, row in quality.iterrows():
    ax.annotate(
        row["field"],
        (row["api_gravity"], row["sulfur_pct"]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=10,
    )

ax.set_xlabel("API-grad (høyere = lettere olje)")
ax.set_ylabel("Svovelinnhold (%) — høyere = surere olje")
ax.set_title("Kvalitet: nordsjøfelt — API vs. svovel")
ax.grid(True, alpha=0.3)

# Konvensjon i bransjen: "sweet spot" er øverst-til-høyre på API-aksen
# og lavt på svovel — altså nede-til-høyre i denne plotten.
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Svovelgrense sweet/sour (0.5%)")
ax.legend()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / "01_quality_scatter.png"
fig.tight_layout()
fig.savefig(out_path, dpi=120)
print(f"\nPlott lagret: {out_path}")
