"""
Steg 2: Hent Brent spot-pris (FOB Europe) fra EIA og lagre som ryddig CSV.

Datakilde:
  EIA - U.S. Energy Information Administration
  Serie: RBRTE (Europe Brent Spot Price FOB, USD/fat)
  URL:   https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls
  Frekvens: daglig, fra 1987-05-20 til i dag.

Hva scriptet gjør:
  1. Laster ned XLS-fila fra EIA (kun hvis vi ikke har den lokalt fra før).
  2. Leser arket "Data 1" hvor selve prisene ligger.
  3. Rydder opp: gir kolonnene fornuftige navn, gjør Date til ekte datoer.
  4. Lagrer en ren CSV i data/raw/brent_spot_eia.csv.
  5. Plotter hele prishistorikken og lagrer PNG i data/processed/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"
LOCAL_XLS = RAW_DIR / "eia_brent_RBRTEd.xls"
OUTPUT_CSV = RAW_DIR / "brent_spot_eia.csv"


def download_if_missing(url: str, dest: Path) -> None:
    """Last ned URL til dest, men bare hvis fila ikke finnes fra før.
    Slik slipper vi å banke på EIA hver gang vi kjører scriptet."""
    if dest.exists():
        print(f"Bruker lokal kopi: {dest.name}")
        return
    print(f"Laster ned fra EIA: {url}")
    # requests henter URL-en; raise_for_status() kaster feil hvis HTTP ikke er 200 OK.
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Lagret {len(response.content):,} bytes til {dest}")


def load_brent(xls_path: Path) -> pd.DataFrame:
    """Les Brent-prisene ut av EIA-XLS-fila og returner en ryddig DataFrame.

    EIA-arket har to rader med metadata på toppen ('Sourcekey' + kolonnenavn),
    så vi hopper over dem med skiprows=2 og setter egne kolonnenavn.
    """
    df = pd.read_excel(
        xls_path,
        sheet_name="Data 1",
        engine="xlrd",
        skiprows=2,                  # hopp over de to metadata-radene
        names=["date", "brent_usd"], # gi kolonnene rene navn
    )
    # Konverter til ekte typer. errors='coerce' gjør ubrukelige rader til NaT/NaN
    # som vi så fjerner med dropna(). Trygt mot evt. tomme rader nederst i arket.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["brent_usd"] = pd.to_numeric(df["brent_usd"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def plot_brent(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df["brent_usd"], color="darkred", linewidth=0.8)
    ax.set_xlabel("Dato")
    ax.set_ylabel("Brent spot (USD/fat)")
    ax.set_title("Brent spot-pris (EIA, FOB Europe)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Plott lagret: {out_path}")


def main() -> None:
    download_if_missing(EIA_URL, LOCAL_XLS)
    brent = load_brent(LOCAL_XLS)

    print(f"\nAntall observasjoner: {len(brent):,}")
    print(f"Periode: {brent['date'].min().date()} til {brent['date'].max().date()}")
    print(f"Pris-spenn: {brent['brent_usd'].min():.2f} - {brent['brent_usd'].max():.2f} USD/fat")
    print("\nFørste 3 rader:")
    print(brent.head(3).to_string(index=False))
    print("\nSiste 3 rader:")
    print(brent.tail(3).to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    brent.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRyddig CSV lagret: {OUTPUT_CSV}")

    plot_brent(brent, PROCESSED_DIR / "02_brent_history.png")


if __name__ == "__main__":
    main()
