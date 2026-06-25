"""
Steg 4: Hent kvartalsvise normpris-XLSX fra regjeringen.no, parse
"Differensialer"-arket i hver fil, og bygg én ryddig CSV med differensialer
(felt × måned × USD/fat).

Hva en "differensial" er:
  Forskjellen mellom feltets normpris og Brent benchmark, i USD per fat.
  Positiv = premium (feltet handles dyrere enn Brent).
  Negativ = rabatt   (feltet handles billigere enn Brent).

Pipeline:
  1. Skrap arkiv-siden for å finne alle XLSX-lenker.
  2. Filtrer bort enkeltfelt-revisjoner — vi vil ha hovedfilene per kvartal.
  3. Last ned hver hovedfil (cachet lokalt — kjør på nytt = gratis).
  4. Parse "Differensialer"-arket: finn rad med månedsnavn, les felt+verdier.
  5. Konkatener til langformat: (field, year, month, differential_usd, ...).
  6. Skriv samlet CSV og rapporter dekning for våre fire mål-felt.
"""

from pathlib import Path
import re
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX_DIR = PROJECT_ROOT / "data" / "raw" / "normpris_xlsx"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"

ARCHIVE_PAGE = (
    "https://www.regjeringen.no/no/tema/energi/olje-og-gass/"
    "petroleumsprisradet-og-normprisene/id661459/"
)
BASE_URL = "https://www.regjeringen.no"

# Norske månedsnavn → månedsnummer. Brukes for å gjenkjenne kolonneoverskrifter.
MONTHS_NB = {
    "januar": 1, "februar": 2, "mars": 3,
    "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9,
    "oktober": 10, "november": 11, "desember": 12,
}

# Feltnavn vi gjenkjenner i URL-er — brukes for å filtrere bort enkeltfelt-
# revisjoner (f.eks. "np-1q-2024-endelige-normpriser-troll-februar.xlsx").
KNOWN_FIELDS_IN_URL = [
    "alvheim", "balder", "draugen", "ekofisk", "gina-krog", "goliat",
    "grane", "gudrun", "gullfaks", "heidrun", "johan-sverdrup", "knarr",
    "martin-linge", "norne", "oseberg", "skarv", "statfjord", "troll",
    "asgard", "yme", "njord", "valhall", "ula",
]

TARGET_FIELDS = ["JOHAN SVERDRUP", "VALHALL", "TROLL", "EKOFISK"]


def fetch_archive_links() -> list[str]:
    """Hent arkiv-siden og returner alle relative URL-er til XLSX-filer."""
    html = requests.get(ARCHIVE_PAGE, timeout=60).text
    pattern = r'href="([^"]*normpris[^"]*\.xlsx)"'
    return sorted(set(re.findall(pattern, html, flags=re.IGNORECASE)))


def is_main_quarterly_file(url: str) -> bool:
    """Behold hovedfiler (dekker alle felt). Filtrer bort feltspesifikke
    revisjoner og 'kopi-av'/'omgjoring' osv."""
    name = url.lower().rsplit("/", 1)[-1]
    if any(x in name for x in ["revider", "omgjoring", "kopi-av"]):
        return False
    if any(field in name for field in KNOWN_FIELDS_IN_URL):
        return False
    return True


def parse_quarter_and_year(url: str) -> tuple[int, int] | None:
    """Hent ut (kvartal, år) fra filnavnet. Tåler 'Nq', 'qN', og 'YYYY' i hvilken
    som helst rekkefølge."""
    name = url.lower().rsplit("/", 1)[-1]
    quarter = re.search(r"\b(\d)q\b|\bq(\d)\b", name)
    year = re.search(r"\b(20\d{2})\b", name)
    if not quarter or not year:
        return None
    q = int(next(g for g in quarter.groups() if g))
    if q < 1 or q > 4:
        return None
    return q, int(year.group(1))


def download_if_missing(full_url: str, dest: Path) -> bool:
    """Returnerer True hvis fila finnes lokalt etter kallet."""
    if dest.exists():
        return True
    try:
        response = requests.get(full_url, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"    nedlasting feilet: {e}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return True


def parse_differentials_sheet(xlsx_path: Path) -> pd.DataFrame:
    """Returner langformat: en rad per (felt, måned). Tom hvis arket ikke finnes
    eller ikke har gjenkjennbar struktur."""
    try:
        sheet = pd.read_excel(
            xlsx_path,
            sheet_name="Differensialer",
            header=None,
            engine="openpyxl",
        )
    except (ValueError, KeyError):
        return pd.DataFrame()

    # Finn raden som inneholder månedsnavn (f.eks. April / Mai / Juni).
    month_row_idx = None
    month_cols: dict[int, int] = {}
    for row_idx in range(len(sheet)):
        row_cols = {}
        for col_idx in range(sheet.shape[1]):
            cell = sheet.iat[row_idx, col_idx]
            if isinstance(cell, str) and cell.strip().lower() in MONTHS_NB:
                row_cols[col_idx] = MONTHS_NB[cell.strip().lower()]
        if row_cols:
            month_row_idx = row_idx
            month_cols = row_cols
            break

    if not month_cols:
        return pd.DataFrame()

    rows = []
    for row_idx in range(month_row_idx + 1, len(sheet)):
        # Hent feltnavn — første ikke-tomme strengkolonne som ikke er en
        # månedskolonne.
        field_name = None
        for col_idx in range(sheet.shape[1]):
            if col_idx in month_cols:
                continue
            cell = sheet.iat[row_idx, col_idx]
            if isinstance(cell, str) and cell.strip():
                field_name = cell.strip()
                break
        if not field_name:
            continue
        # Hopp over tittel-/oppsummeringsrader.
        lower = field_name.lower()
        if lower.startswith(("normpris", "differensial", "snitt", "gjennomsnitt", "kvartal")):
            continue

        for col_idx, month_num in month_cols.items():
            cell = sheet.iat[row_idx, col_idx]
            if pd.isna(cell):
                continue
            try:
                diff = float(cell)
            except (TypeError, ValueError):
                continue
            rows.append({
                "field": field_name.upper(),
                "month": month_num,
                "differential_usd": diff,
            })
    return pd.DataFrame(rows)


def main() -> None:
    RAW_XLSX_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Henter arkiv-side fra regjeringen.no ...")
    all_links = fetch_archive_links()
    print(f"  totalt {len(all_links)} XLSX-lenker funnet")

    main_links = [l for l in all_links if is_main_quarterly_file(l)]
    print(f"  beholder {len(main_links)} hovedfiler etter filtering")

    pieces: list[pd.DataFrame] = []
    skipped: list[str] = []

    for rel_url in main_links:
        filename = rel_url.rsplit("/", 1)[-1]
        local = RAW_XLSX_DIR / filename
        full_url = BASE_URL + rel_url

        if not download_if_missing(full_url, local):
            skipped.append(f"{filename} (nedlasting)")
            continue

        qy = parse_quarter_and_year(rel_url)
        if qy is None:
            skipped.append(f"{filename} (kvartal/år)")
            continue
        quarter, year = qy

        df = parse_differentials_sheet(local)
        if df.empty:
            skipped.append(f"{filename} (ingen Differensialer-ark)")
            continue

        df["year"] = year
        df["source_quarter"] = quarter
        df["source_file"] = filename
        pieces.append(df)
        print(f"  {filename}: {len(df)} rader (Q{quarter} {year})")

    if not pieces:
        print("\nFant ingen brukbare data — avbryter.")
        return

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sort_values(["field", "year", "month"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Resultat ===")
    print(f"Rader totalt: {len(combined):,}")
    print(f"Unike felt:   {combined['field'].nunique()}")
    print(f"Tidsspenn:    {combined['year'].min()}-{combined['year'].max()}")
    print(f"Lagret:       {OUTPUT_CSV}")

    print("\n=== Dekning for våre fire mål-felt ===")
    for target in TARGET_FIELDS:
        sub = combined[combined["field"] == target]
        if sub.empty:
            print(f"  {target}: INGEN treff i Differensialer-arket")
        else:
            yrs = sorted(sub["year"].unique())
            print(f"  {target}: {len(sub):>3} obs, år {yrs[0]}–{yrs[-1]} "
                  f"(snitt diff: {sub['differential_usd'].mean():+.2f} USD/fat)")

    if skipped:
        print(f"\nHoppet over {len(skipped)} filer (gammelt format e.l.):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... og {len(skipped) - 10} til")


if __name__ == "__main__":
    main()
