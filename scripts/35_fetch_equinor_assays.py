"""
Hent alle Equinor crude oil assays (ekte primærkilde-data).

Equinor publiserer XLSX-assays for 46 grades på sin offentlige Sanity CDN.
Indeksen (navn → URL) er bygd fra https://www.equinor.com/energy/crude-oil-assays
og lagret i data/raw/equinor_assay_index.json.

Dette scriptet:
  1. Leser indeksen
  2. Laster ned hver XLSX (cachet lokalt)
  3. Lagrer i data/raw/equinor_assays/

Parse-logikk ligger i scripts/36_parse_equinor_assays.py.
"""

from pathlib import Path
import json
import re
import time
import requests

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "equinor_assays"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "equinor_assay_index.json"


def safe_filename(grade: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", grade).strip("_").lower()


def download(item: dict) -> tuple[Path | None, int]:
    fname = safe_filename(item["grade"]) + ".xlsx"
    dest = CACHE_DIR / fname
    if dest.exists():
        return dest, dest.stat().st_size
    try:
        resp = requests.get(item["url"], timeout=60, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        time.sleep(0.3)
        return dest, len(resp.content)
    except Exception as e:
        print(f"    Feilet ({item['grade']}): {e}")
        return None, 0


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_FILE.exists():
        print(f"FEIL: {INDEX_FILE} mangler. Kjør først indekseringen.")
        return

    index = json.loads(INDEX_FILE.read_text())
    print(f"=== Laster ned {len(index)} Equinor crude assays ===\n")

    ok, failed = [], []
    for item in index:
        path, size = download(item)
        if path:
            ok.append(item)
            cached = "(cached)" if size == path.stat().st_size and path.exists() else ""
            print(f"  ✓ {item['grade']:35s} | {item['date']:12s} | {size:>7,} bytes")
        else:
            failed.append(item)

    print(f"\n=== Resultat ===")
    print(f"  Lastet ned: {len(ok)}/{len(index)}")
    print(f"  Cache:      {CACHE_DIR}")
    if failed:
        print(f"  Feilet:     {[i['grade'] for i in failed]}")


if __name__ == "__main__":
    main()
