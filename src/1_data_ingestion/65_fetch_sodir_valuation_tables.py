"""
Script 65 — Hent Sodir-tabellene NAV-motoren trenger
====================================================================
Henter de manglende FactPages-tabellene for felt-for-felt NAV på NCS:

  1. discovery_reserves        — ressursanslag per funn (klasse 4F/5F/7F,
                                 utvinnbar olje/gass/NGL/kondensat)
  2. company_reserves          — reserver per selskap per felt (netto andeler
                                 + WI %) → eierskapsmatrisen
  3. field_investment_expected — forventede fremtidige investeringer per felt
                                 (fast NOK) → capex-forecast
  4. licence_licensee_hst      — lisensandeler over tid (WI for funn som ikke
                                 er bokført på felt)
  5. field_owner_hst           — feltets eierhistorikk (supplement til 2)

Samme offentlige CSV-API som scripts 09/48. Output: data/raw/sodir/
"""

from pathlib import Path
import sys
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "sodir"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://factpages.sodir.no/public?/Factpages/external/tableview/"
    "{table}&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&"
    "IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
)

# tabellnavn → (outputfil, kandidatnavn i prioritert rekkefølge)
TABLES = {
    "discovery_reserves": ["discovery_reserves"],
    "company_reserves": ["company_reserves", "field_reserves_by_company"],
    "field_investment_expected": ["field_investment_expected",
                                  "field_investment_forecast"],
    "licence_licensee_hst": ["licence_licensee_hst", "licence_licensee"],
    "field_owner_hst": ["field_owner_hst", "field_licensee_hst"],
}


def fetch_table(candidates: list[str], out_path: Path, force: bool = False) -> pd.DataFrame | None:
    if out_path.exists() and not force:
        print(f"  cache: {out_path.name}")
        return pd.read_csv(out_path, encoding="utf-8-sig", low_memory=False)
    for table in candidates:
        url = BASE.format(table=table)
        try:
            r = requests.get(url, timeout=180)
            # Sodir svarer 200 med HTML-feilside på ukjente tabeller — sjekk innhold
            if r.status_code == 200 and not r.content[:200].lstrip().startswith(b"<"):
                out_path.write_bytes(r.content)
                df = pd.read_csv(out_path, encoding="utf-8-sig", low_memory=False)
                print(f"  ✓ {table} → {out_path.name}  ({len(df)} rader, {len(df.columns)} kolonner)")
                return df
            print(f"  ✗ {table}: ikke gyldig CSV (status {r.status_code})")
        except requests.RequestException as e:
            print(f"  ✗ {table}: {e}")
    print(f"  !! ingen kandidat traff for {out_path.name}")
    return None


def main() -> None:
    force = "--force" in sys.argv
    print("=" * 60)
    print("SCRIPT 65: Sodir-tabeller for NAV-motoren")
    print("=" * 60)
    results = {}
    for key, candidates in TABLES.items():
        out = CACHE_DIR / f"sodir_{key}.csv"
        results[key] = fetch_table(candidates, out, force)

    print("\nOppsummering:")
    for key, df in results.items():
        status = f"{len(df)} rader" if df is not None else "MANGLER"
        print(f"  {key:28s} {status}")
        if df is not None:
            print(f"    kolonner: {', '.join(list(df.columns)[:8])}"
                  + (" …" if len(df.columns) > 8 else ""))


if __name__ == "__main__":
    main()
