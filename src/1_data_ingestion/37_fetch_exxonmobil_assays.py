"""
Hent alle ExxonMobil crude oil assays (XLSX-format).

ExxonMobil publiserer 43 crude assays på:
  https://corporate.exxonmobil.com/what-we-do/energy-supply/crude-trading/crude-oil-assays

Dette scriptet:
  1. Laster ned alle XLSX-filer
  2. Lagrer i data/raw/exxonmobil_assays/
  3. Bygger en indeksfil for parsing
"""

from pathlib import Path
import json
import time
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "exxonmobil_assays"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "exxonmobil_assay_index.json"

BASE_URL = "https://corporate.exxonmobil.com/-/media/global/files/crude-oils/xls"

# Full index of all ExxonMobil crude assays (scraped May 2026)
ASSAYS = [
    {"grade": "Alaskan North Slope (ANS)", "file": "2024/alaska_north_slope.xlsx"},
    {"grade": "Azeri BTC", "file": "2024/azeri_btc.xlsx"},
    {"grade": "Azeri Light", "file": "2024/azeri_light.xlsx"},
    {"grade": "Bakken", "file": "2024/bakken.xlsx"},
    {"grade": "Banyu Urip", "file": "2024/banyu_urip.xlsx"},
    {"grade": "Bonga", "file": "2025/bonga.xlsx"},
    {"grade": "CLOV", "file": "2024/clov.xlsx"},
    {"grade": "Cold Lake Blend", "file": "2024/cold_lake_blend.xlsx"},
    {"grade": "Coral Condensate", "file": "2024/coral_condensate.xlsx"},
    {"grade": "CPC Blend", "file": "2025/cpc_blend.xlsx"},
    {"grade": "Dalia", "file": "2024/dalia.xlsx"},
    {"grade": "Domestic Sweet", "file": "2024/domestic_sweet.xlsx"},
    {"grade": "Ebok", "file": "2024/ebok.xlsx"},
    {"grade": "Erha", "file": "2024/erha.xlsx"},
    {"grade": "Gindungo", "file": "2024/gindungo.xlsx"},
    {"grade": "Gippsland Condensate", "file": "2025/gippsland_condensate.xlsx"},
    {"grade": "Girassol", "file": "2024/girassol.xlsx"},
    {"grade": "Golden Arrowhead", "file": "2025/golden_arrowheadv2.xlsx"},
    {"grade": "Gorgon", "file": "2024/gorgon.xlsx"},
    {"grade": "Hebron", "file": "2025/hebron.xlsx"},
    {"grade": "Hibernia Blend", "file": "2025/hibernia.xlsx"},
    {"grade": "HOOPS Blend", "file": "2025/hoops.xlsx"},
    {"grade": "Hungo Blend", "file": "2024/hungo_blend.xlsx"},
    {"grade": "Kearl", "file": "2024/kearl.xlsx"},
    {"grade": "Kissanje Blend", "file": "2024/kissanje_blend.xlsx"},
    {"grade": "Kutubu", "file": "2025/kutubu.xlsx"},
    {"grade": "Liza", "file": "2024/liza_v-2.xlsx"},
    {"grade": "Mondo Blend", "file": "2024/mondo_blend.xlsx"},
    {"grade": "Mostarda", "file": "2024/mostarda.xlsx"},
    {"grade": "Payara Gold", "file": "2024/payara_gold.xlsx"},
    {"grade": "Pazflor", "file": "2024/pazflor.xlsx"},
    {"grade": "Qua Iboe", "file": "2024/qua_iboe.xlsx"},
    {"grade": "Saxi Batuque", "file": "2024/saxi_batuque.xlsx"},
    {"grade": "Tapis", "file": "2024/tapis.xlsx"},
    {"grade": "Terengganu Condensate", "file": "2024/terengganu_condensate.xlsx"},
    {"grade": "Thunder Horse", "file": "2024/thunder_horse.xlsx"},
    {"grade": "Unity Gold", "file": "2025/unity_gold.xlsx"},
    {"grade": "Upper Zakum", "file": "2024/upper_zakum.xlsx"},
    {"grade": "Usan", "file": "2025/usan.xlsx"},
    {"grade": "WTI Light", "file": "2024/wti_light.xlsx"},
    {"grade": "Yoho", "file": "2024/yoho.xlsx"},
    {"grade": "Zafiro Blend", "file": "2024/zafiro_blend.xlsx"},
]


def download(item: dict) -> tuple[Path | None, int]:
    """Last ned en XLSX-fil fra ExxonMobil."""
    fname = item["file"].split("/")[-1]
    dest = CACHE_DIR / fname
    if dest.exists():
        return dest, dest.stat().st_size

    url = f"{BASE_URL}/{item['file']}"
    try:
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        time.sleep(0.3)
        return dest, len(resp.content)
    except Exception as e:
        print(f"    FEILET ({item['grade']}): {e}")
        return None, 0


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Laster ned {len(ASSAYS)} ExxonMobil crude assays ===\n")

    ok, failed = [], []
    for item in ASSAYS:
        path, size = download(item)
        if path:
            ok.append({**item, "local_file": str(path.name)})
            print(f"  ✓ {item['grade']:35s} | {size:>7,} bytes")
        else:
            failed.append(item)

    # Lagre indeks
    index_data = []
    for item in ok:
        index_data.append({
            "grade": item["grade"],
            "url": f"{BASE_URL}/{item['file']}",
            "local_file": item["local_file"],
            "source": "ExxonMobil",
        })
    INDEX_FILE.write_text(json.dumps(index_data, indent=2, ensure_ascii=False))

    print(f"\n=== Resultat ===")
    print(f"  Lastet ned: {len(ok)}/{len(ASSAYS)}")
    print(f"  Cache:      {CACHE_DIR}")
    print(f"  Indeks:     {INDEX_FILE}")
    if failed:
        print(f"  Feilet:     {[i['grade'] for i in failed]}")


if __name__ == "__main__":
    main()
