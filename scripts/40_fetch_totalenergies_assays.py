"""
Hent alle TotalEnergies crude oil assays (XLSX-format).

TotalEnergies publiserer 71 crude assays på:
  https://trading.totalenergies.com/en/business-customers/oil/crude-assays/

XLSX-filer lastes ned direkte fra deres CDN.
"""

from pathlib import Path
import json
import time
import requests

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "totalenergies_assays"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "totalenergies_assay_index.json"

BASE = "https://trading.totalenergies.com/wp-content/uploads/2025/11"

ASSAYS = [
    # Africa
    {"grade": "Akpo Blend", "url": f"{BASE}/AKPO-BLEND.xlsx"},
    {"grade": "Al Jurf", "url": f"{BASE}/AL-JURF.xlsx"},
    {"grade": "Amenam Blend", "url": f"{BASE}/AMENAM-BLEND.xlsx"},
    {"grade": "Bonga", "url": f"{BASE}/BONGA.xlsx"},
    {"grade": "Bonny Light", "url": f"{BASE}/BONNY-LIGHT.xlsx"},
    {"grade": "Brass River", "url": f"{BASE}/BRASS-RIVER.xlsx"},
    {"grade": "Cabinda", "url": f"{BASE}/CABINDA.xlsx"},
    {"grade": "Clov", "url": f"{BASE}/CLOV.xlsx"},
    {"grade": "Dalia", "url": f"{BASE}/DALIA.xlsx"},
    {"grade": "Djeno", "url": f"{BASE}/DJENO.xlsx"},
    {"grade": "Ea Blend", "url": f"{BASE}/EA-BLEND.xlsx"},
    {"grade": "Egina", "url": f"{BASE}/EGINA.xlsx"},
    {"grade": "El Sharara", "url": f"{BASE}/EL-SHARARA.xlsx"},
    {"grade": "Es Sider", "url": f"{BASE}/ES-SIDER.xlsx"},
    {"grade": "Forcados", "url": f"{BASE}/FORCADOS.xlsx"},
    {"grade": "Gindungo", "url": f"{BASE}/GINDUNGO.xlsx"},
    {"grade": "Girassol", "url": f"{BASE}/GIRASSOL.xlsx"},
    {"grade": "Mandji", "url": f"{BASE}/MANDJI.xlsx"},
    {"grade": "Mostarda", "url": f"{BASE}/MOSTARDA.xlsx"},
    {"grade": "Nemba", "url": f"{BASE}/NEMBA.xlsx"},
    {"grade": "N'kossa Blend", "url": f"{BASE}/NKOSSA-BLEND.xlsx"},
    # Asia
    {"grade": "Badak", "url": f"{BASE}/BADAK.xlsx"},
    {"grade": "Bekapai", "url": f"{BASE}/BEKAPAI.xlsx"},
    {"grade": "Bontang Return Condensate", "url": f"{BASE}/CONDENSATE-BONTANG-RETURN.xlsx"},
    {"grade": "Ichthys Condensate", "url": f"{BASE}/CONDENSATE-ICHTHYS.xlsx"},
    {"grade": "Senipah Condensate", "url": f"{BASE}/CONDENSATE-SENIPAH.xlsx"},
    {"grade": "Handil Mix", "url": f"{BASE}/HANDIL-MIX.xlsx"},
    {"grade": "Seria Light", "url": f"{BASE}/SERIA-LIGHT.xlsx"},
    # Europe
    {"grade": "Asgard Blend", "url": f"{BASE}/ASGARD-BLEND.xlsx"},
    {"grade": "Brent", "url": f"{BASE}/BRENT.xlsx"},
    {"grade": "Snohvit Condensate", "url": f"{BASE}/CONDENSATE-SNOHVIT.xlsx"},
    {"grade": "Culzean", "url": f"{BASE}/CULZEAN.xlsx"},
    {"grade": "DUC", "url": f"{BASE}/DUC.xlsx"},
    {"grade": "Dumbarton", "url": f"{BASE}/DUMBARTON.xlsx"},
    {"grade": "Ekofisk", "url": f"{BASE}/EKOFISK.xlsx"},
    {"grade": "Flotta Gold", "url": f"{BASE}/FLOTTA-GOLD.xlsx"},
    {"grade": "Forties", "url": f"{BASE}/FORTIES.xlsx"},
    {"grade": "Gryphon", "url": f"{BASE}/GRYPHON.xlsx"},
    {"grade": "Gudrun Blend", "url": f"{BASE}/GUDRUN-BLEND.xlsx"},
    {"grade": "Gullfaks", "url": f"{BASE}/GULLFAKS.xlsx"},
    {"grade": "Harding", "url": f"{BASE}/HARDING.xlsx"},
    {"grade": "Johan Sverdrup", "url": f"{BASE}/JOHAN-SVERDRUP.xlsx"},
    {"grade": "Oseberg", "url": f"{BASE}/OSEBERG.xlsx"},
    {"grade": "Troll", "url": f"{BASE}/TROLL.xlsx"},
    # Middle East
    {"grade": "Al Shaheen", "url": f"{BASE}/AL-SHAHEEN.xlsx"},
    {"grade": "Das Blend", "url": f"{BASE}/DAS-BLEND.xlsx"},
    {"grade": "Murban", "url": f"{BASE}/MURBAN.xlsx"},
    {"grade": "Oman", "url": f"{BASE}/OMAN.xlsx"},
    {"grade": "Qatar Marine", "url": f"{BASE}/QATAR-MARINE.xlsx"},
    {"grade": "Upper Zakum", "url": f"{BASE}/UPPER-ZAKHUM.xlsx"},
    # Latin America
    {"grade": "Atapu", "url": f"{BASE}/ATAPU.xlsx"},
    {"grade": "Lapa", "url": f"{BASE}/LAPA.xlsx"},
    {"grade": "Mero", "url": f"{BASE}/MERO.xlsx"},
    {"grade": "Sepia", "url": f"{BASE}/SEPIA.xlsx"},
    {"grade": "Sururu", "url": f"{BASE}/SURURU.xlsx"},
    # North America
    {"grade": "Cascade Chinook Blend", "url": f"{BASE}/CASCADE-CHINOOK-BLEND.xlsx"},
    {"grade": "Fort Hills Dilbit", "url": f"{BASE}/FORT-HILLS-REDUCED-CARBON-LIFE-CYCLE-DILBIT.xlsx"},
    {"grade": "Synbit SHB", "url": f"{BASE}/SYNBIT-SHB.xlsx"},
]


def download(item: dict) -> tuple[Path | None, int]:
    fname = item["url"].split("/")[-1]
    dest = CACHE_DIR / fname
    if dest.exists():
        return dest, dest.stat().st_size
    try:
        resp = requests.get(item["url"], timeout=60, headers={
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


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Laster ned {len(ASSAYS)} TotalEnergies crude assays ===\n")

    ok, failed = [], []
    for item in ASSAYS:
        path, size = download(item)
        if path:
            ok.append({**item, "local_file": path.name})
            print(f"  ✓ {item['grade']:35s} | {size:>7,} bytes")
        else:
            failed.append(item)

    index_data = [{
        "grade": i["grade"],
        "url": i["url"],
        "local_file": i["local_file"],
        "source": "TotalEnergies",
    } for i in ok]
    INDEX_FILE.write_text(json.dumps(index_data, indent=2, ensure_ascii=False))

    print(f"\n=== Resultat ===")
    print(f"  Lastet ned: {len(ok)}/{len(ASSAYS)}")
    print(f"  Cache:      {CACHE_DIR}")
    print(f"  Indeks:     {INDEX_FILE}")
    if failed:
        print(f"  Feilet:     {[i['grade'] for i in failed]}")


if __name__ == "__main__":
    main()
