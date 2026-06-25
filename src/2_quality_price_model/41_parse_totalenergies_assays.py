"""
Parse TotalEnergies assay XLSX-er → strukturert CSV.

TotalEnergies-formatet er annerledes enn Equinor/ExxonMobil:
  - Sheet: "Assay"
  - Whole crude props i rader 5-25 (col B=label, col E=verdi)
  - TBP kutt i seksjoner: Light Naphtha, Heavy Naphtha, Kerosene, Gasoil,
    Vacuum Distillate, Residue

Vi tar ikke-overlappende kutt:
  - Light naphtha: 15-65°C (eller 15-80°C)
  - Heavy naphtha: 80-150°C (eller 100-150°C)
  - Kerosene: 150-230°C
  - Diesel: 230-375°C
  - VGO: 375-550°C
  - Vacuum residue: >550°C
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSAY_DIR = PROJECT_ROOT / "data" / "raw" / "totalenergies_assays"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "totalenergies_assay_index.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "totalenergies_assays_parsed.csv"

# Preferred non-overlapping cut ranges for our model
# (start, end) → model variable
CUT_MAP = {
    (15, 65): "light_naphtha_pct",    # C5-65°C
    (15, 80): "light_naphtha_pct",    # alt
    (80, 150): "heavy_naphtha_pct",   # 80-150°C
    (100, 150): "heavy_naphtha_pct",  # alt
    (65, 150): "heavy_naphtha_pct",   # alt
    (150, 230): "kerosene_pct",
    (150, 250): "kerosene_pct",       # alt
    (230, 375): "diesel_pct",         # gasoil = diesel + heavy diesel
    (230, 400): "diesel_pct",         # alt
    (250, 350): "diesel_pct",         # alt
    (250, 375): "diesel_pct",         # alt
    (375, 550): "vgo_pct",
    (375, 565): "vgo_pct",            # alt
    (400, 580): "vgo_pct",            # alt
}
# Residue patterns
RESIDUE_PATTERNS = [">550", ">565", ">580", "> 550", "> 565", "> 580"]


def parse_cut_range(s: str) -> tuple[float | None, float | None]:
    """Parse '15-65' eller '> 550' til (start, end)."""
    s = str(s).strip()
    m = re.match(r">\s*(\d+)", s)
    if m:
        return float(m.group(1)), 999.0
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def parse_assay(xlsx_path: Path) -> dict | None:
    """Parse én TotalEnergies assay XLSX."""
    # Prøv ulike sheet-navn
    df = None
    for sn in ["Assay", "Assay TOTAL", "Assay TE", "Sheet1"]:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sn, header=None, engine="openpyxl")
            break
        except Exception:
            continue
    if df is None:
        print(f"    Kunne ikke finne Assay-sheet i {xlsx_path.name}")
        return None

    result = {}

    # Whole crude properties (col B=1, col E=4)
    def get_prop(label_sub: str, label_col: int = 1, value_col: int = 4) -> float | None:
        for r in range(min(30, df.shape[0])):
            cell = df.iat[r, label_col] if label_col < df.shape[1] else None
            if isinstance(cell, str) and label_sub.lower() in cell.lower():
                if value_col < df.shape[1]:
                    v = df.iat[r, value_col]
                    if pd.notna(v):
                        try:
                            fv = float(v)
                            return fv
                        except (TypeError, ValueError):
                            return None
        return None

    result["api_gravity"] = get_prop("°API")
    result["density_g_cc"] = None
    dens_kgm3 = get_prop("Density at 15")
    if dens_kgm3:
        result["density_g_cc"] = dens_kgm3 / 1000.0

    result["sulfur_pct"] = get_prop("Sulphur, wt%")
    if result["sulfur_pct"] is None:
        result["sulfur_pct"] = get_prop("Sulfur, wt%")
    result["tan_mgkoh"] = get_prop("Acidity, mg KOH")
    result["pour_point_c"] = get_prop("Pour Point")

    # Nitrogen — wt% → ppm
    n_pct = get_prop("Total Nitrogen, wt%")
    if n_pct and n_pct < 1:
        result["nitrogen_ppm"] = n_pct * 10000  # wt% → ppm
    elif n_pct:
        result["nitrogen_ppm"] = n_pct

    result["wax_pct"] = get_prop("Wax, wt%")
    result["rvp_psi"] = None  # TotalEnergies uses kPa
    rvp_kpa = get_prop("RVP at 37.8")
    if rvp_kpa:
        result["rvp_psi"] = rvp_kpa * 0.145038  # kPa → psi

    result["nickel_ppm"] = get_prop("Nickel, mg/kg")
    result["vanadium_ppm"] = get_prop("Vanadium, mg/kg")
    result["mercaptan_sulphur_ppm"] = get_prop("Mercaptan Sulphur")
    if result["mercaptan_sulphur_ppm"] is None:
        result["mercaptan_sulphur_ppm"] = get_prop("Mercaptan Sulfur")

    # Viscosity — TotalEnergies gir ved 10°C og 50°C, ikke 20°C/40°C
    # Vi prøver å finne verdier
    for r in range(min(30, df.shape[0])):
        cell = df.iat[r, 1] if 1 < df.shape[1] else None
        if isinstance(cell, str) and "viscosity" in cell.lower():
            # Sjekk temperaturen i neste kolonne
            temp_cell = df.iat[r, 2] if 2 < df.shape[1] else None
            if isinstance(temp_cell, str) and "50" in temp_cell:
                v = df.iat[r, 4] if 4 < df.shape[1] else None
                if pd.notna(v):
                    try:
                        result["viscosity_cst_40c"] = float(v)  # ~50°C ≈ 40°C approx
                    except (TypeError, ValueError):
                        pass
            # Sjekk neste rad for 50°C
            if r + 1 < df.shape[0]:
                temp_cell2 = df.iat[r + 1, 2] if 2 < df.shape[1] else None
                if isinstance(temp_cell2, str) and "50" in temp_cell2:
                    v = df.iat[r + 1, 4] if 4 < df.shape[1] else None
                    if pd.notna(v):
                        try:
                            result["viscosity_cst_40c"] = float(v)
                        except (TypeError, ValueError):
                            pass

    # Hent dato fra siste rad
    for r in range(df.shape[0] - 1, max(0, df.shape[0] - 5), -1):
        for c in range(df.shape[1]):
            v = df.iat[r, c]
            if pd.notna(v) and isinstance(v, pd.Timestamp):
                result["assay_date"] = v.strftime("%Y-%m-%d")
                break
            elif pd.notna(v):
                try:
                    ts = pd.Timestamp(v)
                    result["assay_date"] = ts.strftime("%Y-%m-%d")
                    break
                except Exception:
                    pass

    # === Parse TBP cuts ===
    # Finn kutt-seksjoner og hent yields
    cuts = {
        "lpg_pct": 0.0,
        "light_naphtha_pct": 0.0,
        "heavy_naphtha_pct": 0.0,
        "kerosene_pct": 0.0,
        "diesel_pct": 0.0,
        "heavy_diesel_pct": 0.0,
        "vgo_pct": 0.0,
        "vacuum_resid_pct": 0.0,
    }

    # Hent CCR / asphaltenes fra residue-seksjonen
    # Scan for non-overlapping cut yields
    found_cuts = {}  # model_var → (range_str, yield_wt)

    for r in range(28, min(df.shape[0], 65)):
        # Kolonne C (idx 2) har kutt-range, kolonne D (idx 3) har yield wt%
        cut_cell = df.iat[r, 2] if 2 < df.shape[1] else None
        yield_cell = df.iat[r, 3] if 3 < df.shape[1] else None

        if cut_cell is None or yield_cell is None:
            continue
        if not isinstance(cut_cell, str):
            try:
                cut_cell = str(cut_cell)
            except Exception:
                continue

        start, end = parse_cut_range(cut_cell)
        if start is None:
            continue

        try:
            yield_val = float(yield_cell)
        except (TypeError, ValueError):
            continue

        # Map til modellvariabel
        if end >= 999:  # Residue
            # Ta minste residue
            if start <= 555:
                model_var = "vacuum_resid_pct"
                if model_var not in found_cuts or start < parse_cut_range(found_cuts[model_var][0])[0]:
                    found_cuts[model_var] = (cut_cell, yield_val)
        else:
            key = (int(start), int(end))
            if key in CUT_MAP:
                model_var = CUT_MAP[key]
                # Ta den mest spesifikke (nærmest modellens definisjon)
                if model_var not in found_cuts:
                    found_cuts[model_var] = (cut_cell, yield_val)

    # Hent også Conradson/CCR fra VD-seksjonen og asphaltenes
    for r in range(28, min(df.shape[0], 65)):
        label = df.iat[r, 1] if 1 < df.shape[1] else ""
        if not isinstance(label, str):
            continue
        # Residue section header
        if "residue" in label.lower():
            # Neste rader har yield for ">375", ">550", etc.
            pass
        # Look for whole crude Conradson in VD section
        for c in range(df.shape[1]):
            cell = df.iat[r, c]
            if isinstance(cell, str) and "conrad" in cell.lower():
                # Get value from residue row
                pass

    # Sett kutt-verdier
    for model_var, (range_str, yield_val) in found_cuts.items():
        cuts[model_var] = yield_val

    # Spesialbehandling: diesel_pct i TotalEnergies inkluderer heavy diesel
    # "230-375" er gasoil = diesel + heavy diesel
    # Vi splitter ikke videre her

    result.update(cuts)

    return result


def main():
    index = json.loads(INDEX_FILE.read_text())
    print(f"=== Parser {len(index)} TotalEnergies assays ===\n")

    rows = []
    success, failed = 0, 0
    for item in index:
        path = ASSAY_DIR / item["local_file"]
        if not path.exists():
            print(f"  ✗ {item['grade']:35s} (mangler fil)")
            failed += 1
            continue

        parsed = parse_assay(path)
        if parsed is None:
            print(f"  ✗ {item['grade']:35s} (parse-feil)")
            failed += 1
            continue

        row = {
            "grade_total": item["grade"],
            "source": "TotalEnergies",
            "source_url": item["url"],
            "source_date": parsed.pop("assay_date", ""),
        }
        row.update(parsed)

        # Aggregater
        ln = row.get("light_naphtha_pct", 0) or 0
        hn = row.get("heavy_naphtha_pct", 0) or 0
        ke = row.get("kerosene_pct", 0) or 0
        di = row.get("diesel_pct", 0) or 0
        hd = row.get("heavy_diesel_pct", 0) or 0
        vg = row.get("vgo_pct", 0) or 0
        vr = row.get("vacuum_resid_pct", 0) or 0

        row["naphtha_pct"] = ln + hn
        row["middle_distillate_pct"] = ke + di
        row["bottom_of_barrel_pct"] = vg + vr
        row["high_value_yield_pct"] = ln + hn + ke + di + hd

        has_api = row.get("api_gravity") is not None
        has_cuts = any(row.get(k) for k in ["naphtha_pct", "diesel_pct", "vgo_pct"])

        api_s = f"{row.get('api_gravity', 0):5.1f}" if row.get("api_gravity") else "  N/A"
        sul_s = f"{row.get('sulfur_pct', 0):5.3f}" if row.get("sulfur_pct") else "  N/A"
        hvp = f"{row.get('high_value_yield_pct', 0):5.1f}" if has_cuts else "  N/A"
        status = "✓" if has_api and has_cuts else "△" if has_api else "✗"

        if has_api:
            success += 1
        else:
            failed += 1

        print(f"  {status} {item['grade']:35s} | API {api_s} | S {sul_s}% | HVY {hvp}% | {row.get('source_date', '')}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n=== Resultat ===")
    print(f"  Vellykket:  {success}")
    print(f"  Feilet:     {failed}")
    print(f"  Output:     {OUTPUT_CSV}")
    print(f"  Kolonner:   {len(df.columns)}")

    # QC
    if not df.empty:
        for col in ["api_gravity", "sulfur_pct", "vgo_pct", "vacuum_resid_pct"]:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:25s}: mean={vals.mean():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")


if __name__ == "__main__":
    main()
