"""
Parse ExxonMobil assay XLSX-er → strukturert CSV.

ExxonMobil-formatet er nesten identisk med Equinor:
  - Summary (C) sheet
  - Rad 42: API Gravity (col B=label, col C=whole crude)
  - Rad 47: Total Sulfur (NB: "Sulfur" ikke "Sulphur")
  - Cut Data: Start/End temperaturer + Yield (% wt) / Yield (% vol)
  - Atmospheric: C5-65, 65-100, 100-150, 150-200, 200-250, 250-300, 300-350, 350-370
  - Vacuum: 370-450, 450-500, 500-550, 550-FBP

Gjenbruker parse-logikk fra scripts/36_parse_equinor_assays.py.
"""

from pathlib import Path
import json
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSAY_DIR = PROJECT_ROOT / "data" / "raw" / "exxonmobil_assays"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "exxonmobil_assay_index.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "exxonmobil_assays_parsed.csv"


def find_value(df: pd.DataFrame, label_substring: str, label_col: int = 1,
               value_col: int = 2) -> float | None:
    """Søk i kolonne `label_col` etter celle som inneholder `label_substring`,
    returner verdien i kolonne `value_col` på samme rad."""
    for r in range(df.shape[0]):
        cell = df.iat[r, label_col] if label_col < df.shape[1] else None
        if isinstance(cell, str) and label_substring.lower() in cell.lower():
            if value_col < df.shape[1]:
                v = df.iat[r, value_col]
                if pd.notna(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
    return None


def find_value_multi_col(df: pd.DataFrame, label_substring: str,
                          value_cols: list[int] = None) -> float | None:
    """Søk etter labelen i hvilken som helst kolonne, ta verdien til høyre."""
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            cell = df.iat[r, c]
            if isinstance(cell, str) and label_substring.lower() in cell.lower():
                cols_to_try = value_cols if value_cols else [c + 1, c + 2, c + 3]
                for vc in cols_to_try:
                    if vc < df.shape[1]:
                        v = df.iat[r, vc]
                        if pd.notna(v):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                continue
    return None


def parse_distillation_cuts(df: pd.DataFrame) -> dict:
    """Hent yield-fordeling fra cut data-tabellen — identisk logikk som Equinor."""
    # Finn rad for "Yield (% wt)" — ExxonMobil bruker wt, Equinor bruker vol
    yield_row = None
    for r in range(df.shape[0]):
        for c in range(min(3, df.shape[1])):
            cell = df.iat[r, c]
            if isinstance(cell, str):
                if "yield" in cell.lower() and "vol" in cell.lower():
                    yield_row = r
                    break
        if yield_row is not None:
            break

    if yield_row is None:
        for r in range(df.shape[0]):
            for c in range(min(3, df.shape[1])):
                cell = df.iat[r, c]
                if isinstance(cell, str) and "yield" in cell.lower() and "wt" in cell.lower():
                    yield_row = r
                    break
            if yield_row is not None:
                break

    if yield_row is None:
        return {}

    # Finn Start (°C)-raden
    start_row = None
    for r in range(df.shape[0]):
        cell = df.iat[r, 1] if 1 < df.shape[1] else None
        if isinstance(cell, str) and "start" in cell.lower() and "°c" in cell.lower():
            start_row = r
            break

    if start_row is None:
        return {}

    # Bygg map fra kolonne → start-temperatur
    col_to_start = {}
    for c in range(2, df.shape[1]):
        v = df.iat[start_row, c]
        if pd.notna(v):
            col_to_start[c] = str(v).strip()

    # Hent slutt-labels
    end_row = start_row + 1
    col_to_end = {}
    for c in col_to_start:
        if end_row < df.shape[0]:
            v = df.iat[end_row, c]
            if pd.notna(v):
                col_to_end[c] = str(v).strip()

    def parse_temp(s: str) -> float | None:
        u = s.upper().strip()
        if u in ("IBP",):
            return 0.0
        if u in ("C4",):
            return 0.0
        if u in ("C5",):
            return 36.0
        if "FBP" in u:
            return 999.0
        try:
            return float(u)
        except ValueError:
            return None

    # Bygg liste over kutt
    cuts_list = []
    for c, sl in col_to_start.items():
        el = col_to_end.get(c, "")
        st = parse_temp(sl)
        et = parse_temp(el)
        if st is None or et is None:
            continue
        yv = df.iat[yield_row, c]
        if pd.isna(yv):
            continue
        try:
            y = float(yv)
        except (TypeError, ValueError):
            continue
        cuts_list.append((c, sl, el, st, et, y))

    if not cuts_list:
        return {}

    # Filtrer duplikat-totaler
    final = []
    for i, (c, sl, el, st, et, y) in enumerate(cuts_list):
        is_total = "FBP" in el.upper() or et >= 999
        if is_total:
            has_subcuts = any(
                cc > c and st2 >= st
                for cc, _, _, st2, _, _ in cuts_list[i+1:]
            )
            if has_subcuts:
                continue
        if sl.upper() == "IBP" and ("FBP" in el.upper() or et >= 999):
            if len(cuts_list) > 1:
                continue
        final.append((c, sl, el, st, et, y))

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

    for c, sl, el, st, et, y in final:
        mid = (st + et) / 2
        if mid < 35:
            cuts["lpg_pct"] += y
        elif mid < 65:
            cuts["light_naphtha_pct"] += y
        elif mid < 150:
            cuts["heavy_naphtha_pct"] += y
        elif mid < 250:
            cuts["kerosene_pct"] += y
        elif mid < 350:
            cuts["diesel_pct"] += y
        elif mid < 370:
            cuts["heavy_diesel_pct"] += y
        elif mid < 550:
            cuts["vgo_pct"] += y
        else:
            cuts["vacuum_resid_pct"] += y

    return cuts


def parse_assay(xlsx_path: Path) -> dict | None:
    """Parse én ExxonMobil assay XLSX."""
    # Prøv ulike sheet-navn
    sheet_name = None
    for sn in ["Summary (C)", "Summary", "Sheet1"]:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sn, header=None, engine="openpyxl")
            sheet_name = sn
            break
        except Exception:
            continue

    if sheet_name is None:
        print(f"    Kunne ikke finne Summary-sheet i {xlsx_path.name}")
        return None

    result = {}

    # Whole crude properties
    # ExxonMobil bruker "Sulfur" (ikke "Sulphur" som Equinor)
    result["api_gravity"] = find_value(df, "API Gravity", 1, 2)
    result["density_g_cc"] = find_value(df, "Density @ 15", 1, 2)

    # Prøv begge stavemåter for sulfur — sjekk også for wppm-enhet
    result["sulfur_pct"] = find_value(df, "Total Sulfur", 1, 2)
    if result["sulfur_pct"] is None:
        result["sulfur_pct"] = find_value(df, "Total Sulphur", 1, 2)
    # Konverter wppm → % hvis enheten er wppm
    if result["sulfur_pct"] is not None:
        for r in range(df.shape[0]):
            cell = df.iat[r, 1] if 1 < df.shape[1] else None
            if isinstance(cell, str) and "total sulfur" in cell.lower():
                if "wppm" in cell.lower() or "ppm" in cell.lower():
                    result["sulfur_pct"] = result["sulfur_pct"] / 10000.0
                break

    result["pour_point_c"] = find_value(df, "Pour Point", 1, 2)
    result["viscosity_cst_20c"] = find_value(df, "Viscosity @ 20", 1, 2)
    result["viscosity_cst_40c"] = find_value(df, "Viscosity @ 40", 1, 2)
    result["nitrogen_ppm"] = find_value(df, "Total Nitrogen", 1, 2)
    result["basic_nitrogen_ppm"] = find_value(df, "Basic Nitrogen", 1, 2)

    result["mercaptan_sulphur_ppm"] = find_value(df, "Mercaptan Sulfur", 1, 2)
    if result["mercaptan_sulphur_ppm"] is None:
        result["mercaptan_sulphur_ppm"] = find_value(df, "Mercaptan Sulphur", 1, 2)

    result["rvp_psi"] = find_value(df, "Reid Vapour Pressure", 1, 2)
    if result["rvp_psi"] is None:
        result["rvp_psi"] = find_value(df, "Reid Vapor Pressure", 1, 2)

    result["asphaltenes_pct"] = find_value(df, "C7 Asphaltenes", 1, 2)
    result["mcr_pct"] = find_value(df, "Micro Carbon Residue", 1, 2)
    result["ccr_pct"] = find_value(df, "Conradson Carbon", 1, 2)
    if result["ccr_pct"] is None:
        result["ccr_pct"] = find_value(df, "Rams. Carbon Residue", 1, 2)
    result["vanadium_ppm"] = find_value(df, "Vanadium", 1, 2)
    result["nickel_ppm"] = find_value(df, "Nickel", 1, 2)
    result["tan_mgkoh"] = find_value(df, "Total Acid Number", 1, 2)
    result["wax_pct"] = find_value(df, "Total Wax", 1, 2)
    if result["wax_pct"] is None:
        result["wax_pct"] = find_value(df, "Wax", 1, 2)
    result["paraffins_pct"] = find_value(df, "Paraffins", 1, 2)
    result["naphthenes_pct"] = find_value(df, "Naphthenes", 1, 2)
    result["aromatics_pct"] = find_value(df, "Aromatics", 1, 2)
    result["hydrogen_pct"] = find_value(df, "Hydrogen", 1, 2)
    result["uopk"] = find_value(df, "UOPK", 1, 2)

    # Fallbacks
    if result["api_gravity"] is None:
        result["api_gravity"] = find_value_multi_col(df, "API Gravity")
    if result["sulfur_pct"] is None:
        result["sulfur_pct"] = find_value_multi_col(df, "Total Sulfur")

    # Hent assay-dato
    for r in range(df.shape[0]):
        cell = df.iat[r, 1] if 1 < df.shape[1] else None
        if isinstance(cell, str) and "assay date" in cell.lower():
            v = df.iat[r, 2] if 2 < df.shape[1] else None
            if pd.notna(v):
                try:
                    result["assay_date"] = pd.Timestamp(v).strftime("%Y-%m-%d")
                except Exception:
                    result["assay_date"] = str(v)
            break

    # Hent origin
    for r in range(df.shape[0]):
        cell = df.iat[r, 1] if 1 < df.shape[1] else None
        if isinstance(cell, str) and "origin" in cell.lower():
            v = df.iat[r, 2] if 2 < df.shape[1] else None
            if pd.notna(v) and isinstance(v, str):
                result["origin"] = v.strip()
            break

    # Destillasjonskutt
    cuts = parse_distillation_cuts(df)
    result.update(cuts)

    return result


def main() -> None:
    index = json.loads(INDEX_FILE.read_text())
    print(f"=== Parser {len(index)} ExxonMobil assays ===\n")

    rows = []
    success, partial, failed = 0, 0, 0
    for item in index:
        fname = item["local_file"]
        path = ASSAY_DIR / fname
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
            "grade_exxonmobil": item["grade"],
            "source": "ExxonMobil",
            "source_url": item["url"],
            "source_date": parsed.pop("assay_date", ""),
            "origin": parsed.pop("origin", ""),
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

        if has_api:
            status = "✓" if has_cuts else "△"
            success += 1 if has_cuts else 0
            partial += 0 if has_cuts else 1
        else:
            status = "✗"
            failed += 1

        api_s = f"{row.get('api_gravity', 0):5.1f}" if row.get("api_gravity") else "  N/A"
        sul_s = f"{row.get('sulfur_pct', 0):5.2f}" if row.get("sulfur_pct") else "  N/A"
        hvp = f"{row.get('high_value_yield_pct', 0):5.1f}" if row.get("high_value_yield_pct") else "  N/A"
        print(f"  {status} {item['grade']:35s} | API {api_s} | S {sul_s}% | HVY {hvp}% | {row.get('origin', '')}")

        rows.append(row)

    # Lagre
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n=== Resultat ===")
    print(f"  Vellykket:  {success}")
    print(f"  Delvis:     {partial}")
    print(f"  Feilet:     {failed}")
    print(f"  Output:     {OUTPUT_CSV}")
    print(f"  Kolonner:   {len(df_out.columns)}")

    # Quick QC
    if not df_out.empty:
        for col in ["api_gravity", "sulfur_pct", "vgo_pct", "vacuum_resid_pct", "high_value_yield_pct"]:
            if col in df_out.columns:
                vals = df_out[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:25s}: mean={vals.mean():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")


if __name__ == "__main__":
    main()
