"""
Parse Equinor assay XLSX-er → strukturert CSV.

For hver assay henter vi:
  Whole crude:
    - api_gravity, sulfur_pct, tan_mgkoh
    - pour_point_c, viscosity_cst_20c
    - nitrogen_ppm, vanadium_ppm, nickel_ppm
    - asphaltenes_pct, mcr_pct (Micro Carbon Residue), ccr_pct (Rams.)
    - wax_pct, mercaptan_sulphur_ppm
    - rvp_psi

  Distillation yields (vol%) — mappet til våre standard-kutt:
    - lpg_pct (IBP-C4)
    - light_naphtha_pct (C5-65°C)
    - heavy_naphtha_pct (65-150°C)
    - kerosene_pct (150-250°C)
    - diesel_pct (250-350°C)
    - heavy_diesel_pct (350-370°C)
    - vgo_pct (370-550°C)
    - vacuum_resid_pct (>550°C)

  Aggregater:
    - naphtha_pct = light + heavy naphtha
    - middle_distillate_pct = kerosene + diesel
    - bottom_of_barrel_pct = VGO + vacuum residue
    - high_value_yield_pct = naphtha + middle distillate (excl LPG, VGO, residue)

  Pluss kvalitative felt:
    - source_url, source_date, grade_raw (Equinor's navn)
"""

from pathlib import Path
import json
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
ASSAY_DIR = PROJECT_ROOT / "data" / "raw" / "equinor_assays"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "equinor_assay_index.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "verified_crude_assays.csv"
RAW_CUTS_CSV = PROJECT_ROOT / "data" / "raw" / "verified_assays_full_cuts.csv"


def safe_filename(grade: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", grade).strip("_").lower()


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
                # Prøv flere mulige verdi-kolonner
                cols_to_try = value_cols if value_cols else [c + 1, c + 2, c + 3]
                for vc in cols_to_try:
                    if vc < df.shape[1]:
                        v = df.iat[r, vc]
                        if pd.notna(v):
                            try:
                                fv = float(v)
                                return fv
                            except (TypeError, ValueError):
                                continue
    return None


def parse_distillation_cuts(df: pd.DataFrame) -> dict:
    """Hent yield-fordeling fra cut data-tabellen.

    Strukturen er:
      r34: Start (°C) → IBP, C5, 65, 100, 150, 200, 250, 300, 350, 370 (atmos)
                       + 370, 450, 500, 550 (vacuum)
      r35: End (°C)   → C4, 65, 100, 150, 200, 250, 300, 350, 370, FBP
                       + 450, 500, 550, FBP
      r39: Yield (% vol)  ← vi tar volum, ikke vekt

    Vi mapper til:
      lpg = IBP-C4 (1. cut)
      light_naphtha = C5-65
      heavy_naphtha = 65-150 (kan være 2 sub-cuts)
      kerosene = 150-250
      diesel = 250-350
      heavy_diesel = 350-370
      vgo = 370-550 (3 cuts)
      vacuum_resid = >550
    """
    # Finn rad for "Yield (% vol)" eller "Yield (% wt)" — vol foretrekkes
    yield_row = None
    use_col_label = None
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
        # Prøv yield (% wt) som fallback
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

    # Finn Start (°C)-raden for å mappe kolonner til kutt-grenser
    start_row = None
    for r in range(df.shape[0]):
        cell = df.iat[r, 1] if 1 < df.shape[1] else None
        if isinstance(cell, str) and "start" in cell.lower() and "°c" in cell.lower():
            start_row = r
            break

    if start_row is None:
        return {}

    # Bygg map fra kolonne → start-temperatur (eller "IBP", "C5", osv.)
    col_to_start = {}
    for c in range(2, df.shape[1]):
        v = df.iat[start_row, c]
        if pd.notna(v):
            col_to_start[c] = str(v).strip()

    # Hent også slutt-labels
    end_row = start_row + 1
    col_to_end = {}
    for c in col_to_start:
        if end_row < df.shape[0]:
            v = df.iat[end_row, c]
            if pd.notna(v):
                col_to_end[c] = str(v).strip()

    def parse_temp(s: str) -> float | None:
        """IBP, C4, C5, FBP, eller numerisk temp i °C."""
        u = s.upper().strip()
        if u in ("IBP",):
            return 0.0
        if u in ("C4",):
            return 0.0  # ~butan = 0°C
        if u in ("C5",):
            return 36.0  # pentan koker 36°C
        if "FBP" in u:
            return 999.0  # final boiling point
        try:
            return float(u)
        except ValueError:
            return None

    # Bygg liste over kutt med (col, start_temp, end_temp, yield)
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

    # FILTRER: hopp over duplikat-"totaler" som har sub-cuts senere.
    # En "total" har end=FBP eller end > 500. Hvis senere kolonne har start >= denne kuttens start,
    # er denne kuttet en sum-rad som vi ikke skal dobbel-telle.
    final = []
    for i, (c, sl, el, st, et, y) in enumerate(cuts_list):
        is_total = "FBP" in el.upper() or et >= 999
        if is_total:
            # Sjekk om noen senere kutt har start >= denne kuttens start
            has_subcuts = any(
                cc > c and st2 >= st
                for cc, _, _, st2, _, _ in cuts_list[i+1:]
            )
            if has_subcuts:
                continue  # Hopp over — det er en duplikat-total
        # Hopp over hele-crude (IBP→FBP)
        if sl.upper() == "IBP" and ("FBP" in el.upper() or et >= 999):
            # Bare hvis det er en hel-crude. Sjekk: er det andre kutt i listen?
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
        # Bruk midt-temp for kategorisering
        mid = (st + et) / 2
        if mid < 35:  # IBP-C4 = LPG
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
    """Parse én Equinor assay XLSX."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Summary", header=None, engine="openpyxl")
    except Exception as e:
        print(f"    Kunne ikke lese {xlsx_path.name}: {e}")
        return None

    result = {}

    # Whole crude properties (label i kol 1, verdi i kol 2)
    result["api_gravity"] = find_value(df, "API Gravity", 1, 2)
    result["density_g_cc"] = find_value(df, "Density @ 15", 1, 2)
    result["sulfur_pct"] = find_value(df, "Total Sulphur", 1, 2)
    result["pour_point_c"] = find_value(df, "Pour Point", 1, 2)
    result["viscosity_cst_20c"] = find_value(df, "Viscosity @ 20", 1, 2)
    result["viscosity_cst_40c"] = find_value(df, "Viscosity @ 40", 1, 2)
    result["nitrogen_ppm"] = find_value(df, "Total Nitrogen", 1, 2)
    result["basic_nitrogen_ppm"] = find_value(df, "Basic Nitrogen", 1, 2)
    result["mercaptan_sulphur_ppm"] = find_value(df, "Mercaptan Sulphur", 1, 2)
    result["rvp_psi"] = find_value(df, "Reid Vapour Pressure", 1, 2)
    result["asphaltenes_pct"] = find_value(df, "C7 Asphaltenes", 1, 2)
    result["mcr_pct"] = find_value(df, "Micro Carbon Residue", 1, 2)
    result["ccr_pct"] = find_value(df, "Rams. Carbon Residue", 1, 2)
    if result["ccr_pct"] is None:
        # Fallback: Conradson Carbon Residue
        result["ccr_pct"] = find_value(df, "Conradson", 1, 2)
    result["vanadium_ppm"] = find_value(df, "Vanadium", 1, 2)
    result["nickel_ppm"] = find_value(df, "Nickel", 1, 2)
    result["tan_mgkoh"] = find_value(df, "Total Acid Number", 1, 2)
    result["wax_pct"] = find_value(df, "Wax", 1, 2)
    result["paraffins_pct"] = find_value(df, "Paraffins", 1, 2)
    result["naphthenes_pct"] = find_value(df, "Naphthenes", 1, 2)
    result["aromatics_pct"] = find_value(df, "Aromatics", 1, 2)
    result["hydrogen_pct"] = find_value(df, "Hydrogen", 1, 2)
    result["uopk"] = find_value(df, "UOPK", 1, 2)

    # Hvis nøkkelvariabler mangler, prøv alternative kolonner (oppe i headeren)
    if result["api_gravity"] is None:
        result["api_gravity"] = find_value_multi_col(df, "API Gravity")
    if result["sulfur_pct"] is None:
        result["sulfur_pct"] = find_value_multi_col(df, "Total Sulphur")

    # Destillasjonskutt
    cuts = parse_distillation_cuts(df)
    result.update(cuts)

    return result


def main() -> None:
    index = json.loads(INDEX_FILE.read_text())
    print(f"=== Parser {len(index)} Equinor assays ===\n")

    rows = []
    success, partial, failed = 0, 0, 0
    for item in index:
        fname = safe_filename(item["grade"]) + ".xlsx"
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

        # Sjekk om vi har de viktigste tallene
        critical = ["api_gravity", "sulfur_pct"]
        nice = ["vacuum_resid_pct", "ccr_pct", "vanadium_ppm"]
        crit_ok = all(parsed.get(k) is not None for k in critical)
        nice_count = sum(1 for k in nice if parsed.get(k) is not None and parsed.get(k) != 0)

        status = "✓✓" if crit_ok and nice_count >= 2 else "✓" if crit_ok else "?"
        print(f"  {status} {item['grade']:35s} | API={parsed.get('api_gravity'):>5.2f} "
              f"S={parsed.get('sulfur_pct') or 0:>5.2f}% | "
              f"VR={parsed.get('vacuum_resid_pct') or 0:>5.1f}% "
              f"CCR={parsed.get('ccr_pct') or 0:>5.2f} "
              f"V={parsed.get('vanadium_ppm') or 0:>6.1f}ppm")

        row = {
            "grade_equinor": item["grade"],
            "source": "Equinor",
            "source_url": item["url"],
            "source_date": item["date"],
            "description": item["description"],
            **parsed,
        }
        # Avledede aggregater
        if cuts := {k: parsed.get(k, 0) for k in ["light_naphtha_pct", "heavy_naphtha_pct",
                    "kerosene_pct", "diesel_pct", "heavy_diesel_pct", "vgo_pct", "vacuum_resid_pct"]}:
            row["naphtha_pct"] = cuts["light_naphtha_pct"] + cuts["heavy_naphtha_pct"]
            row["middle_distillate_pct"] = cuts["kerosene_pct"] + cuts["diesel_pct"] + cuts["heavy_diesel_pct"]
            row["bottom_of_barrel_pct"] = cuts["vgo_pct"] + cuts["vacuum_resid_pct"]
            row["high_value_yield_pct"] = row["naphtha_pct"] + row["middle_distillate_pct"]

        rows.append(row)
        if crit_ok and nice_count >= 2:
            success += 1
        elif crit_ok:
            partial += 1
        else:
            failed += 1

    df_out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n=== Oppsummering ===")
    print(f"  Full success: {success}/{len(index)}")
    print(f"  Partial:      {partial}/{len(index)}")
    print(f"  Failed:       {failed}/{len(index)}")
    print(f"  Lagret:       {OUTPUT_CSV}")
    print(f"  Kolonner:     {df_out.shape[1]}")
    print(f"\nDekning per kolonne:")
    for c in df_out.columns:
        if c in ("grade_equinor", "source", "source_url", "source_date", "description"):
            continue
        n = df_out[c].notna().sum()
        if n > 0 and pd.api.types.is_numeric_dtype(df_out[c]):
            mean = df_out[c].mean()
            print(f"  {c:30s}: {n:>2d}/{len(df_out)} obs, mean={mean:.2f}")


if __name__ == "__main__":
    main()
