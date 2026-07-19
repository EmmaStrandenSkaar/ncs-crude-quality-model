"""
Script 67 — NCS decline-benchmarks per feltstørrelsesklasse
====================================================================
Gjenskaper "Tail & Benchmarks"-metodikken fra Aker BP-modellen for hele
NCS: median år-over-år decline per år-etter-peak, gruppert etter feltets
endelige størrelse (ultimate recovery):

  Giant ≥ 1000 | Stor 300–1000 | Mellom 100–300 | Liten 30–100 | Marginal < 30
  (mmboe)

Brukes av volummodulen (68) til 2U-haler: analog-median decline med
gulv −5 %/år og cutoff 2 % av peak.

Valideringsankere (modellens verdier, år 1 etter peak):
  Giant −8.53 %, Stor −8.85 %, Mellom −14.39 %, Liten −22.29 %,
  Marginal −37.31 %, Alle −17.69 %

Output: data/processed/nav/decline_benchmarks.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "sodir"
OUT_DIR = ROOT / "data" / "processed" / "nav"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "decline_benchmarks.csv"

SM3_TO_BBL = 6.2898
CLASSES = [("Giant", 1000, np.inf), ("Stor", 300, 1000), ("Mellom", 100, 300),
           ("Liten", 30, 100), ("Marginal", 0, 30)]
MAX_YEARS_PAST_PEAK = 30
MIN_HISTORY_YEARS = 4          # krever litt historikk for meningsfull peak
CUTOFF_PCT_OF_PEAK = 0.02      # haleregel (dokumentert, brukes i script 68)
DECLINE_FLOOR = -0.05          # haleregel (dokumentert, brukes i script 68)


def size_class(mmboe: float) -> str:
    for name, lo, hi in CLASSES:
        if lo <= mmboe < hi:
            return name
    return "Marginal"


def main() -> None:
    print("=" * 60)
    print("SCRIPT 67: NCS decline-benchmarks (median per størrelsesklasse)")
    print("=" * 60)

    prod = pd.read_csv(RAW / "sodir_field_production_yearly.csv", encoding="utf-8-sig")
    prod = prod.rename(columns={"prfInformationCarrier": "field",
                                "prfYear": "year", "prfPrdOeNetMillSm3": "oe"})
    prod["field"] = prod.field.str.upper().str.strip()
    annual = prod.groupby(["field", "year"], as_index=False).oe.sum()

    # Ultimate recovery per felt: nyeste reservevintage (utvinnbar OE)
    res = pd.read_csv(RAW / "sodir_field_reserves.csv", encoding="utf-8-sig")
    res["field"] = res.fldName.str.upper().str.strip()
    latest = res.sort_values("fldVersion").groupby("field").tail(1)
    ur_mmboe = (latest.set_index("field").fldRecoverableOE * SM3_TO_BBL).to_dict()

    # decline-observasjoner: per felt, år-over-år etter peak-året
    obs = []
    n_fields = 0
    for field, g in annual.groupby("field"):
        g = g.sort_values("year").reset_index(drop=True)
        g = g[g.oe > 0]
        if len(g) < MIN_HISTORY_YEARS or field not in ur_mmboe:
            continue
        n_fields += 1
        peak_i = int(g.oe.idxmax())
        peak_year = int(g.loc[peak_i, "year"])
        peak_oe = float(g.loc[peak_i, "oe"])
        cls = size_class(ur_mmboe[field])
        gg = g.set_index("year").oe
        for k in range(1, MAX_YEARS_PAST_PEAK + 1):
            y = peak_year + k
            if y in gg.index and (y - 1) in gg.index and gg[y - 1] > 0:
                obs.append({"field": field, "size_class": cls,
                            "years_past_peak": k,
                            "decline": gg[y] / gg[y - 1] - 1.0,
                            "pct_of_peak": gg[y] / peak_oe})
    df = pd.DataFrame(obs)
    print(f"\nFelt med historikk + reserver: {n_fields}, observasjoner: {len(df)}")

    # medianer per klasse × år-etter-peak (+ Alle)
    med = (df.groupby(["size_class", "years_past_peak"])
             .agg(median_decline=("decline", "median"), n=("decline", "size"))
             .reset_index())
    all_ = (df.groupby("years_past_peak")
              .agg(median_decline=("decline", "median"), n=("decline", "size"))
              .reset_index())
    all_["size_class"] = "Alle"
    bench = pd.concat([med, all_], ignore_index=True)
    bench["cutoff_pct_of_peak"] = CUTOFF_PCT_OF_PEAK
    bench["decline_floor"] = DECLINE_FLOOR
    bench.to_csv(OUT, index=False)

    # valider mot modellens år-1-medianer
    ANCHORS = {"Giant": -8.53, "Stor": -8.85, "Mellom": -14.39,
               "Liten": -22.29, "Marginal": -37.31, "Alle": -17.69}
    print("\nÅr 1 etter peak — median decline vs modellens ankere:")
    yr1 = bench[bench.years_past_peak == 1].set_index("size_class")
    for cls, anchor in ANCHORS.items():
        if cls in yr1.index:
            v = yr1.loc[cls, "median_decline"] * 100
            n = int(yr1.loc[cls, "n"])
            diff = v - anchor
            flag = "✓" if abs(diff) < 3.0 else "✗"
            print(f"  {cls:9s} {v:7.2f}%  (anker {anchor:7.2f}%, avvik {diff:+5.2f} pp, n={n}) {flag}")
        else:
            print(f"  {cls:9s} — mangler")

    print(f"\nLagret: {OUT}")


if __name__ == "__main__":
    main()
