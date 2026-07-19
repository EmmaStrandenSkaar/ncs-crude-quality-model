"""
Script 66 — Eierskapsmatrise: WI per (selskap, felt) og (selskap, funn)
====================================================================
Bygger den sentrale eierskapsmatrisen NAV-motoren netter ned med:

  FELT: fra sodir_company_reserves (reserver per selskap per felt) —
        WI (cmpShare) + netto gjenværende volumer per strøm, 31.12-årgang.
  FUNN: sodir_discovery → eier (utvinningstillatelse eller BAA) →
        aktive lisensandeler fra sodir_licence_licensee_hst.

Valideringsanker (fra Aker BP-modellen): Johan Sverdrup WI = 31.5733 %.

Output: data/processed/nav/ownership_matrix.csv
        (asset_type, asset_name, npdid, company, wi_pct,
         net_oil/gas/ngl/cond/oe_msm3 for felt)
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "sodir"
OUT_DIR = ROOT / "data" / "processed" / "nav"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "ownership_matrix.csv"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, encoding="utf-8-sig", low_memory=False)


def field_ownership() -> pd.DataFrame:
    cr = load("sodir_company_reserves.csv")
    df = pd.DataFrame({
        "asset_type": "field",
        "asset_name": cr.fldName.str.upper().str.strip(),
        "npdid": cr.fldNpdidField,
        "company": cr.cmpLongName.str.strip(),
        "wi_pct": cr.cmpShare,
        "net_oil_msm3": cr.cmpRemainingOil,
        "net_gas_bsm3": cr.cmpRemainingGas,
        "net_ngl_mt": cr.cmpRemainingNGL,
        "net_cond_msm3": cr.cmpRemainingCondensate,
        "net_oe_msm3": cr.cmpRemainingOE,
        "vintage": cr.cmpDateOffResEstDisplay,
    })
    return df


def discovery_ownership() -> pd.DataFrame:
    disc = load("sodir_discovery.csv")
    lic = load("sodir_licence_licensee_hst.csv")

    # aktive andeler: ValidTo tom eller 9999
    to = lic.prlLicenseeDateValidTo.astype(str)
    active = lic[lic.prlLicenseeDateValidTo.isna() | to.str.contains("9999")].copy()
    active["prlName"] = active.prlName.str.strip()

    # kun funn som ikke allerede er felt (feltene dekkes av company_reserves)
    d = disc[disc.fldName.isna() | (disc.fldName.astype(str).str.strip() == "")].copy()
    # eier via utvinningstillatelse: dscOwnerName = lisensnavn når OwnerKind er PL
    d["owner"] = d.dscOwnerName.astype(str).str.strip()

    rows = []
    unmatched = []
    for _, r in d.iterrows():
        shares = active[active.prlName == r.owner]
        if r.dscOwnerKind == "PRODUCTION LICENCE" and len(shares):
            for _, s in shares.iterrows():
                rows.append({
                    "asset_type": "discovery",
                    "asset_name": str(r.dscName).upper().strip(),
                    "npdid": r.dscNpdidDiscovery,
                    "company": s.cmpLongName.strip(),
                    "wi_pct": s.prlLicenseeInterest,
                    "owner_licence": r.owner,
                })
        else:
            # BAA (unitiserte områder) har ikke lisensandeler i tabellen —
            # eierskap må settes manuelt ved behov. Flagges med wi_pct NaN.
            unmatched.append({
                "asset_type": "discovery",
                "asset_name": str(r.dscName).upper().strip(),
                "npdid": r.dscNpdidDiscovery,
                "company": None,
                "wi_pct": None,
                "owner_licence": f"{r.dscOwnerKind}: {r.owner}",
            })
    return pd.DataFrame(rows + unmatched)


def main() -> None:
    print("=" * 60)
    print("SCRIPT 66: Eierskapsmatrise (WI per selskap per felt/funn)")
    print("=" * 60)

    fields = field_ownership()
    discs = discovery_ownership()
    matrix = pd.concat([fields, discs], ignore_index=True)
    matrix.to_csv(OUT, index=False)

    n_f = fields.asset_name.nunique()
    n_c = fields.company.nunique()
    print(f"\nFelt: {n_f} felt × {n_c} selskaper ({len(fields)} rader)")
    n_dm = discs.wi_pct.notna().sum()
    n_dd = discs[discs.wi_pct.notna()].asset_name.nunique()
    n_du = discs[discs.wi_pct.isna()].asset_name.nunique()
    print(f"Funn: {n_dd} funn med lisens-WI ({n_dm} andeler); {n_du} uten (BAA/mangler)")

    # WI-sum-kontroll per felt (skal være ~100)
    ws = fields.groupby("asset_name").wi_pct.sum()
    bad = ws[(ws < 99.5) | (ws > 100.5)]
    print(f"\nWI-sum-kontroll felt: {len(ws) - len(bad)}/{len(ws)} summerer til 100 ±0.5")
    if len(bad):
        print("  Avvik:", {k: round(v, 2) for k, v in bad.head(8).items()})

    # Valideringsanker
    js = fields[(fields.asset_name == "JOHAN SVERDRUP")
                & (fields.company.str.contains("Aker BP"))]
    if len(js):
        wi = js.wi_pct.iloc[0]
        ok = abs(wi - 31.5733) < 1e-3
        print(f"\nANKER Johan Sverdrup × Aker BP: WI = {wi} "
              f"({'✓ matcher modellen (31.5733)' if ok else '✗ AVVIK fra 31.5733!'})")

    print(f"\nLagret: {OUT}")


if __name__ == "__main__":
    main()
