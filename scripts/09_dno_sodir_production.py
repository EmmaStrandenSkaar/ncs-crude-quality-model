"""
Steg 9: Hent faktisk produksjon for DNOs North Sea-felt fra SODIR FactPages,
kombiner med kvalitetsdata og vis det komplette spreaden.

Datakilde:
  SODIR FactPages månedlig produksjon per felt (offentlig, gratis CSV):
  https://factpages.sodir.no/en/field/TableView/Production/Saleable/Monthly

MERK om API/svovel:
  SODIR har IKKE API-grad eller svovelinnhold i sine felt-tabeller. Den dataen
  finnes i operatørenes crude assays. Vi bruker vår crude_quality.csv for det.

Kurdistan:
  DNOs Kurdistan-produksjon (Tawke-lisensen) er ikke på NCS og rapporteres
  ikke til SODIR. Vi bruker DNOs egne rapporter (~70,100 boepd gross i 2025,
  ~52,600 boepd net = 75% eierandel).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_CSV = PROJECT_ROOT / "data" / "raw" / "crude_quality.csv"
MODEL_JSON = PROJECT_ROOT / "data" / "model" / "model_E_bfoet.json"
DIFF_CSV = PROJECT_ROOT / "data" / "processed" / "normpris_differentials_long.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "sodir"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SODIR_PROD_URL = (
    "https://factpages.sodir.no/public?/Factpages/external/tableview/"
    "field_production_monthly&rs:Command=Render&rc:Toolbar=false&"
    "rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
)
SODIR_FIELD_URL = (
    "https://factpages.sodir.no/public?/Factpages/external/tableview/"
    "field&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&"
    "IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
)

SM3_TO_BBL = 6.2898          # 1 Sm³ = 6.2898 bbl
DAYS_PER_MONTH = 365.25 / 12
GJ_PER_BSCM = 1e9            # rough: 1 Bsm3 gas ≈ ~5.8 MMBTU ≈ ~1 boe per Sm3

# DNO North Sea field map: sodir_name -> (crude_quality_key, dno_ownership_pct, note)
# Ownership from public filings / investor presentations.
DNO_NCS_FIELDS = {
    "EKOFISK":       ("EKOFISK",      7.60,  "Sval (Ekofisk complex)"),
    "ELDFISK":       ("EKOFISK",      7.60,  "Sval (Ekofisk complex)"),
    "EMBLA":         ("EKOFISK",      7.60,  "Sval (Ekofisk complex)"),
    "TOR":           ("EKOFISK",      7.60,  "Sval (Ekofisk complex)"),
    "MARTIN LINGE":  ("MARTIN LINGE", 15.00, "Sval"),
    "KVITEBJØRN":    ("ÅSGARD",       10.00, "Sval - cond tied to Troll area"),
    "NOVA":          (None,           15.00, "Sval - quality estimert"),
    "MARIA":         ("HEIDRUN",      10.00, "Sval - tieback til Heidrun"),
    "GUDRUN":        ("GUDRUN",       10.00, "Sval"),
    "MARULK":        (None,           50.00, "DNO Norge - gass"),
    "TRYM":          (None,           50.00, "DNO Norge - gass/kond"),
    "VERDANDE":      ("NORNE",        14.83, "DNO - Norne area, opp sent 2025"),
}

# Kurdistan (ikke SODIR - fra DNO årsrapport 2025)
KURDISTAN = {
    "TAWKE":     {"api": 28.0, "sulfur": 3.50, "dno_pct": 75.0, "gross_kbpd": 45.0},
    "PESHKABIR": {"api": 26.0, "sulfur": 3.80, "dno_pct": 75.0, "gross_kbpd": 15.0},
}
KURDISTAN_DIFF = -8.0   # estimert; typisk $6-12 rabatt til Brent

# West Africa (Côte d'Ivoire)
WEST_AFRICA_NET_KBPD = 3.3
WEST_AFRICA_DIFF = -0.10


def fetch_sodir(url: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"  bruker cache: {cache_path.name}")
        return pd.read_csv(cache_path, encoding="utf-8-sig")
    print(f"  laster ned fra SODIR: {cache_path.name}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return pd.read_csv(cache_path, encoding="utf-8-sig")


def get_2025_avg_production(prod: pd.DataFrame) -> pd.DataFrame:
    """Gjennomsnittlig månedlig produksjon per felt i 2025."""
    p25 = prod[prod["prfYear"] == 2025].copy()
    agg = p25.groupby("prfInformationCarrier").agg(
        oil_msm3=("prfPrdOilNetMillSm3",   "mean"),
        gas_bsm3=("prfPrdGasNetBillSm3",   "mean"),
        ngl_msm3=("prfPrdNGLNetMillSm3",   "mean"),
        cond_msm3=("prfPrdCondensateNetMillSm3", "mean"),
        n_months=("prfYear", "size"),
    ).reset_index().rename(columns={"prfInformationCarrier": "sodir_name"})

    # Konverter til kbpd
    agg["oil_kbpd"]  = agg["oil_msm3"]  * 1000 * SM3_TO_BBL / DAYS_PER_MONTH
    agg["ngl_kbpd"]  = agg["ngl_msm3"]  * 1000 * SM3_TO_BBL / DAYS_PER_MONTH
    agg["cond_kbpd"] = agg["cond_msm3"] * 1000 * SM3_TO_BBL / DAYS_PER_MONTH
    # Gass: 1 Sm3 ≈ 1 boe (rough)
    agg["gas_kboepd"] = agg["gas_bsm3"] * 1e9 / 1e3 / DAYS_PER_MONTH
    agg["total_kboepd"] = agg["oil_kbpd"] + agg["ngl_kbpd"] + agg["cond_kbpd"] + agg["gas_kboepd"]
    return agg


def build_ncs_rows(prod_agg: pd.DataFrame, quality: pd.DataFrame) -> list[dict]:
    rows = []
    for sodir_name, (quality_key, dno_pct, note) in DNO_NCS_FIELDS.items():
        p = prod_agg[prod_agg["sodir_name"] == sodir_name]
        if p.empty:
            gross_oil = np.nan
            gross_boe = np.nan
            n_months = 0
        else:
            gross_oil = float(p["oil_kbpd"].iloc[0])
            gross_boe = float(p["total_kboepd"].iloc[0])
            n_months = int(p["n_months"].iloc[0])

        dno_oil_kbpd = gross_oil * dno_pct / 100 if not np.isnan(gross_oil) else np.nan
        dno_boe_kbpd = gross_boe * dno_pct / 100 if not np.isnan(gross_boe) else np.nan

        # Kvalitetsdata
        if quality_key and quality_key in quality["field"].values:
            q = quality[quality["field"] == quality_key].iloc[0]
            api   = float(q["api_gravity"])
            sulfur = float(q["sulfur_pct"])
            q_conf = q["confidence"]
        else:
            api = np.nan; sulfur = np.nan; q_conf = "ukjent"

        rows.append(dict(
            field=sodir_name, region="Nordsjoen", note=note,
            dno_pct=dno_pct, quality_key=quality_key,
            gross_oil_kbpd=gross_oil, dno_oil_kbpd=dno_oil_kbpd,
            gross_boe_kbpd=gross_boe, dno_boe_kbpd=dno_boe_kbpd,
            api=api, sulfur=sulfur, q_conf=q_conf, n_sodir_months=n_months,
        ))
    return rows


def get_differential(field: str, region: str, diff_df: pd.DataFrame) -> tuple[float, str]:
    """Hent faktisk snitt-diff fra normpris hvis tilgjengelig."""
    if region == "Kurdistan":
        return KURDISTAN_DIFF, "markedsestimat"
    if region == "VestAfrika":
        return WEST_AFRICA_DIFF, "estimat"
    m = diff_df[diff_df["field"].str.upper() == field.upper()]
    if not m.empty:
        return float(m["differential_usd"].mean()), "normpris"
    return np.nan, "ingen data"


def main() -> None:
    print("Henter data...")
    prod = fetch_sodir(SODIR_PROD_URL, CACHE_DIR / "sodir_field_production_monthly.csv")
    quality = pd.read_csv(QUALITY_CSV)
    quality["field"] = quality["field"].str.upper().str.strip()
    diff = pd.read_csv(DIFF_CSV)
    diff["field"] = diff["field"].str.upper().str.strip()

    prod_agg = get_2025_avg_production(prod)

    # NCS-rader
    ncs_rows = build_ncs_rows(prod_agg, quality)

    # Kurdistan-rader
    for kname, kdata in KURDISTAN.items():
        ncs_rows.append(dict(
            field=kname, region="Kurdistan", note="Tawke-lisensen (75%, operator)",
            dno_pct=kdata["dno_pct"],
            gross_oil_kbpd=kdata["gross_kbpd"],
            dno_oil_kbpd=kdata["gross_kbpd"] * kdata["dno_pct"] / 100,
            gross_boe_kbpd=kdata["gross_kbpd"],
            dno_boe_kbpd=kdata["gross_kbpd"] * kdata["dno_pct"] / 100,
            api=kdata["api"], sulfur=kdata["sulfur"],
            q_conf="medium", n_sodir_months=0, quality_key=None,
        ))

    # Vest-Afrika
    ncs_rows.append(dict(
        field="CI-26", region="VestAfrika", note="Côte d'Ivoire",
        dno_pct=27.0, gross_oil_kbpd=WEST_AFRICA_NET_KBPD / 0.27,
        dno_oil_kbpd=WEST_AFRICA_NET_KBPD,
        gross_boe_kbpd=WEST_AFRICA_NET_KBPD / 0.27,
        dno_boe_kbpd=WEST_AFRICA_NET_KBPD,
        api=35.0, sulfur=0.20, q_conf="low", n_sodir_months=0, quality_key=None,
    ))

    df = pd.DataFrame(ncs_rows)

    # Legg til differensial
    df[["diff_usd", "diff_source"]] = df.apply(
        lambda r: pd.Series(get_differential(r["quality_key"] or r["field"], r["region"], diff)),
        axis=1,
    )

    # Produksjonsvektet bidrag til realisert pris
    df["dno_prod_for_weight"] = df["dno_oil_kbpd"].fillna(0)
    df["rev_contribution"] = df["dno_prod_for_weight"] * df["diff_usd"].fillna(0)

    # === Sammendragstabell ===
    print("\n" + "="*110)
    print("DNO PORTEFØLJE — SODIR-PRODUKSJON + KVALITET + DIFFERENSIAL (2025)")
    print("Merk: NCS-produksjon fra SODIR. Kurdistan/WA fra DNO årsrapport.")
    print("API/svovel fra crude assays (ikke SODIR — SODIR har ikke kvalitetsdata)")
    print("="*110)
    show_cols = ["field","region","dno_pct","dno_oil_kbpd","dno_boe_kbpd",
                 "api","sulfur","diff_usd","diff_source","q_conf","n_sodir_months"]
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # Produksjonsvektet differensial per region
    print("\n=== Produksjonsvektet differensial per region ===")
    for region in ["Kurdistan","Nordsjoen","VestAfrika"]:
        sub = df[(df["region"] == region) & (df["dno_prod_for_weight"] > 0) & df["diff_usd"].notna()]
        if sub.empty:
            continue
        wtd = np.average(sub["diff_usd"], weights=sub["dno_prod_for_weight"])
        tot = sub["dno_oil_kbpd"].sum()
        tot_boe = sub["dno_boe_kbpd"].sum()
        print(f"  {region:<15} {tot:>6.1f} kbpd olje | {tot_boe:>6.1f} kboepd | vektet diff: {wtd:+.2f} USD/fat")

    sub_all = df[df["dno_prod_for_weight"] > 0]
    wtd_all = np.average(sub_all["diff_usd"].fillna(0), weights=sub_all["dno_prod_for_weight"])
    print(f"  {'TOTALT':<15} {sub_all['dno_oil_kbpd'].sum():>6.1f} kbpd olje | vektet diff: {wtd_all:+.2f} USD/fat")

    # Lagre
    out_csv = OUT_DIR / "09_dno_sodir_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nTabell lagret: {out_csv}")

    # === Plott ===
    REGION_COLORS = {"Kurdistan": "#c0392b", "Nordsjoen": "#2980b9", "VestAfrika": "#27ae60"}
    df_plot = df[df["api"].notna() & df["sulfur"].notna()].copy()
    # Størrelse basert på faktisk DNO-netto-produksjon
    df_plot["marker_size"] = 80 + df_plot["dno_oil_kbpd"].fillna(1) * 30

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    # Venstre: API vs svovel
    ax = axes[0]
    for region, color in REGION_COLORS.items():
        sub = df_plot[df_plot["region"] == region]
        ax.scatter(sub["api"], sub["sulfur"], s=sub["marker_size"],
                   c=color, alpha=0.75, edgecolor="black", lw=0.7, label=region, zorder=3)
        for _, row in sub.iterrows():
            label = f"{row['field'].title()}\n({row['dno_oil_kbpd']:.1f}k)" if pd.notna(row['dno_oil_kbpd']) else row['field'].title()
            ax.annotate(label, (row["api"], row["sulfur"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=7.5)

    # Sweet spot
    ax.add_patch(plt.Rectangle((30, 0), 12, 0.5, alpha=0.07, color="green", zorder=1))
    ax.text(30.5, 0.02, "Raffineri sweet spot", fontsize=8, color="darkgreen", alpha=0.7)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.4, lw=1)
    ax.text(50.5, 0.52, "sweet/sour grense", fontsize=8, color="gray")
    ax.set_xlabel("API-grad (høyere = lettere)")
    ax.set_ylabel("Svovel (%)")
    ax.set_title("DNO: oljekvalitet per felt\n(størrelse = DNO netto oljeprod. kbpd)\nKilde: SODIR prod. + operator assays")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(22, 57)
    ax.set_ylim(-0.1, 4.3)

    # Høyre: waterfall / bar chart - DNO netto produksjon per felt farget etter region
    ax = axes[1]
    df_bar = df_plot.sort_values("dno_oil_kbpd", ascending=True).dropna(subset=["dno_oil_kbpd"])
    colors_bar = [REGION_COLORS[r] for r in df_bar["region"]]
    bars = ax.barh(df_bar["field"].str.title(), df_bar["dno_oil_kbpd"],
                   color=colors_bar, alpha=0.8, edgecolor="black", lw=0.5)
    # Annotate med diff
    for i, (_, row) in enumerate(df_bar.iterrows()):
        if pd.notna(row["diff_usd"]):
            ax.text(row["dno_oil_kbpd"] + 0.05, i,
                    f"{row['diff_usd']:+.1f}$/fat ({row['diff_source']})",
                    va="center", fontsize=7.5)
    ax.set_xlabel("DNO netto oljeproduksjon (kbpd, 2025)")
    ax.set_title("DNO: netto produksjon per felt\nmed estimert differensial mot Brent")
    ax.grid(True, axis="x", alpha=0.3)

    # Legg til legend for region
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    fig.suptitle(f"DNO ASA — Oljekvalitet & Pris-eksponering (Q2 2026)\n"
                 f"NCS-produksjon: SODIR FactPages | API/svovel: operator assays | Kurdistan: DNO rapporter",
                 fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / "09_dno_sodir_spread.png"
    fig.savefig(out_png, dpi=130)
    print(f"Plott lagret: {out_png}")


if __name__ == "__main__":
    main()
