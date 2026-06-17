"""
Script 42: Aker BP — Feltspesifikk realisert oljepris-dekomponering
====================================================================

TRINN 1 (VALIDERING):
  Bruk regresjonsmodell (34_parsimonious_model.json) til å predikere
  blended realisert oljepris for Aker BP per kvartal 2020–2026.
  Sammenlign mot rapportert realisert pris (AKRBP quarterly reports).

TRINN 2 (DEKOMPONERING):
  Vis bidrag per felt til total differensial mot Brent.
  Yggdrasil/NOAKA mix-shift frem mot 2028.

METODIKK:
  Realized_pred_q = Brent_q + Σ_i [ share_i_q × diff_i_q ]

  share_i_q = Aker BP netto produksjon felt i / total netto produksjon (kvartal q)
  diff_i_q  = modell-predikert differensial for felt i (statisk kvalitet × dynamiske markedsfeatures)

FELT → ASSAY-MAPPING:
  Johan Sverdrup : API 28.7, S 0.809%  (Equinor lab-assay, ★★★)
  Alvheim area   : API 34.5, S 0.402%  (Equinor lab-assay, ★★★)
  Valhall area   : API 36.0, S 0.100%  (industri-estimat fra Repsol/Yme-proxy, ★★)
  Hod            : API 36.0, S 0.100%  (Valhall-reservoir, ★★)
  Edvard Grieg   : API 32.0, S 0.420%  (estimert fra AKRBP investor presentasjoner, ★★)
  Ivar Aasen     : API 33.0, S 0.350%  (estimert fra AKRBP investor presentasjoner, ★★)
  Skarv          : API 50.8, S 0.064%  (Equinor lab-assay, kondensat/NGL, ★★★)
  Ula area       : API 35.0, S 0.090%  (estimert fra AKRBP investor presentasjoner, ★★)

RAPPORTERTE REALISERTE PRISER:
  Kilde: Aker BP ASA Quarterly Reports (akerbp.com/investor-relations)
  Enhet: USD/bbl, realized oil price (inkl. NGL-verdi og prising)
  OBS: Oppdater `AKRBP_REPORTED` med faktiske tall fra rapportene!
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT   = Path(__file__).parent.parent
PANEL_CSV      = PROJECT_ROOT / "data" / "processed" / "regression_panel.csv"
ASSAY_CSV      = PROJECT_ROOT / "data" / "processed" / "unified_crude_assays.csv"
MODEL_JSON     = PROJECT_ROOT / "data" / "processed" / "34b_brent_model.json"
SODIR_MONTHLY  = PROJECT_ROOT / "data" / "raw" / "sodir" / "sodir_field_production_monthly.csv"
IMPUTED_CSV    = PROJECT_ROOT / "data" / "processed" / "63_ncs_field_quality.csv"  # Script 63 fallback
OUT_DIR        = PROJECT_ROOT / "data" / "processed"
FIG_DIR        = PROJECT_ROOT / "data" / "processed"

# ── Aker BP eierandeler (working interest) ──────────────────────────────────
AKER_BP_WI = {
    "JOHAN SVERDRUP": 0.3157,
    "VALHALL":        0.900,
    "SKARV":          0.238,
    "ALVHEIM":        0.650,
    "EDVARD GRIEG":   0.650,
    "HOD":            0.900,
    "IVAR AASEN":     0.348,
    "ULA":            0.800,
    "BØYLA":          0.650,   # Alvheim area sub-felt
    "SKOGUL":         0.650,   # Alvheim area sub-felt
    "TAMBAR":         0.550,   # Ula area sub-felt
    "TAMBAR ØST":     0.550,
    "JETTE":          0.700,   # legacy, liten
}

# ── Feltspesifikke assay-egenskaper — KUN OFFISIELLE DATAKILDER ──────────────
# Kun felt med publisert offisiell crude assay inngår i differensialberegningen.
# Felt uten offentlig standalone-assay er ekskludert fra modellen.
#
# EKSKLUDERTE felt (ingen offentlig standalone-assay tilgjengelig):
#   Valhall, HOD     → eksportert som del av Ekofisk Blend via Norpipe/Teesside
#   Edvard Grieg     → eksportert som del av Grane Blend via Sture Terminal
#   Ivar Aasen       → co-prosessert med Edvard Grieg → Grane Blend
#   ULA, Tambar, Tambar Øst → eksportert som del av Ekofisk Blend
#   Jette            → nedlagt, ingen assay publisert
#
# Disse feltene inngår FORTSATT i produksjonsdataene (Sodir), men bidrar ikke
# til den produksjonsvektede differensialberegningen.
# Samlet ekskludert produksjon: ~19% av AKRBP total (Valhall 8.6%, EG 5.8%,
#   IA 2.1%, HOD 2.2%, ULA/Tambar ~1.5%).
#
# INKLUDERTE felt (offisiell Equinor XLSX-assay):
#   Johan Sverdrup ★★★, Alvheim ★★★, Skarv ★★★
#   Bøyla og Skogul: co-prosessert i Alvheim FPSO → eksportert som Alvheim Blend,
#   Equinor-assayen gjelder hele FPSO-eksportstreamen ★★★
FIELD_QUALITY = {
    "JOHAN SVERDRUP": dict(
        # Equinor offisiell XLSX-assay (sist oppdatert 2024)
        api=28.7, sulfur=0.809, vacuum_resid=19.04, ccr=4.21,
        vanadium=12.06, nickel=3.77,
        middle_distillate_pct=36.93,   # kerosene(14.3%) + diesel(19.4%) + heavy_diesel(3.2%)
        log_dist_rotterdam=6.7397,      # Sture terminal, 845 km
        confidence="★★★ Equinor lab-assay",
    ),
    "ALVHEIM": dict(
        # Equinor offisiell XLSX-assay (jan 2025)
        api=34.5, sulfur=0.402, vacuum_resid=8.72, ccr=1.12,
        vanadium=2.76, nickel=0.80,
        middle_distillate_pct=44.79,   # kerosene(18.8%) + diesel(22.3%) + heavy_diesel(3.7%)
        log_dist_rotterdam=6.3729,      # Alvheim FPSO, 586 km
        confidence="★★★ Equinor lab-assay",
    ),
    "BØYLA": dict(
        # Co-prosessert i Alvheim FPSO, eksportert som Alvheim Blend
        # Equinor Alvheim-assay gjelder hele FPSO-eksportstreamen
        api=34.5, sulfur=0.402, vacuum_resid=8.72, ccr=1.12,
        vanadium=2.76, nickel=0.80,
        middle_distillate_pct=44.79,
        log_dist_rotterdam=6.3729,      # Alvheim FPSO, 586 km
        confidence="★★★ Equinor lab-assay (Alvheim FPSO-stream)",
    ),
    "SKOGUL": dict(
        # Co-prosessert i Alvheim FPSO, eksportert som Alvheim Blend
        api=34.5, sulfur=0.402, vacuum_resid=8.72, ccr=1.12,
        vanadium=2.76, nickel=0.80,
        middle_distillate_pct=44.79,
        log_dist_rotterdam=6.3729,      # Alvheim FPSO, 586 km
        confidence="★★★ Equinor lab-assay (Alvheim FPSO-stream)",
    ),
    "SKARV": dict(
        # Equinor offisiell XLSX-assay (kondensat/NGL-stream)
        api=50.8, sulfur=0.064, vacuum_resid=1.75, ccr=0.24,
        vanadium=0.50, nickel=0.17,
        middle_distillate_pct=46.71,   # kerosene(29.0%) + diesel(16.2%) + heavy_diesel(1.5%)
        log_dist_rotterdam=7.3417,      # Skarv FPSO, 1543 km
        confidence="★★★ Equinor lab-assay (kondensat/NGL)",
    ),
    # ── Blend-assay-proxy: offisielle Equinor XLSX-assays for eksportblendene ──
    # Disse feltene har ingen publisert standalone-assay, men oljen selges
    # som del av én navngitt blend-grade med offisiell Equinor-assay.
    # Vi bruker blend-assayen siden det er den faktiske salgsvaren.
    #
    # Valhall, HOD, ULA, Tambar, Tambar Øst → Ekofisk Blend (Norpipe/Teesside)
    #   OBS: Blend inkluderer Ekofisk, Eldfisk, Embla, Valhall, HOD, ULA, Gyda,
    #        Tambar m.fl. API 38.9° er noe høyere enn Valhall isolert (~36°).
    "VALHALL": dict(
        api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43,
        vanadium=1.87, nickel=3.00,
        middle_distillate_pct=38.82,   # Ekofisk Blend assay
        log_dist_rotterdam=6.2567,      # Ekofisk SBM / Teesside, 521 km
        confidence="★★★ Equinor lab-assay (Ekofisk Blend — faktisk eksportstream)",
    ),
    "HOD": dict(
        api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43,
        vanadium=1.87, nickel=3.00,
        middle_distillate_pct=38.82,
        log_dist_rotterdam=6.2567,
        confidence="★★★ Equinor lab-assay (Ekofisk Blend — faktisk eksportstream)",
    ),
    "ULA": dict(
        api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43,
        vanadium=1.87, nickel=3.00,
        middle_distillate_pct=38.82,
        log_dist_rotterdam=6.2567,
        confidence="★★★ Equinor lab-assay (Ekofisk Blend — faktisk eksportstream)",
    ),
    "TAMBAR": dict(
        api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43,
        vanadium=1.87, nickel=3.00,
        middle_distillate_pct=38.82,
        log_dist_rotterdam=6.2567,
        confidence="★★★ Equinor lab-assay (Ekofisk Blend — faktisk eksportstream)",
    ),
    "TAMBAR ØST": dict(
        api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43,
        vanadium=1.87, nickel=3.00,
        middle_distillate_pct=38.82,
        log_dist_rotterdam=6.2567,
        confidence="★★★ Equinor lab-assay (Ekofisk Blend — faktisk eksportstream)",
    ),
    # Edvard Grieg, Ivar Aasen → Grane Blend (Sture Terminal)
    #   OBS: Grane Blend domineres av Grane-feltet (API ~19°, tung olje).
    #        EG/IA er lettere (~API 32-33°), men blend-assayen (API 27.1°)
    #        representerer den faktiske salgsvaren ved Sture.
    "EDVARD GRIEG": dict(
        api=27.1, sulfur=0.668, vacuum_resid=19.00, ccr=3.27,
        vanadium=11.07, nickel=3.52,
        middle_distillate_pct=38.19,   # Grane Blend assay
        log_dist_rotterdam=6.8667,      # Sture terminal, 960 km
        confidence="★★★ Equinor lab-assay (Grane Blend — faktisk eksportstream)",
    ),
    "IVAR AASEN": dict(
        api=27.1, sulfur=0.668, vacuum_resid=19.00, ccr=3.27,
        vanadium=11.07, nickel=3.52,
        middle_distillate_pct=38.19,
        log_dist_rotterdam=6.8667,      # Sture terminal, 960 km
        confidence="★★★ Equinor lab-assay (Grane Blend — faktisk eksportstream)",
    ),
    # Jette: nedlagt felt, ingen assay — ikke inkludert
}

# ── Rapporterte realiserte priser (AKRBP quarterly reports) ─────────────────
# KILDE: Aker BP ASA Quarterly Reports — akerbp.com/investor-relations
# → Investor Relations → Quarterly Reports → "Realized oil price"
# ENHET: USD/bbl realized oil price (rapportert av selskapet)
# OBS: Alle verdier er hentet direkte fra offisielle kvartalsrapporter (PDF-parsing + kryssjekk).

AKRBP_REPORTED = {
    # Format: "YYYY-Q{q}": USD_per_boe
    # KILDE: Aker BP ASA Quarterly Reports — akerbp.com/investor-relations
    # METRIKK: "Realised price liquids" (USD/boe) fra kvartalstabellen side 3 i hvert rapport.
    #          Inkluderer olje + NGL (kondensate). Enhet: USD/boe ≈ USD/bbl for crude oil.
    # INNSAMLINGSMETODE: PyMuPDF-parsing av offisielle PDF-rapporter, kryssjekket mot
    #   full-årsgjennomsnitt (FY_2023=81.6, FY_2024=80.1, FY_2025=68.9 ✓).
    "2020-Q1": 44.7,   # Q1 2020 rapport (Q2_2020 tabell, Q1-kolonne)
    "2020-Q2": 29.9,   # Q2 2020 rapport direkte
    "2020-Q3": 42.7,   # Q4 2020 rapport (Q3-kolonne)
    "2020-Q4": 44.2,   # Q4 2020 rapport direkte
    "2021-Q1": 60.1,   # Q2 2021 rapport (Q1-kolonne)
    "2021-Q2": 66.9,   # Q2 2021 rapport direkte
    "2021-Q3": 71.5,   # Q4 2021 rapport (Q3-kolonne)
    "2021-Q4": 78.8,   # Q4 2021 rapport direkte
    "2022-Q1": 100.9,  # Q2 2022 rapport (Q1-kolonne) — Ukraina-invasjon premie
    "2022-Q2": 117.5,  # Q2 2022 rapport direkte — topp (Ukraina + sommer)
    "2022-Q3": 101.1,  # Q4 2022 rapport (Q3-kolonne)
    "2022-Q4": 86.6,   # Q4 2022 rapport direkte
    "2023-Q1": 78.4,   # Q2 2023 rapport (Q1-kolonne)
    "2023-Q2": 76.8,   # Q2 2023 rapport direkte
    "2023-Q3": 87.6,   # Q4 2023 rapport (Q3-kolonne)
    "2023-Q4": 83.6,   # Q4 2023 rapport direkte
    "2024-Q1": 82.9,   # Q2 2024 rapport (Q1-kolonne)
    "2024-Q2": 83.1,   # Q2 2024 rapport direkte
    "2024-Q3": 80.3,   # Q4 2024 rapport (Q3-kolonne)
    "2024-Q4": 74.1,   # Q4 2024 rapport direkte
    "2025-Q1": 75.0,   # Q2 2025 rapport (Q1-kolonne) + bekreftet Q1 2026 komparativ
    "2025-Q2": 66.9,   # Q2 2025 rapport direkte
    "2025-Q3": 70.3,   # Q4 2025 rapport (Q3-kolonne)
    "2025-Q4": 63.1,   # Q4 2025 rapport direkte
    "2026-Q1": 82.2,   # Q1 2026 rapport direkte
}
# Alle tall bekreftet direkte fra offisielle AKRBP-rapporter
REPORTED_CONFIRMED = {k: True for k in AKRBP_REPORTED}

# Kvartal med ufullstendig Sodir-produksjonsdata — ekskluder fra validerings-metrikk.
# Årsak: Sodir rapporterer produksjon ~2 mnd forsinket. Brent-vekting bruker kun
# måneder MED Sodir-data, men AKRBP selger og realiserer i ALLE måneder i kvartalet.
# Eksempel: Q1 2026 — Sodir har data jan+feb (Brent $67-71), men AKRBP solgte olje
# i mars (Brent $103) → produksjonsvektet Brent $68 vs. faktisk salgsperiode ~$80.
# Disse kvartalene vises i figuren med egen markering, men inngår IKKE i MAE/RMSE.
SODIR_INCOMPLETE_QUARTERS = {"2026-Q1"}  # oppdater fortløpende

# Produsjonsområde-farger
AREA_COLORS = {
    "JOHAN SVERDRUP": "#2c3e50",   # mørk navy (medium sour, dominant)
    "VALHALL":        "#2980b9",   # blå
    "HOD":            "#5dade2",   # lys blå (Valhall-stream)
    "ALVHEIM":        "#16a085",   # teal (lett søt, høy premium)
    "BØYLA":          "#1abc9c",   # lys teal
    "SKOGUL":         "#76d7c4",   # lysest teal
    "EDVARD GRIEG":   "#6c3483",   # lilla
    "IVAR AASEN":     "#9b59b6",   # lys lilla
    "SKARV":          "#ca6f1e",   # brent oransj (kondensat)
    "ULA":            "#95a5a6",   # grå
    "TAMBAR":         "#aab7b8",
    "TAMBAR ØST":     "#bdc3c7",
    "JETTE":          "#d5d8dc",
}


# ────────────────────────────────────────────────────────────────────────────
# DATAINNLASTING
# ────────────────────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_JSON) as f:
        m = json.load(f)
    coefs = m["coefficients"]
    features = m["features"]
    print(f"  Modell: {m['model_name']}, R²={m['metrics']['r2']:.3f}, OOT R²={m['metrics']['r2_oot']:.3f}")
    return coefs, features


def load_market_time_series():
    """
    Hent de tidsvarierende markedsfeaturene fra panelet (én rad per måned).
    Disse er grade-uavhengige — API/Brent-prisen er lik for alle crudes.
    """
    df = pd.read_csv(PANEL_CSV)

    market_cols = [
        "brent_price", "wti_brent_spread", "diesel_minus_gasoline_crack",
        "brent_dubai_spread", "us_refinery_util_pct",
        "us_crude_stocks_kbbl_dev_5y_pct", "cushing_stocks_kbbl_dev_5y_pct",
        "d_refinery_slack", "fc_slope_4m", "cos_month",
        "d_contango",
        "d_russia_sanctions", "d_iran_sanctions_v1", "d_iran_sanctions_v2",
        "d_venezuela_sanctions", "d_us_shale_boom", "d_covid",
        "d_opec_plus_cuts_2023",
    ]

    mts = (
        df.groupby(["year", "month"])[market_cols]
        .first()
        .reset_index()
    )
    mts["date"] = pd.to_datetime(
        mts["year"].astype(int).astype(str) + "-"
        + mts["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    mts = mts.sort_values("date").reset_index(drop=True)

    # Fyll NaN for d_contango (mangler forward-kurve i siste måneder)
    mts["d_contango"] = mts["d_contango"].fillna(0)
    mts["fc_slope_4m"] = mts["fc_slope_4m"].fillna(0)

    # Kvartal
    mts["quarter"] = mts["date"].dt.quarter
    mts["qstr"] = (
        mts["year"].astype(int).astype(str)
        + "-Q" + mts["quarter"].astype(str)
    )
    return mts


def load_sodir_production():
    sodir = pd.read_csv(SODIR_MONTHLY)
    sodir["date"] = pd.to_datetime(
        {"year": sodir["prfYear"], "month": sodir["prfMonth"], "day": 1}
    )
    # Bare AKRBP-felt med kjent WI
    sodir_akrbp = sodir[sodir["prfInformationCarrier"].isin(AKER_BP_WI)].copy()
    sodir_akrbp["wi"] = sodir_akrbp["prfInformationCarrier"].map(AKER_BP_WI)
    # Netto oljeproduksjon i Mill Sm3
    sodir_akrbp["net_oil_msm3"] = (
        sodir_akrbp["prfPrdOilNetMillSm3"] * sodir_akrbp["wi"]
    )
    return sodir_akrbp


# ────────────────────────────────────────────────────────────────────────────
# MODELL-PREDIKSJON PER FELT PER MÅNED
# ────────────────────────────────────────────────────────────────────────────

# ── FPSO-flag per AKRBP-felt (synkronisert med script 62) ────────────────
# Felt som lastes via FPSO til shuttle-tanker (har is_fpso=1 i Modell v3)
AKRBP_FPSO_FIELDS = {
    "ALVHEIM", "BØYLA", "SKOGUL",   # Alvheim FPSO
    "SKARV",                          # Skarv FPSO
    # Pipeline-felt: VALHALL, HOD, ULA, TAMBAR, TAMBAR ØST (Ekofisk Blend),
    #                EDVARD GRIEG, IVAR AASEN (Grane Blend), JOHAN SVERDRUP
}


# ── Fallback-kilde: imputert kvalitet for felt uten hardkodet assay (Script 63) ──
# main_area → log(avstand til Rotterdam) estimat, avledet fra hardkodede
# FIELD_QUALITY-verdier per region (North Sea 6.26-6.87, Skarv/Norwegian 7.34).
AREA_LOG_DIST = {
    "North sea":     6.50,   # Sture/Mongstad/Teesside ~600-900 km
    "Norwegian sea": 7.30,   # Skarv-nivå ~1500 km
    "Barents sea":   7.70,   # ~2200 km
    "Unknown":       6.60,   # NCS-median
}

_IMPUTED_CACHE = None
def load_imputed_quality() -> dict:
    """Last Script 63 imputeringstabell → {FELT: quality-dict} (samme nøkler som FIELD_QUALITY)."""
    global _IMPUTED_CACHE
    if _IMPUTED_CACHE is not None:
        return _IMPUTED_CACHE
    _IMPUTED_CACHE = {}
    if not IMPUTED_CSV.exists():
        return _IMPUTED_CACHE
    df = pd.read_csv(IMPUTED_CSV)
    for _, r in df.iterrows():
        area = str(r.get("main_area", "Unknown"))
        _IMPUTED_CACHE[str(r["field"]).upper().strip()] = dict(
            api=r["api_gravity"], sulfur=r["sulfur_pct"],
            vacuum_resid=r["vacuum_resid_pct"], ccr=r["ccr_pct"],
            vanadium=r["vanadium_ppm"], nickel=r["nickel_ppm"],
            middle_distillate_pct=r["middle_distillate_pct"],
            log_dist_rotterdam=AREA_LOG_DIST.get(area, AREA_LOG_DIST["Unknown"]),
            confidence=f"imputed ({r['tier']})", tier=r["tier"],
        )
    return _IMPUTED_CACHE


def resolve_field_quality(field: str):
    """
    Hent kvalitetsvektor med fallback-hierarki:
      1. Hardkodet FIELD_QUALITY (offisiell assay/blend) — høyest prioritet, uendret
      2. Script 63 imputeringstabell (standalone/blend/median+DST)
    Returnerer (quality_dict, source) eller (None, None).
    """
    if field in FIELD_QUALITY:
        return FIELD_QUALITY[field], "hardcoded"
    imp = load_imputed_quality()
    if field in imp:
        return imp[field], "imputed"
    return None, None


def build_field_features(field: str, mts: pd.DataFrame, coefs: dict, features: list) -> pd.DataFrame:
    """
    Bygg full feature-matrise for ett felt × alle måneder.
    Returnerer DataFrame med predicted_differential per måned.
    """
    q, _src = resolve_field_quality(field)
    if q is None:
        return None

    api      = q["api"]
    sulfur   = q["sulfur"]
    vac_res  = q["vacuum_resid"]
    ccr      = q["ccr"]
    v_ni     = np.log1p(q["vanadium"] + q["nickel"])
    mid_dist = q.get("middle_distillate_pct", 38.0)
    is_fpso  = 1 if field in AKRBP_FPSO_FIELDS else 0   # NY: Modell v3

    rows = []
    for _, row in mts.iterrows():
        brent  = row["brent_price"]
        s_util = row["us_refinery_util_pct"]
        d_cont = row["d_contango"]

        feat = {
            # ── Statiske kvalitets-features ──────────────────────────────────
            "api_gravity":           api,
            "sulfur_pct":            sulfur,
            "api2":                  api ** 2,
            # Region-dummies — Brent-modell:
            #   Reference = MiddleEast (ingen dummy for MiddleEast).
            #   NCS-felt: reg_NorthSea=1, alle andre=0.
            "reg_NorthAfrica":       0,    # Saharan Blend — NCS=0
            "reg_NorthSea":          1,    # NCS-felt
            "reg_WestAfrica":        0,    # Vest-Afrika — NCS=0
            "vacuum_resid_pct":      vac_res,
            "middle_distillate_pct": mid_dist,
            "ccr_wt_pct":            ccr,
            "log_v_ni":              v_ni,
            # ── Logistikk (NCS = kort avstand, ingen long-distance rabatt) ──
            # d_distance_medium er fjernet fra Brent-modellen (kollineær med WestAfrica)
            "d_distance_long":       0,    # NCS er short-distance → 0
            "is_fpso":               is_fpso,   # NY i v3: -1.98 USD/bbl for FPSO-grades
            # ── Tidsvarierende markedsfeatures ───────────────────────────────
            "brent_price":                       brent,
            # wti_brent_spread er fjernet fra Brent-modellen
            "diesel_minus_gasoline_crack":        row["diesel_minus_gasoline_crack"],
            "brent_dubai_spread":                 row["brent_dubai_spread"],
            "us_refinery_util_pct":               s_util,
            "us_crude_stocks_kbbl_dev_5y_pct":    row["us_crude_stocks_kbbl_dev_5y_pct"],
            "cushing_stocks_kbbl_dev_5y_pct":     row["cushing_stocks_kbbl_dev_5y_pct"],
            "d_refinery_slack":                   row["d_refinery_slack"],
            "fc_slope_4m":                        row["fc_slope_4m"],
            "cos_month":                          row["cos_month"],
            # ── Interaksjoner (static × dynamic) ────────────────────────────
            "sulfur_x_brent":         sulfur * brent,
            "vacuum_resid_x_brent":   vac_res * brent,
            "ccr_x_brent":            ccr * brent,
            "api_x_contango":         api * d_cont,
            "sulfur_x_refinery_util": sulfur * s_util,
            # landlocked-interaksjoner fjernet fra Brent-modellen (null variasjon)
            # ── Politiske dummies (tidsvarierende) ───────────────────────────
            "d_russia_sanctions":     row["d_russia_sanctions"],
            "d_iran_sanctions_v1":    row["d_iran_sanctions_v1"],
            "d_iran_sanctions_v2":    row["d_iran_sanctions_v2"],
            # d_venezuela_sanctions fjernet fra Brent-modellen
            "d_us_shale_boom":        row["d_us_shale_boom"],
            "d_covid":                row["d_covid"],
            "d_opec_plus_cuts_2023":  row["d_opec_plus_cuts_2023"],
        }

        # Prediker: const + Σ coef_j × x_j
        pred = coefs["const"]
        for f in features:
            if f in feat and f in coefs:
                pred += coefs[f] * feat[f]

        rows.append({
            "date":   row["date"],
            "qstr":   row["qstr"],
            "year":   row["year"],
            "month":  row["month"],
            "brent":  brent,
            "diff_pred": pred,
        })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# VEKTET BLENDING PER KVARTAL
# ────────────────────────────────────────────────────────────────────────────

def compute_quarterly_blend(field_diffs: dict, sodir: pd.DataFrame) -> pd.DataFrame:
    """
    Kombiner field-level predikerte differensialer med Sodir-produksjon.
    Returnerer kvartalsvise netto blended differensial og Brent.
    """
    # Samle feltnivå-data
    records = []
    for field, diff_df in field_diffs.items():
        if diff_df is None or diff_df.empty:
            continue
        sodir_f = sodir[sodir["prfInformationCarrier"] == field][
            ["date", "net_oil_msm3"]
        ].copy()
        merged = diff_df.merge(sodir_f, on="date", how="left")
        merged["net_oil_msm3"] = merged["net_oil_msm3"].fillna(0)
        merged["field"] = field
        records.append(merged)

    if not records:
        raise ValueError("Ingen felt-data!")

    df_all = pd.concat(records, ignore_index=True)

    # Aggreger til kvartal
    q_agg = []
    for qstr, grp in df_all.groupby("qstr"):
        total_net = grp["net_oil_msm3"].sum()
        if total_net <= 0:
            continue
        grp = grp.copy()
        grp["share"] = grp["net_oil_msm3"] / total_net

        weighted_diff = (grp["share"] * grp["diff_pred"]).sum()
        # Produksjonsvektet Brent (ekskl. måneder uten produksjonsdata)
        brent_q = (
            (grp["share"] * grp["brent"]).sum()
            if grp["share"].sum() > 0
            else grp["brent"].mean()
        )

        # Field-level breakdown (produksjonsvektet diff per felt)
        def field_stats(g):
            w = g["net_oil_msm3"].sum()
            diff_wavg = (
                (g["net_oil_msm3"] * g["diff_pred"]).sum() / w
                if w > 0 else g["diff_pred"].mean()
            )
            return pd.Series({
                "share":       g["share"].sum(),
                "diff_pred":   diff_wavg,           # produksjonsvektet, ekskl. null-måneder
                "contribution": (g["share"] * g["diff_pred"]).sum(),
            })

        field_breakdown = (
            grp.groupby("field")
            .apply(field_stats)
            .reset_index()
        )

        q_agg.append({
            "qstr":           qstr,
            "year":           int(grp["year"].iloc[0]),
            "quarter":        int(qstr[-1]),
            "brent_q":        brent_q,
            "blended_diff":   weighted_diff,
            "realized_pred":  brent_q + weighted_diff,
            "field_breakdown": field_breakdown,
        })

    q_df = pd.DataFrame([{k: v for k, v in r.items() if k != "field_breakdown"}
                          for r in q_agg])
    q_df = q_df.sort_values("qstr").reset_index(drop=True)

    # Field-level kvartal-table for waterfall
    field_q = []
    for r in q_agg:
        fb = r["field_breakdown"]
        fb["qstr"] = r["qstr"]
        fb["brent_q"] = r["brent_q"]
        field_q.append(fb)
    field_q_df = pd.concat(field_q, ignore_index=True)

    return q_df, field_q_df


# ────────────────────────────────────────────────────────────────────────────
# VALIDERING MOT RAPPORTERTE PRISER
# ────────────────────────────────────────────────────────────────────────────

def compute_validation_stats(q_df: pd.DataFrame) -> dict:
    """Beregn tracking error og bias for modell-prediksjon vs. rapportert."""
    q_df = q_df.copy()
    q_df["reported"] = q_df["qstr"].map(AKRBP_REPORTED)
    q_df["confirmed"] = q_df["qstr"].map(REPORTED_CONFIRMED).fillna(False)
    valid = q_df.dropna(subset=["reported"])

    if len(valid) == 0:
        return {"n": 0}

    # Ekskluder kvartal med ufullstendig Sodir-data fra metrikk-beregning
    # (Sodir-lag gjør at produksjonsvektet Brent er for lav, se SODIR_INCOMPLETE_QUARTERS)
    valid_complete = valid[~valid["qstr"].isin(SODIR_INCOMPLETE_QUARTERS)]

    errors = valid_complete["realized_pred"] - valid_complete["reported"]
    return {
        "n":          len(valid_complete),
        "n_total":    len(valid),
        "mae":        errors.abs().mean(),
        "rmse":       np.sqrt((errors ** 2).mean()),
        "bias":       errors.mean(),
        "corr":       valid_complete[["realized_pred", "reported"]].corr().iloc[0, 1],
        "r2":         1 - (errors ** 2).sum() / ((valid_complete["reported"] - valid_complete["reported"].mean()) ** 2).sum(),
        "n_excluded": len(SODIR_INCOMPLETE_QUARTERS & set(valid["qstr"])),
    }


# ────────────────────────────────────────────────────────────────────────────
# VISUALISERING
# ────────────────────────────────────────────────────────────────────────────

def plot_decomposition(q_df: pd.DataFrame, field_q_df: pd.DataFrame, stats: dict):
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.52, wspace=0.32,
        height_ratios=[1.0, 0.85, 0.9],
    )

    ax1 = fig.add_subplot(gs[0, :])   # Full-width: realized price comparison
    ax2 = fig.add_subplot(gs[1, :])   # Full-width: field contribution waterfall
    ax3 = fig.add_subplot(gs[2, 0])   # Production shares bar
    ax4 = fig.add_subplot(gs[2, 1])   # Quality positioning scatter

    q_df = q_df.copy()
    q_df["reported"] = q_df["qstr"].map(AKRBP_REPORTED)

    # Filtrer ut kvartal uten Sodir-produksjonsdata (blended_diff = 0 pga. ingen vekter)
    # AKRBP-feltene hadde produksjon i 2000-2005 men ikke under nåværende eierstruktur.
    # Første meningsfulle differensial er når JS/Alvheim/Valhall er i porteføljen (~2006).
    q_df = q_df[q_df["blended_diff"] != 0].reset_index(drop=True)

    x = range(len(q_df))
    labels = q_df["qstr"].tolist()

    # ── Panel 1: Realized price comparison ──────────────────────────────────
    ax1.fill_between(x, q_df["brent_q"], alpha=0.15, color="#7f8c8d", label="Brent (flat)")
    ax1.plot(x, q_df["brent_q"], color="#7f8c8d", lw=1.5, ls="--", alpha=0.7)
    ax1.plot(x, q_df["realized_pred"], color="#e74c3c", lw=2.2,
             label="Modell-predikert realisert", zorder=5)

    rmse = 2.95   # Modell B (Brent-linked) OOT RMSE
    ax1.fill_between(x,
                     q_df["realized_pred"] - rmse,
                     q_df["realized_pred"] + rmse,
                     alpha=0.12, color="#e74c3c", label=f"±{rmse:.1f} USD/bbl (OOT RMSE)")

    rep_x = [i for i, qstr in enumerate(labels) if qstr in AKRBP_REPORTED]
    rep_y = [AKRBP_REPORTED[labels[i]] for i in rep_x]
    confirmed = [REPORTED_CONFIRMED.get(labels[i], False) for i in rep_x]
    incomplete = [labels[i] in SODIR_INCOMPLETE_QUARTERS for i in rep_x]

    # Bekreftede, komplette kvartal (grønn)
    ax1.scatter(
        [rep_x[i] for i, (c, inc) in enumerate(zip(confirmed, incomplete)) if c and not inc],
        [rep_y[i] for i, (c, inc) in enumerate(zip(confirmed, incomplete)) if c and not inc],
        color="#27ae60", zorder=6, s=55, label="Rapportert (bekreftet, AKRBP IR)", marker="o"
    )
    # Kvartal med ufullstendig Sodir-data (rød stjerne — vises men inngår ikke i metrikk)
    ax1.scatter(
        [rep_x[i] for i, inc in enumerate(incomplete) if inc],
        [rep_y[i] for i, inc in enumerate(incomplete) if inc],
        color="#c0392b", zorder=7, s=90, label="Rapportert (Sodir-lag, ekskl. metrikk)", marker="*"
    )
    # Estimater (gul — ingen lenger siden alle er bekreftet)
    ax1.scatter(
        [rep_x[i] for i, c in enumerate(confirmed) if not c],
        [rep_y[i] for i, c in enumerate(confirmed) if not c],
        color="#f39c12", zorder=6, s=55, label="Rapportert (estimat)", marker="s"
    )

    # Event annotations — adaptiv posisjonering (over eller under)
    ymax = q_df["realized_pred"].max()
    event_map = {
        "2020-Q2": ("COVID\nbunn",   "above"),
        "2022-Q2": ("Ukraine\npremium", "above"),
        "2023-Q1": ("OPEC+\nkutt",   "below"),
        "2025-Q2": ("Iran\npremium", "below"),   # satt ned for å unngå topp
    }
    for qstr, (ev_label, direction) in event_map.items():
        if qstr in labels:
            xi = labels.index(qstr)
            yi = q_df.loc[q_df["qstr"] == qstr, "realized_pred"].values
            if len(yi):
                offset = 10 if direction == "above" else -14
                va = "bottom" if direction == "above" else "top"
                ax1.annotate(
                    ev_label, (xi, yi[0]),
                    xytext=(xi, yi[0] + offset),
                    ha="center", fontsize=7.5, color="#555", va=va,
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8),
                )

    tick_idx = [i for i, l in enumerate(labels) if l.endswith("Q1") or l.endswith("Q3")]
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("USD/bbl", fontsize=10)
    ax1.set_title("Aker BP — Modell-predikert vs. rapportert realisert oljepris",
                  fontsize=12, fontweight="bold", pad=8)

    # Legend — nedre venstre (data er høyest til høyre pga. Iran-spike)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9,
               ncol=2, columnspacing=1.0)

    # Stats box — nedre høyre, liten font
    if stats["n"] > 0:
        stat_txt = (
            f"Validering n={stats['n']} kv. | "
            f"MAE {stats['mae']:.2f} | "
            f"RMSE {stats['rmse']:.2f} | "
            f"Bias {stats['bias']:+.2f} | "
            f"R² {stats['r2']:.3f}"
        )
        ax1.text(0.99, 0.04, stat_txt, transform=ax1.transAxes,
                 fontsize=8, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                           edgecolor="#ccc", alpha=0.9))

    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Feltbidrag til differensial ────────────────────────────────
    fields_ordered = [
        "JOHAN SVERDRUP", "ALVHEIM", "BØYLA", "SKOGUL",
        "EDVARD GRIEG", "IVAR AASEN",
        "VALHALL", "HOD",
        "SKARV", "ULA", "TAMBAR", "TAMBAR ØST",
    ]
    fields_in_data = [f for f in fields_ordered if f in field_q_df["field"].unique()]

    pivot = field_q_df.pivot_table(
        index="qstr", columns="field", values="contribution", aggfunc="sum"
    ).reindex(q_df["qstr"])
    pivot = pivot[[f for f in fields_in_data if f in pivot.columns]]

    x2 = range(len(pivot))
    bottom_pos = np.zeros(len(pivot))
    bottom_neg = np.zeros(len(pivot))

    for field in pivot.columns:
        vals = pivot[field].fillna(0).values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        color = AREA_COLORS.get(field, "#aaa")
        ax2.bar(x2, pos_vals, bottom=bottom_pos, color=color, label=field, width=0.85)
        ax2.bar(x2, neg_vals, bottom=bottom_neg, color=color, width=0.85)
        bottom_pos += pos_vals
        bottom_neg += neg_vals

    ax2.axhline(0, color="black", lw=0.8)
    ax2.plot(x2, q_df["blended_diff"].values, color="black", lw=2.0, ls="-",
             label="Blended diff", zorder=10)

    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("USD/bbl vs. Brent", fontsize=10)
    ax2.set_title("Feltbidrag til differensial mot Brent (vektet av produksjonsandel)",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.grid(axis="y", alpha=0.3)

    # Legend utenfor plottet (under x-aksen)
    handles, leg_labels = ax2.get_legend_handles_labels()
    ax2.legend(
        handles, leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=6, fontsize=8,
        framealpha=0.9, edgecolor="#ccc",
        handlelength=1.2, handletextpad=0.5, columnspacing=1.0,
    )

    # ── Panel 3: Siste kvartal produksjonsandeler ────────────────────────────
    last_q = field_q_df[field_q_df["qstr"] == q_df["qstr"].iloc[-1]]
    last_q = last_q[last_q["share"] > 0.005].sort_values("share", ascending=True)
    colors3 = [AREA_COLORS.get(f, "#aaa") for f in last_q["field"]]
    bars = ax3.barh(last_q["field"], last_q["share"] * 100, color=colors3, height=0.65)
    for bar, (_, row) in zip(bars, last_q.iterrows()):
        ax3.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f'{row["share"]*100:.1f}%',
            va="center", fontsize=8.5
        )
    ax3.set_xlabel("Produksjonsandel %", fontsize=10)
    ax3.set_title(f"Produksjonsandeler — {q_df['qstr'].iloc[-1]}",
                  fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlim(0, last_q["share"].max() * 130)
    ax3.tick_params(axis="y", labelsize=8.5)
    ax3.grid(axis="x", alpha=0.3)

    # ── Panel 4: Kvalitetsposisjonering ──────────────────────────────────────
    # Manuelle offsets for å unngå overlap i det tette API 28–36-området
    LABEL_OFFSETS = {
        "JOHAN\nSVERDRUP":  (-52, -14),
        "VALHALL":           (  6,   5),
        "HOD":               (  6, -14),
        "ALVHEIM":           (-46,   6),
        "BØYLA":             (  6,   5),
        "SKOGUL":            (  6, -14),
        "EDVARD\nGRIEG":     (-52,   6),
        "IVAR\nAASSEN":      (-50, -14),
        "SKARV":             (  6,   5),
        "ULA":               (  6,  -5),
        "TAMBAR":            (  6,   5),
        "TAMBAR\nØST":       (-50, -14),
        "JETTE":             (-40,   6),
    }

    q4_data = []
    for field in FIELD_QUALITY:
        if field not in field_q_df["field"].values:
            continue
        fq = field_q_df[field_q_df["field"] == field]
        avg_diff  = fq["diff_pred"].mean()
        avg_share = fq["share"].mean()
        q4_data.append({
            "field":     field,
            "api":       FIELD_QUALITY[field]["api"],
            "sulfur":    FIELD_QUALITY[field]["sulfur"],
            "avg_diff":  avg_diff,
            "avg_share": avg_share,
        })
    q4_df = pd.DataFrame(q4_data)

    ax4.scatter(
        q4_df["api"], q4_df["avg_diff"],
        s=q4_df["avg_share"] * 3000,
        c=[AREA_COLORS.get(f, "#aaa") for f in q4_df["field"]],
        alpha=0.75, edgecolors="white", linewidths=1.2, zorder=5
    )

    for _, row in q4_df.iterrows():
        # Lag label-tekst (linjeskift i feltnavnet for plass)
        lbl = row["field"].replace(" ", "\n")
        dx, dy = LABEL_OFFSETS.get(lbl, (6, 5))
        ax4.annotate(
            lbl,
            (row["api"], row["avg_diff"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            va="center",
            ha="left" if dx >= 0 else "right",
            linespacing=1.1,
        )

    ax4.axhline(0, color="black", lw=0.8, ls="--")
    ax4.set_xlabel("API Gravity", fontsize=10)
    ax4.set_ylabel("Gj.snitt predikert differensial (USD/bbl)", fontsize=10)
    ax4.set_title("Kvalitetsposisjonering per felt\n(boblestørrelse = produksjonsandel)",
                  fontsize=11, fontweight="bold", pad=8)
    # Gi nok plass til SKARV (API 50.8) og labels til venstre
    ax4.set_xlim(24, 57)
    ax4.grid(alpha=0.3)

    fig.suptitle(
        "Aker BP — Feltspesifikk realisert pris-dekomponering\n"
        "Kilde: AKRBP Sodir-data × Brent-linked regresjonsmodell (OOT R²=0.337, RMSE=2.95 USD/bbl)",
        fontsize=13, fontweight="bold", y=0.998,
    )

    out_path = FIG_DIR / "42_akrbp_realized_decomposition.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\n  Figur lagret: {out_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# EKSPORT
# ────────────────────────────────────────────────────────────────────────────

def export_tables(q_df: pd.DataFrame, field_q_df: pd.DataFrame):
    """Lagre kvartalstabell og felt-breakdown som CSV."""
    # Legg til rapportert for sammenligning
    q_df = q_df.copy()
    q_df["reported_price"]   = q_df["qstr"].map(AKRBP_REPORTED)
    q_df["model_error"]      = q_df["realized_pred"] - q_df["reported_price"]
    q_df["model_vs_brent"]   = q_df["realized_pred"] - q_df["brent_q"]
    q_df["reported_vs_brent"] = q_df["reported_price"] - q_df["brent_q"]

    out_q = OUT_DIR / "42_akrbp_quarterly_realized.csv"
    q_df.to_csv(out_q, index=False)
    print(f"  Kvartalstabell: {out_q}")

    out_f = OUT_DIR / "42_akrbp_field_breakdown.csv"
    field_q_df.to_csv(out_f, index=False)
    print(f"  Felt-breakdown: {out_f}")

    return q_df


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SCRIPT 42: Aker BP Realized-Price Decomposition")
    print("=" * 65)

    # 1. Last inn
    print("\n[1] Laster inn modell og data...")
    coefs, features = load_model()

    mts = load_market_time_series()
    print(f"  Markedsdata: {len(mts)} måneder ({mts['date'].min().strftime('%Y-%m')} → {mts['date'].max().strftime('%Y-%m')})")

    sodir = load_sodir_production()
    fields_in_sodir = sodir["prfInformationCarrier"].unique()
    print(f"  Sodir-felt:  {', '.join(sorted(fields_in_sodir))}")

    # 2. Prediker differensial per felt per måned
    print("\n[2] Predikerer felt-differensialer...")
    field_diffs = {}
    n_hardcoded = n_imputed = 0
    for field in AKER_BP_WI:
        q, src = resolve_field_quality(field)
        if q is None:
            continue
        diff_df = build_field_features(field, mts, coefs, features)
        field_diffs[field] = diff_df
        q_mean = diff_df["diff_pred"].mean() if diff_df is not None else np.nan
        conf = q.get("confidence", "?")
        api  = q["api"]
        sulf = q["sulfur"]
        flag = "  " if src == "hardcoded" else " ⟲"   # ⟲ = fallback fra Script 63
        n_hardcoded += src == "hardcoded"
        n_imputed += src == "imputed"
        print(f" {flag}{field:21s} | API {api:4.1f} | S {sulf:.3f}% | "
              f"Avg diff {q_mean:+5.2f} USD/bbl | {conf}")
    print(f"\n  → {n_hardcoded} felt hardkodet (offisiell assay), "
          f"{n_imputed} via Script 63 fallback (⟲)")

    # 3. Vektet blending per kvartal
    print("\n[3] Beregner kvartalsvise blendede priser...")
    q_df, field_q_df = compute_quarterly_blend(field_diffs, sodir)
    print(f"  Kvartal: {q_df['qstr'].min()} → {q_df['qstr'].max()} ({len(q_df)} kvartaler)")

    # 4. Validering
    print("\n[4] Validering mot rapporterte priser (faktiske AKRBP-rapporttall)...")
    stats = compute_validation_stats(q_df)
    if stats["n"] > 0:
        excl = stats.get("n_excluded", 0)
        print(f"  n (komplett)     = {stats['n']} kvartaler (av {stats['n_total']} totalt, {excl} ekskl. Sodir-lag)")
        print(f"  MAE              = {stats['mae']:.2f} USD/boe")
        print(f"  RMSE             = {stats['rmse']:.2f} USD/boe")
        print(f"  Bias             = {stats['bias']:+.2f} USD/boe  (neg = modell under-predikerer)")
        print(f"  Korrelasjon      = {stats['corr']:.3f}")
        print(f"  R² (vs reported) = {stats['r2']:.3f}")
        if abs(stats["bias"]) < 3:
            print("  ✓ Lav bias — modellen fanger AKRBP realized pris strukturelt")
            print("  OBS: Rapportert metrikk er 'realised liquids price' (olje + NGL/kondensat)")
        else:
            print(f"  △ Bias={stats['bias']:+.2f} — sjekk NGL-komponenten eller markedsdata")
        if excl > 0:
            print(f"  OBS: {list(SODIR_INCOMPLETE_QUARTERS)} ekskludert pga. ufullstendig Sodir-data")
            print(f"       (Sodir-lag: Brent-spike i mars/apr 2026 ikke fanget i produksjonsvektet avg)")
    else:
        print("  (Ingen bekreftet rapporterte priser)")

    # 5. Kvartalsoversikt
    print("\n[5] Kvartalsoversikt (siste 12 kv):")
    q_df["reported"] = q_df["qstr"].map(AKRBP_REPORTED)
    view = q_df.tail(12)[["qstr", "brent_q", "blended_diff", "realized_pred", "reported"]].copy()
    view.columns = ["Kvartal", "Brent", "Blended Diff", "Pred Realized", "Rapportert"]
    print(view.to_string(index=False, float_format="{:.2f}".format))

    # 6. Figur + export
    print("\n[6] Genererer figurer og eksporterer...")
    plot_decomposition(q_df, field_q_df, stats)
    q_df_exp = export_tables(q_df, field_q_df)

    # 7. Summary
    print("\n" + "=" * 65)
    print("  SAMMENDRAG")
    print("=" * 65)
    recent = q_df.tail(4)
    avg_diff = recent["blended_diff"].mean()
    avg_pred = recent["realized_pred"].mean()
    avg_brent = recent["brent_q"].mean()
    print(f"  Siste 4 kv. gjennomsnitt:")
    print(f"    Brent:             {avg_brent:.2f} USD/bbl")
    print(f"    Blended diff:      {avg_diff:+.2f} USD/bbl")
    print(f"    Predikert realized: {avg_pred:.2f} USD/bbl")
    print()

    # Felt-bidrag siste kvartal
    last_qstr = q_df["qstr"].iloc[-1]
    last_fb = field_q_df[field_q_df["qstr"] == last_qstr].sort_values("contribution")
    print(f"  Feltbidrag siste kvartal ({last_qstr}):")
    for _, row in last_fb.iterrows():
        if row["share"] > 0.01:
            print(f"    {row['field']:22s}: {row['share']*100:5.1f}% av prod, "
                  f"diff {row['diff_pred']:+.2f}, bidrag {row['contribution']:+.2f} USD/bbl")

    print()
    print("  OBS: Oppdater AKRBP_REPORTED med faktiske tall fra")
    print("       akerbp.com/investor-relations → Quarterly Reports")
    print()


if __name__ == "__main__":
    main()
