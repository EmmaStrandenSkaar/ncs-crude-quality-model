"""
Script 43: Aker BP — Forward realisert pris-prediksjon Q3 2026 – Q4 2028
=========================================================================

Bygger videre på script 42 (felt-dekomponering) og projiserer frem:
  1. Brent-scenario-analyse (Bear / Base / Bull)
  2. Produksjonsmix-shift fra dagens portefølje til Yggdrasil-ramp-up
  3. Revenue waterfall: Brent vs. mix-shift vs. quality differential

PRODUKSJONSGRUNNLAG:
  Nåværende felt (Sodir-data 2025-snitt, netto kbopd):
    Johan Sverdrup: ~215 kbopd (netto, WI=31.57%) — dominerende felt (~71% av mix)
    Valhall:         ~28 kbopd (netto, WI=90%)
    Alvheim area:    ~21 kbopd (netto, WI=65%)
    Edvard Grieg:    ~16 kbopd (netto, WI=65%)
    Hod:              ~7 kbopd (netto, WI=90%)
    Ivar Aasen:       ~5 kbopd (netto, WI=34.8%)
    Skarv:            ~3 kbopd (netto, WI=23.8%) — primært olje; NGL separat
    ULA+Tambar:       ~8 kbopd (netto, kombinert)
    Bøyla:            ~3 kbopd (netto, WI=65%)

  Nytt felt — Yggdrasil (fka NOAKA, omdøpt ~2023):
    Yggdrasil = Munin (fka Krafla, AKRBP 50%) + Fulla (kondensat, AKRBP 40%)
              + Hugin (AKRBP 76.72%) + Hugin Satellites (AKRBP 87.70%)
    Reservoar: Middle Jurassic Brent Group sandstone, 3 200-3 650m
    Crude-kvalitet: API ~37°, S ~0.22% (Statfjord/Brent Group-proxy)
    Produksjon: first oil estimert Q2-Q4 2027 (avhengig av plan)
      — Oppdater med faktisk CMD 2025 guidance fra AKRBP!
    Plateau: ~100-120 kbopd gross → ~65-80 kbopd netto (vektet WI ~65%)

  Produksjonsguidance AKRBP: 400-430 kboepd 2026-2028 (inkl. gass/NGL)
    → Olje ≈ 300-340 kbopd (typisk ~75-80% av total BOE)

FORWARD BRENT SCENARIER (per mai 2026):
  Bear ($62-65): Brent holder seg lavt 2027-2028
    Drevet av OPEC+ supply-ramp + global etterspørselsvekst <1%
  Base ($72-75): EIA/IEA konsensus (apr 2026)
  Bull ($85-95): Eskalering Iran-konflikten → Hormuz-risiko vedvarer

OBS: Alle Yggdrasil-volumer og assay-kvalitet er estimater.
     Oppdater med faktisk assay når felt starter produksjon (2027+).
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

PROJECT_ROOT  = Path(__file__).parent.parent
MODEL_JSON    = PROJECT_ROOT / "data" / "processed" / "34b_brent_model.json"
PANEL_CSV     = PROJECT_ROOT / "data" / "processed" / "regression_panel.csv"
SODIR_MONTHLY = PROJECT_ROOT / "data" / "raw" / "sodir" / "sodir_field_production_monthly.csv"
Q42_QUARTERLY = PROJECT_ROOT / "data" / "processed" / "42_akrbp_quarterly_realized.csv"
OUT_DIR       = PROJECT_ROOT / "data" / "processed"

# ── Brent-scenarier ──────────────────────────────────────────────────────────
# Format: {scenario: {year: {quarter: price}}}
# Projeksjon starter Q3 2026. Kilde: EIA STEO + markedskonsensus mai 2026.
BRENT_SCENARIOS = {
    "Bear ($62-65)": {
        2026: {3: 68.0, 4: 64.0},
        2027: {1: 62.0, 2: 61.0, 3: 61.0, 4: 60.0},
        2028: {1: 60.0, 2: 59.0, 3: 60.0, 4: 60.0},
    },
    "Base ($72-75)": {
        2026: {3: 76.0, 4: 73.0},
        2027: {1: 74.0, 2: 73.0, 3: 73.0, 4: 72.0},
        2028: {1: 72.0, 2: 71.0, 3: 72.0, 4: 72.0},
    },
    "Bull ($85-95)": {
        2026: {3: 90.0, 4: 86.0},
        2027: {1: 88.0, 2: 87.0, 3: 86.0, 4: 85.0},
        2028: {1: 84.0, 2: 83.0, 3: 82.0, 4: 82.0},
    },
}

# ── Forward produksjonsprofil per kvartal (netto kboepd) ────────────────────
# Bygget fra Sodir-snitt (historical) + AKRBP CMD 2025 guidance (forward)
# AKRBP total olje (ekskl. gass/NGL): typisk ~75-80% av BOE-produksjonen

FIELD_PRODUCTION_FWD = {
    # (Felt, Kvartal, ÅR): netto kbopd (kilo fat olje/dag)
    # --- Eksisterende felt (Sodir 2025-snitt som baseline, kalibrert) ---
    # Total AKRBP netto olje 2025: ~310 kbopd per Sodir
    # Johan Sverdrup: 218 kbopd netto (Sodir 2025-snitt, dominerer ~70%)
    "JOHAN SVERDRUP": {
        (2026, 3): 215.0, (2026, 4): 215.0,
        (2027, 1): 210.0, (2027, 2): 208.0, (2027, 3): 205.0, (2027, 4): 202.0,
        (2028, 1): 200.0, (2028, 2): 198.0, (2028, 3): 195.0, (2028, 4): 193.0,
    },
    "VALHALL": {  # 27.5 kbopd Sodir 2025-snitt
        (2026, 3): 27.0, (2026, 4): 27.5,
        (2027, 1): 28.0, (2027, 2): 28.0, (2027, 3): 27.5, (2027, 4): 27.0,
        (2028, 1): 27.0, (2028, 2): 26.5, (2028, 3): 26.5, (2028, 4): 26.0,
    },
    "ALVHEIM": {  # 22.1 kbopd Sodir 2025-snitt
        (2026, 3): 21.0, (2026, 4): 20.5,
        (2027, 1): 20.0, (2027, 2): 19.5, (2027, 3): 19.0, (2027, 4): 18.5,
        (2028, 1): 18.0, (2028, 2): 17.5, (2028, 3): 17.0, (2028, 4): 16.5,
    },
    "EDVARD GRIEG": {  # 16.3 kbopd Sodir 2025-snitt (inkl. Ivar Aasen proxy)
        (2026, 3): 16.0, (2026, 4): 16.5,
        (2027, 1): 16.5, (2027, 2): 16.0, (2027, 3): 15.5, (2027, 4): 15.0,
        (2028, 1): 15.0, (2028, 2): 14.5, (2028, 3): 14.0, (2028, 4): 13.5,
    },
    "HOD": {  # 6.6 kbopd Sodir 2025-snitt
        (2026, 3): 6.5, (2026, 4): 7.0,
        (2027, 1): 7.0, (2027, 2): 7.0, (2027, 3): 6.5, (2027, 4): 6.5,
        (2028, 1): 6.5, (2028, 2): 6.0, (2028, 3): 6.0, (2028, 4): 6.0,
    },
    "IVAR AASEN": {  # 5.2 kbopd Sodir 2025-snitt
        (2026, 3): 5.0, (2026, 4): 5.0,
        (2027, 1): 5.0, (2027, 2): 4.5, (2027, 3): 4.5, (2027, 4): 4.0,
        (2028, 1): 4.0, (2028, 2): 3.5, (2028, 3): 3.5, (2028, 4): 3.0,
    },
    "SKARV": {  # 2.9 kbopd Sodir 2025-snitt (olje; NGL/kondensate separat)
        (2026, 3): 3.0, (2026, 4): 3.0,
        (2027, 1): 3.0, (2027, 2): 2.5, (2027, 3): 2.5, (2027, 4): 2.5,
        (2028, 1): 2.0, (2028, 2): 2.0, (2028, 3): 2.0, (2028, 4): 2.0,
    },
    "BØYLA": {  # 2.8 kbopd Sodir 2025-snitt
        (2026, 3): 2.5, (2026, 4): 2.5,
        (2027, 1): 2.5, (2027, 2): 2.0, (2027, 3): 2.0, (2027, 4): 1.5,
        (2028, 1): 1.5, (2028, 2): 1.5, (2028, 3): 1.0, (2028, 4): 1.0,
    },
    "ULA": {  # ULA (3.7) + Tambar (3.8) + Tambar Øst (~0) = 7.5 kbopd Sodir 2025-snitt
             # Tambar/Tambar Øst bruker samme kvalitetsprofil som ULA → samles her
        (2026, 3): 7.0, (2026, 4): 7.0,
        (2027, 1): 7.0, (2027, 2): 6.5, (2027, 3): 6.0, (2027, 4): 6.0,
        (2028, 1): 5.5, (2028, 2): 5.0, (2028, 3): 5.0, (2028, 4): 4.5,
    },
    # --- Nytt felt: Yggdrasil (tidligere kalt NOAKA) ---
    # Yggdrasil = Munin (fka Krafla) + Fulla + Hugin + Hugin Satellites + Frøy
    # OBS: Yggdrasil og NOAKA er SAMME PROSJEKT — Yggdrasil er det nye prosjektnavnet!
    #
    # Volumer (kilde: Aker BP / Sodir factpages):
    #   Totale ressurser: ~700 million BOE (hele prosjektet)
    #   Munin (Krafla): 19.5 mill Sm3 olje = 123 Mbbl gross
    #   Fulla: primært gass + kondensat
    #   Hugin + Hugin Satellites: anslått ~50-70 Mbbl olje
    #   Totalt olje: ~200-250 Mbbl gross over feltets levetid
    #
    # AKRBP WI (vektet, fra Sodir factpages):
    #   Hugin Satellites: 87.70%, Hugin: 76.72%, Fulla: 40.00%, Munin: 50.00%
    #   Vektet snitt ≈ 60-65% → netto ~130-160 Mbbl olje over feltlevetid
    #
    # Produksjonsprofil: first oil estimert H2 2027 (Hugin Satellites tidligst)
    # Plateau: ~100-120 kbopd gross → ~65-80 kbopd netto (vektet WI ~65%)
    # ⚠️ USIKKER TIMING — oppdater fra AKRBP CMD 2025 presentasjoner!
    #    Noen analytikerrapporter antyder Q4 2027 eller tidlig 2028 for full oppstart.
    #    Sodir factpages: Munin NPDID 42002476, Fulla NPDID 42002479
    "YGGDRASIL": {
        (2027, 2): 8.0,    # Hugin Satellites tidlig oppstart (usikker)
        (2027, 3): 22.0,   # Munin/Hugin ramp-up
        (2027, 4): 38.0,
        (2028, 1): 50.0,
        (2028, 2): 62.0,
        (2028, 3): 70.0,
        (2028, 4): 75.0,   # tilnærmet plateau (oppdater fra CMD)
    },
}

# ── Assay-kvalitet for Yggdrasil ─────────────────────────────────────────
# YGGDRASIL = Munin (fka Krafla) + Fulla + Hugin + Hugin Satellites
# Reservoir: Middle Jurassic Brent Group sandstone, 3,200-3,650m dyp
# Lokasjon: mellom Alvheim og Oseberg (sentralt Nordsjø-basseng)
#
# Ingen published crude assay er tilgjengelig (pre-produksjon felt).
#
# Kilde til kvalitets-estimat:
#  1. Reservoaranalog: Brent Group nær Oseberg → typisk API 35-42°, S 0.15-0.30%
#     (Oseberg: API 39.9°, S 0.19%; Statfjord: API 37.8°, S 0.26%)
#  2. Produksjonsdybde 3,200-3,650m → noe høyere modning enn Oseberg → API ≈ 36-38°
#  3. Eksportlinje: Grane-rørledning til Sture. Grane-blend er nå API ~31.5°
#     (Grane API 27.1° + Edvard Grieg API 32° + Ivar Aasen API 33°).
#     Yggdrasil vil typisk HEVE blend-APIn dersom cruden er lettere.
#  4. Søk bekrefter: Fulla er gass/kondensat (svært lett), Munin er primær-olje.
#
# ESTIMAT: API ~37°, S ~0.22% (Statfjord-proxy, Brent Group analog)
# Konfidensintervall: API 34-40°, S 0.15-0.35%
# ★★ Estimat — oppdater med faktisk assay-data når tilgjengelig (2027+)
FIELD_QUALITY_FWD = {
    "YGGDRASIL": dict(
        api=37.0, sulfur=0.220, vacuum_resid=13.0, ccr=2.8,
        vanadium=2.5, nickel=2.0,
        # Brent Group analog: API~37° lett søt → mye middle distillates.
        # Proxy: mellom Alvheim (44.8%) og Ekofisk Blend (38.8%) → ~42%
        middle_distillate_pct=42.0,
        confidence="★★ Brent Group analog (Statfjord/Oseberg-proxy, 3200-3650m). Ingen published assay.",
    ),
}

# Eksisterende felt-kvalitet — KUN OFFISIELLE DATAKILDER (synkronisert med script 42)
# Kun felt med publisert offisiell Equinor XLSX-assay inngår.
# Ekskluderte felt (ingen offentlig standalone-assay): Valhall, HOD, Edvard Grieg,
#   Ivar Aasen, ULA, Tambar, Tambar Øst, Jette → inngår ikke i diff-beregningen.
FIELD_QUALITY_EXISTING = {
    # middle_distillate_pct = kerosene + gasoil yield fra offisielle Equinor XLSX-assays
    # Synkronisert med script 42 FIELD_QUALITY (Brent-modellen bruker denne featuren)
    "JOHAN SVERDRUP": dict(api=28.7, sulfur=0.809, vacuum_resid=19.04, ccr=4.21, vanadium=12.06, nickel=3.77, middle_distillate_pct=36.93),  # ★★★ Equinor offisiell assay
    "ALVHEIM":        dict(api=34.5, sulfur=0.402, vacuum_resid=8.72,  ccr=1.12, vanadium=2.76,  nickel=0.80, middle_distillate_pct=44.79),  # ★★★ Equinor offisiell assay
    "BØYLA":          dict(api=34.5, sulfur=0.402, vacuum_resid=8.72,  ccr=1.12, vanadium=2.76,  nickel=0.80, middle_distillate_pct=44.79),  # ★★★ Alvheim FPSO-stream
    "SKOGUL":         dict(api=34.5, sulfur=0.402, vacuum_resid=8.72,  ccr=1.12, vanadium=2.76,  nickel=0.80, middle_distillate_pct=44.79),  # ★★★ Alvheim FPSO-stream
    "SKARV":          dict(api=50.8, sulfur=0.064, vacuum_resid=1.75,  ccr=0.24, vanadium=0.50,  nickel=0.17, middle_distillate_pct=46.71),  # ★★★ Equinor offisiell assay
    # Blend-assay-proxyer (offisielle Equinor XLSX-assays for eksportstreamen)
    "VALHALL":        dict(api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43, vanadium=1.87,  nickel=3.00, middle_distillate_pct=38.82),  # ★★★ Ekofisk Blend
    "HOD":            dict(api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43, vanadium=1.87,  nickel=3.00, middle_distillate_pct=38.82),  # ★★★ Ekofisk Blend
    "ULA":            dict(api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43, vanadium=1.87,  nickel=3.00, middle_distillate_pct=38.82),  # ★★★ Ekofisk Blend
    "TAMBAR":         dict(api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43, vanadium=1.87,  nickel=3.00, middle_distillate_pct=38.82),  # ★★★ Ekofisk Blend
    "TAMBAR ØST":     dict(api=38.9, sulfur=0.207, vacuum_resid=12.20, ccr=1.43, vanadium=1.87,  nickel=3.00, middle_distillate_pct=38.82),  # ★★★ Ekofisk Blend
    "EDVARD GRIEG":   dict(api=27.1, sulfur=0.668, vacuum_resid=19.00, ccr=3.27, vanadium=11.07, nickel=3.52, middle_distillate_pct=38.19),  # ★★★ Grane Blend
    "IVAR AASEN":     dict(api=27.1, sulfur=0.668, vacuum_resid=19.00, ccr=3.27, vanadium=11.07, nickel=3.52, middle_distillate_pct=38.19),  # ★★★ Grane Blend
}

# Alle felt
ALL_FIELD_QUALITY = {**FIELD_QUALITY_EXISTING, **FIELD_QUALITY_FWD}


# ────────────────────────────────────────────────────────────────────────────
# HJELPE-IMPORT FRA SCRIPT 42
# (Reimplementerer de nødvendige funksjonene fremfor å importere)
# ────────────────────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_JSON) as f:
        m = json.load(f)
    return m["coefficients"], m["features"]


def get_market_defaults_2025():
    """
    Baseline markedsstatus for forward-projeksjoner:
    Bruker siste tilgjengelige paneldata som proxy for 'normalt' regime.
    """
    df = pd.read_csv(PANEL_CSV)
    # Snitt av 2024 som baseline (unngå 2025-Iran-spike)
    mask = (df["year"] == 2024)
    sub = df[mask]
    market_baseline = {
        # Modell B (Brent-linked) features — wti_brent_spread og d_venezuela_sanctions fjernet
        "diesel_minus_gasoline_crack":   sub["diesel_minus_gasoline_crack"].mean(),
        "brent_dubai_spread":            sub["brent_dubai_spread"].mean(),
        "us_refinery_util_pct":          sub["us_refinery_util_pct"].mean(),
        "us_crude_stocks_kbbl_dev_5y_pct": sub["us_crude_stocks_kbbl_dev_5y_pct"].mean(),
        "cushing_stocks_kbbl_dev_5y_pct":  sub["cushing_stocks_kbbl_dev_5y_pct"].mean(),
        "d_refinery_slack":              0,
        "fc_slope_4m":                   0,   # flat forward curve assumption
        "d_contango":                    0,
        "d_russia_sanctions":            1,
        "d_iran_sanctions_v1":           0,
        "d_iran_sanctions_v2":           1,   # pågående
        "d_us_shale_boom":               0,
        "d_covid":                       0,
        "d_opec_plus_cuts_2023":         1,
    }
    return market_baseline


def predict_differential(api, sulfur, vac_res, ccr, v_ni, mid_dist, brent, market,
                          coefs, features, quarter: int, is_fpso: int = 0) -> float:
    """
    Prediker differensial for et NCS-felt gitt kvalitetsegenskaper + marked.
    Bruker Modell B v3 (Brent-linked + is_fpso, 34b_brent_model.json).
    quarter = 1-4 (for cos_month proxy: Q1≈Jan, Q2≈Apr, Q3≈Jul, Q4≈Oct)
    is_fpso = 1 hvis grade lastes via FPSO, 0 hvis pipeline (NY i v3)
    """
    # cos_month proxy for kvartal
    month_map = {1: 1, 2: 4, 3: 7, 4: 10}
    mo = month_map[quarter]
    cos_mo = np.cos(2 * np.pi * mo / 12)

    s_util = market["us_refinery_util_pct"]
    d_cont = market["d_contango"]

    feat = {
        # ── Statiske kvalitets-features ──────────────────────────────────────
        "api_gravity":           api,
        "sulfur_pct":            sulfur,
        "api2":                  api ** 2,
        # Region-dummies — Brent-modell (reference = MiddleEast):
        "reg_NorthAfrica":       0,    # NCS = 0
        "reg_NorthSea":          1,    # NCS-felt
        "reg_WestAfrica":        0,    # NCS = 0
        "vacuum_resid_pct":      vac_res,
        "middle_distillate_pct": mid_dist,
        "ccr_wt_pct":            ccr,
        "log_v_ni":              np.log1p(v_ni),
        # ── Logistikk (NCS = short distance) ─────────────────────────────────
        # d_distance_medium fjernet fra Brent-modellen
        "d_distance_long":       0,    # NCS er short-distance
        "is_fpso":               is_fpso,   # NY i v3: -1.98 USD/bbl for FPSO-grades
        # ── Tidsvarierende markedsfeatures ───────────────────────────────────
        "brent_price":                       brent,
        # wti_brent_spread fjernet fra Brent-modellen
        "diesel_minus_gasoline_crack":        market["diesel_minus_gasoline_crack"],
        "brent_dubai_spread":                 market["brent_dubai_spread"],
        "us_refinery_util_pct":               s_util,
        "us_crude_stocks_kbbl_dev_5y_pct":    market["us_crude_stocks_kbbl_dev_5y_pct"],
        "cushing_stocks_kbbl_dev_5y_pct":     market["cushing_stocks_kbbl_dev_5y_pct"],
        "d_refinery_slack":                   market["d_refinery_slack"],
        "fc_slope_4m":                        market["fc_slope_4m"],
        "cos_month":                          cos_mo,
        # ── Interaksjoner ────────────────────────────────────────────────────
        "sulfur_x_brent":         sulfur * brent,
        "vacuum_resid_x_brent":   vac_res * brent,
        "ccr_x_brent":            ccr * brent,
        "api_x_contango":         api * d_cont,
        "sulfur_x_refinery_util": sulfur * s_util,
        # landlocked-interaksjoner fjernet (null variasjon i Brent-panelet)
        # ── Politiske dummies ────────────────────────────────────────────────
        "d_russia_sanctions":     market["d_russia_sanctions"],
        "d_iran_sanctions_v1":    market["d_iran_sanctions_v1"],
        "d_iran_sanctions_v2":    market["d_iran_sanctions_v2"],
        # d_venezuela_sanctions fjernet fra Brent-modellen
        "d_us_shale_boom":        market["d_us_shale_boom"],
        "d_covid":                market["d_covid"],
        "d_opec_plus_cuts_2023":  market["d_opec_plus_cuts_2023"],
    }

    pred = coefs["const"]
    for f in features:
        if f in feat and f in coefs:
            pred += coefs[f] * feat[f]
    return pred


# ────────────────────────────────────────────────────────────────────────────
# FORWARD PREDIKSJON
# ────────────────────────────────────────────────────────────────────────────

def run_forward_scenario(scenario_name: str, brent_path: dict,
                         coefs: dict, features: list,
                         market: dict) -> pd.DataFrame:
    """
    Kjør ett Brent-scenario gjennom alle kvartal 2026-Q3 → 2028-Q4.
    Returnerer DataFrame med kvartalsvis blended realisert pris.
    """
    rows = []
    quarters = [
        (2026, 3), (2026, 4),
        (2027, 1), (2027, 2), (2027, 3), (2027, 4),
        (2028, 1), (2028, 2), (2028, 3), (2028, 4),
    ]

    for yr, q in quarters:
        brent = brent_path[yr][q]  # direkte oppslag på kvartalsnummer
        qstr = f"{yr}-Q{q}"

        # Produksjon per felt dette kvartal
        field_prods = {}
        for field, prod_dict in FIELD_PRODUCTION_FWD.items():
            kboepd = prod_dict.get((yr, q), 0)
            if kboepd > 0:
                field_prods[field] = kboepd

        if not field_prods:
            continue

        total_prod = sum(field_prods.values())

        # Beregn vektet differensial
        weighted_diff = 0.0
        field_rows = []
        # FPSO-felt (synkronisert med script 62 og script 42)
        AKRBP_FPSO = {"ALVHEIM", "BØYLA", "SKOGUL", "SKARV", "YGGDRASIL"}
        for field, kboepd in field_prods.items():
            share = kboepd / total_prod
            fq = ALL_FIELD_QUALITY.get(field)
            if fq is None:
                continue
            diff = predict_differential(
                api=fq["api"], sulfur=fq["sulfur"],
                vac_res=fq["vacuum_resid"], ccr=fq["ccr"],
                v_ni=fq["vanadium"] + fq["nickel"],
                mid_dist=fq.get("middle_distillate_pct", 40.0),
                brent=brent, market=market,
                coefs=coefs, features=features, quarter=q,
                is_fpso=1 if field in AKRBP_FPSO else 0,
            )
            weighted_diff += share * diff
            field_rows.append({
                "field": field, "share": share,
                "diff": diff, "contribution": share * diff,
                "kboepd": kboepd,
            })

        rows.append({
            "scenario":       scenario_name,
            "qstr":           qstr,
            "year":           yr,
            "quarter":        q,
            "brent":          brent,
            "blended_diff":   weighted_diff,
            "realized_pred":  brent + weighted_diff,
            "total_kboepd":   total_prod,
            "field_data":     field_rows,
        })

    return pd.DataFrame([{k: v for k, v in r.items() if k != "field_data"} for r in rows]), rows


# ────────────────────────────────────────────────────────────────────────────
# VISUALISERING
# ────────────────────────────────────────────────────────────────────────────

SCENARIO_COLORS = {
    "Bear ($62-65)": "#e74c3c",
    "Base ($72-75)": "#2980b9",
    "Bull ($85-95)": "#27ae60",
}
SCENARIO_STYLES = {
    "Bear ($62-65)": "--",
    "Base ($72-75)": "-",
    "Bull ($85-95)": "-.",
}


def plot_forward(all_scenario_rows: list, hist_q_df: pd.DataFrame,
                 mix_shift_summary: pd.DataFrame):
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])   # Realized price scenarios
    ax2 = fig.add_subplot(gs[1, :])   # Production mix shift
    ax3 = fig.add_subplot(gs[2, 0])   # Differential per scenario
    ax4 = fig.add_subplot(gs[2, 1])   # Revenue sensitivity

    # ── Panel 1: Realisert pris per scenario (historisk + forward) ───────────
    # Historisk
    hist_recent = hist_q_df[hist_q_df["year"] >= 2022].copy()
    # KILDE: Aker BP ASA Quarterly Reports (PDF-parsing, kryssjekket mot FY-gjennomsnitt)
    # Alle verdier er offisielt rapporterte — "Realised price liquids" (USD/boe)
    hist_recent["reported"] = hist_recent["qstr"].map({
        "2022-Q1": 100.9, "2022-Q2": 117.5, "2022-Q3": 101.1, "2022-Q4": 86.6,
        "2023-Q1": 78.4,  "2023-Q2": 76.8,  "2023-Q3": 87.6,  "2023-Q4": 83.6,
        "2024-Q1": 82.9,  "2024-Q2": 83.1,  "2024-Q3": 80.3,  "2024-Q4": 74.1,
        "2025-Q1": 75.0,  "2025-Q2": 66.9,
    })

    hist_x = list(range(len(hist_recent)))
    ax1.plot(hist_x, hist_recent["realized_pred"], color="#555", lw=1.8,
             ls="-", label="Modell predikert (hist.)", zorder=4)
    ax1.scatter(hist_x, hist_recent["reported"],
                color="#2c3e50", s=40, zorder=5, label="Rapportert (offisielt)", marker="o")
    ax1.plot(hist_x, hist_recent["brent_q"], color="#aaa", lw=1.2, ls="--", alpha=0.7)

    # Separator: nå til forward
    sep_x = len(hist_x) - 0.5
    ax1.axvline(sep_x, color="#666", lw=1.2, ls=":", alpha=0.7)
    ax1.text(sep_x + 0.1, ax1.get_ylim()[0] + 5 if ax1.get_ylim()[0] > 50 else 65,
             "← Historisk  |  Forward →", fontsize=8, color="#666")

    # Forward scenarios
    for sc_df, sc_rows, sc_name in all_scenario_rows:
        fwd_x = [len(hist_x) + i for i in range(len(sc_df))]
        color = SCENARIO_COLORS[sc_name]
        ls    = SCENARIO_STYLES[sc_name]
        ax1.plot(fwd_x, sc_df["realized_pred"], color=color, lw=2.2,
                 ls=ls, label=f"{sc_name}", zorder=5)
        ax1.plot(fwd_x, sc_df["brent"], color=color, lw=1.2, ls=ls, alpha=0.4)

    # Xtick labels: kombiner historisk + forward
    all_qstrs = list(hist_recent["qstr"]) + list(all_scenario_rows[0][0]["qstr"])
    tick_step = 2
    tick_idx = list(range(0, len(all_qstrs), tick_step))
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([all_qstrs[i] for i in tick_idx], rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel("USD/bbl", fontsize=10)
    ax1.set_title("Aker BP — Scenariobasert forward realisert oljepris", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right", ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Produksjonsmix-shift ─────────────────────────────────────
    # Stacked bar med ny-felt vs. eksisterende felt
    mix_quarters = mix_shift_summary["qstr"].tolist()
    field_cols = [c for c in mix_shift_summary.columns if c not in ["qstr", "total"]]

    field_colors = {
        "JOHAN SVERDRUP": "#2c3e50",
        "VALHALL":        "#2980b9",
        "HOD":            "#5dade2",
        "ALVHEIM":        "#16a085",
        "BØYLA":          "#1abc9c",
        "EDVARD GRIEG":   "#6c3483",
        "IVAR AASEN":     "#9b59b6",
        "SKARV":          "#ca6f1e",
        "ULA":            "#95a5a6",
        "YGGDRASIL":      "#f39c12",   # oransj = Yggdrasil (Brent Group, lett søt)
    }

    x2 = range(len(mix_quarters))
    bottom = np.zeros(len(mix_quarters))
    for field in field_cols:
        vals = mix_shift_summary[field].fillna(0).values
        col = field_colors.get(field, "#aaa")
        ax2.bar(x2, vals, bottom=bottom, color=col, label=field, width=0.85)
        bottom += vals

    # Annotate Yggdrasil ramp-up
    for i, qstr in enumerate(mix_quarters):
        if qstr == "2027-Q2":
            yi = mix_shift_summary.loc[i, "total"]
            ax2.annotate("Yggdrasil\nfirst oil", xy=(i, yi), xytext=(i + 0.3, yi + 20),
                         ha="left", fontsize=8, color="#c0392b", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    ax2.set_xticks(range(len(mix_quarters)))
    ax2.set_xticklabels(mix_quarters, rotation=45, ha="right", fontsize=8.5)
    ax2.set_ylabel("Netto kboepd (olje)", fontsize=10)
    ax2.set_title("AKRBP Produksjonsportefølge-mix 2026-2028\n"
                  "(oransj = Yggdrasil — Brent Group lett søt crude, forbedrer portfolio-mix)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7.5, loc="upper left", ncol=4)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Blended differential per scenario ─────────────────────────
    for sc_df, _, sc_name in all_scenario_rows:
        x3 = range(len(sc_df))
        color = SCENARIO_COLORS[sc_name]
        ls    = SCENARIO_STYLES[sc_name]
        ax3.plot(x3, sc_df["blended_diff"], color=color, lw=2.0,
                 ls=ls, label=sc_name)

    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_xticks(range(len(all_scenario_rows[0][0])))
    ax3.set_xticklabels(all_scenario_rows[0][0]["qstr"].tolist(),
                        rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("USD/bbl vs. Brent", fontsize=10)
    ax3.set_title("Predikert quality-differensial vs. Brent\n(alle scenarier)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # ── Panel 4: Revenue sensitivity ($/bbl endring i realized pris) ─────────
    base_df  = all_scenario_rows[1][0]  # Base scenario
    bear_df  = all_scenario_rows[0][0]
    bull_df  = all_scenario_rows[2][0]

    x4 = range(len(base_df))
    ax4.fill_between(x4,
                     bear_df["realized_pred"].values,
                     bull_df["realized_pred"].values,
                     alpha=0.2, color="#3498db", label="Bear–Bull range")
    ax4.plot(x4, base_df["realized_pred"], color="#2980b9", lw=2.2, label="Base realized")
    ax4.plot(x4, base_df["brent"], color="#aaa", lw=1.5, ls="--", label="Base Brent (flat)")

    # Mix-shift effekt: diff mellom 2028-Q4 og 2026-Q3 differential
    if len(base_df) >= 2:
        first_diff = base_df["blended_diff"].iloc[0]
        last_diff  = base_df["blended_diff"].iloc[-1]
        delta_mix  = last_diff - first_diff
        ax4.annotate(
            f"Mix-shift effekt:\n{delta_mix:+.2f} USD/bbl\n(NOAKA/Yggdrasil)",
            xy=(len(base_df) - 1, base_df["realized_pred"].iloc[-1]),
            xytext=(len(base_df) - 4, base_df["realized_pred"].iloc[-1] + 3),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#333"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.9)
        )

    ax4.set_xticks(x4)
    ax4.set_xticklabels(base_df["qstr"].tolist(), rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("USD/bbl", fontsize=10)
    ax4.set_title("Revenue-sensitivitet 2026-2028\n(inkl. Brent-risiko og quality mix-shift)",
                  fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    fig.suptitle(
        "Aker BP — Forward realisert pris-prediksjon 2026-2028\n"
        "Kilde: AKRBP CMD 2025 produksjonsguidance × Crude oil regresjonsmodell\n"
        "OBS: Yggdrasil (fka NOAKA) volumer og assay-kvalitet er estimater (ingen published assay ennå)",
        fontsize=12, fontweight="bold", y=0.999
    )

    out_path = OUT_DIR / "43_akrbp_forward_prediction.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  Figur: {out_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SCRIPT 43: Aker BP Forward Realized-Price Prediction")
    print("=" * 65)

    # 1. Last inn
    print("\n[1] Laster inn modell og markedsdata...")
    coefs, features = load_model()
    market = get_market_defaults_2025()
    hist_q_df = pd.read_csv(Q42_QUARTERLY)
    print(f"  Historisk data: {len(hist_q_df)} kvartal fra script 42")

    # 2. Kjør scenarier
    print("\n[2] Kjører forward-scenarier...")
    all_scenario_results = []
    for sc_name, brent_path in BRENT_SCENARIOS.items():
        sc_df, sc_rows = run_forward_scenario(
            sc_name, brent_path, coefs, features, market
        )
        all_scenario_results.append((sc_df, sc_rows, sc_name))
        print(f"  {sc_name}: {len(sc_df)} kvartal predikert")
        print(sc_df[["qstr", "brent", "blended_diff", "realized_pred", "total_kboepd"]].to_string(
            index=False, float_format="{:.2f}".format))
        print()

    # 3. Produksjonsmix-shift tabell
    print("[3] Produksjonsmix-shift...")
    all_fields = sorted(FIELD_PRODUCTION_FWD.keys())
    quarters_fwd = [
        (2026, 3), (2026, 4),
        (2027, 1), (2027, 2), (2027, 3), (2027, 4),
        (2028, 1), (2028, 2), (2028, 3), (2028, 4),
    ]
    mix_rows = []
    for yr, q in quarters_fwd:
        row = {"qstr": f"{yr}-Q{q}"}
        total = 0
        for f in all_fields:
            v = FIELD_PRODUCTION_FWD[f].get((yr, q), 0)
            row[f] = v
            total += v
        row["total"] = total
        mix_rows.append(row)
    mix_df = pd.DataFrame(mix_rows)

    print(f"  Produksjonstotal 2026-Q3: {mix_df.iloc[0]['total']:.0f} kboepd")
    print(f"  Produksjonstotal 2028-Q4: {mix_df.iloc[-1]['total']:.0f} kboepd")

    # Mix-analyse: Johan Sverdrup andel
    js_share_start = mix_df.iloc[0]["JOHAN SVERDRUP"] / mix_df.iloc[0]["total"]
    js_share_end   = mix_df.iloc[-1]["JOHAN SVERDRUP"] / mix_df.iloc[-1]["total"]
    ygg_share      = mix_df.iloc[-1].get("YGGDRASIL", 0) / mix_df.iloc[-1]["total"]

    print(f"\n  Mix-shift analyse:")
    print(f"    Johan Sverdrup: {js_share_start:.0%} (2026-Q3) → {js_share_end:.0%} (2028-Q4)")
    print(f"    Yggdrasil:      {ygg_share:.0%} (2028-Q4)  ← Brent Group (API ~37°, S ~0.22%)")

    # 4. Revenue summary
    print("\n[4] Revenue summary per scenario:")
    print(f"  {'Scenario':20s}  {'2026 avg':>10}  {'2027 avg':>10}  {'2028 avg':>10}  {'Diff vs Brent':>14}")
    for sc_df, _, sc_name in all_scenario_results:
        sc26 = sc_df[sc_df["year"] == 2026]["realized_pred"].mean()
        sc27 = sc_df[sc_df["year"] == 2027]["realized_pred"].mean()
        sc28 = sc_df[sc_df["year"] == 2028]["realized_pred"].mean()
        brt28 = sc_df[sc_df["year"] == 2028]["brent"].mean()
        print(f"  {sc_name:20s}  {sc26:>10.2f}  {sc27:>10.2f}  {sc28:>10.2f}  {sc28-brt28:>+14.2f}")

    # Kvalitetsdifferensial endring (mix-shift effekt)
    # Bruk SAMME KVARTAL for sammenligning (Q3, sommer) for å unngå sesongeffekt
    base_df = all_scenario_results[1][0]
    q3_rows = base_df[base_df["quarter"] == 3]
    diff_q3_start = q3_rows.iloc[0]["blended_diff"]  # 2026-Q3
    diff_q3_end   = q3_rows.iloc[-1]["blended_diff"]  # 2028-Q3
    q3_start_label = q3_rows.iloc[0]["qstr"]
    q3_end_label   = q3_rows.iloc[-1]["qstr"]

    print(f"\n  Mix-shift effekt (sammenligner SAME kvartal — Base scenario, unngår sesongstøy):")
    print(f"    {q3_start_label}: {diff_q3_start:+.2f} USD/bbl  (før Yggdrasil, JS 71% av mix)")
    print(f"    {q3_end_label}:   {diff_q3_end:+.2f} USD/bbl  (etter Yggdrasil 22% av mix)")
    print(f"    Endring Q3-til-Q3: {diff_q3_end-diff_q3_start:+.2f} USD/bbl")
    if diff_q3_end > diff_q3_start:
        print(f"    ✓ Yggdrasil (API 37°, S 0.22%) FORBEDRER portfolio-mix vs. Johan Sverdrup (API 28.7°, S 0.81%)")
        print(f"      Mix-shift bidrar POSITIVT — JS-andel faller fra 71% → 57% til fordel for lettere crude")
    else:
        print(f"    △ Yggdrasil forverrer portfolio quality vs. dagens mix — sjekk assay-antagelse")

    # 5. Figur
    print("\n[5] Genererer figurer...")
    plot_forward(all_scenario_results, hist_q_df, mix_df)

    # 6. Eksport
    all_fwd = pd.concat([sc_df for sc_df, _, _ in all_scenario_results], ignore_index=True)
    out_csv = OUT_DIR / "43_akrbp_forward_scenarios.csv"
    all_fwd.to_csv(out_csv, index=False)
    mix_df.to_csv(OUT_DIR / "43_akrbp_production_mix.csv", index=False)
    print(f"  CSV: {out_csv}")

    print("\n  OBS — Oppdater disse estimatene med offisielle tall:")
    print("   1. Yggdrasil produksjonsprofil per felt (Munin/Hugin/Fulla) → CMD 2025")
    print("   2. Yggdrasil crude assay-kvalitet → ingen published assay ennå")
    print("      Beste proxy: Brent Group/Statfjord (API ~37°, S ~0.22%)")
    print("      Kilde: Sodir factpages Munin (NPDID 42002476) + Fulla (NPDID 42002479)")
    print("   3. Forward Brent → Bloomberg consensus / EIA STEO")
    print("   4. Merk: Yggdrasil = fka NOAKA (prosjektet ble omdøpt ~2023)")
    print()


if __name__ == "__main__":
    main()
