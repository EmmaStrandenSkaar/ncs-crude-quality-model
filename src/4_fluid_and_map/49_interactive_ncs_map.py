"""
Script 49 — Interaktivt NCS-kart (FactMaps-stil) med modellpredikert kvalitetsdiff.

LAG:
  Bakgrunn:   CartoDB Positron (lys, nøytral)
  Lag 1 (på): Felt-polygoner fargelagt etter modellpredikert differensial vs. Brent
              (heatmap: rød = rabatt, grønn = premium)
  Lag 2 (på): Aktive funn under utbygging (Yggdrasil/Hugin/Munin/Fulla)
              med stiplet kant og 'planlagt produksjon' i popup
  Lag 3 (av): Produksjonslisenser fargelagt etter operatør
  Lag 4 (av): NCS-blokk-grid (referanse)

POPUP per felt:
  · Felt-navn, operatør, status, lisens, oppstartsår
  · Assay-kvalitet (API, S, vac.resid, CCR, middle distillate)
  · Modellpredikert differensial vs. Dated Brent (USD/bbl)
  · Top 5 forklaringsvariabler (coef × feature-verdi)
  · Siste normpris-differensial hvor tilgjengelig

OUTPUT:
  data/processed/49_ncs_interactive_map.html  — selvstendig HTML, ~10-20 MB
"""

from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import json
import numpy as np
import pandas as pd
import folium
from folium import plugins
from folium.map import CustomPane
from branca.colormap import LinearColormap
from branca.element import Template, MacroElement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEO_DIR      = PROJECT_ROOT / "data" / "raw" / "sodir_geo"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
OUT_HTML     = PROC_DIR / "49_ncs_interactive_map.html"

MODEL_JSON   = PROC_DIR / "34b_brent_model.json"
IMPUTED_CSV  = PROC_DIR / "63_ncs_field_quality.csv"                              # Script 63 felt→kvalitet
PROFILES_JSON = PROC_DIR / "64_field_production_profiles.json"                    # Script 64 produksjonsprofiler

# ── DIREKTE assay-mapping (★★★ Equinor lab-assay) ──────────────────────────
# Felt med egen publisert offisiell Equinor XLSX-assay
DIRECT_ASSAY = {
    "JOHAN SVERDRUP": "Johan Sverdrup",
    "ALVHEIM":        "Alvheim",
    "GRANE":          "Grane",
    "OSEBERG":        "Oseberg",
    "TROLL":          "Troll",
    "EKOFISK":        "Ekofisk",
    "STATFJORD":      "Statfjord",
    "GULLFAKS":       "Gullfaks",
    "HEIDRUN":        "Heidrun",
    "NORNE":          "Norne",
    "SKARV":          "Skarv",
    "BALDER":         "Balder",
    "DRAUGEN":        "Draugen",
    "ÅSGARD":         "Asgard",
    "GUDRUN":         "Gudrun",
    "GOLIAT":         "Goliat",
    "GINA KROG":      "Gina Krog",
    "JOTUN":          "Jotun",
    "KNARR":          "Knarr",
    "MARTIN LINGE":   "Martin Linge",
    "NJORD":          "Njord",
}

# ── FPSO vs pipeline-flag per Sodir-feltnavn (UPPERCASE) ───────────────────
# Synkronisert med script 62. Brukes for is_fpso-featuren i Modell v3.
SODIR_FPSO_FIELDS = {
    "ALVHEIM", "BØYLA", "SKOGUL", "VOLUND", "VILJE",
    "SKARV", "AASTA HANSTEEN",
    "ÅSGARD", "HEIDRUN", "NORNE", "DRAUGEN",
    "GOLIAT", "BALDER", "JOTUN", "GINA KROG", "MARTIN LINGE",
    "KNARR", "NJORD", "MARULK", "SKULD", "URD", "MARIA",
}


# ── BLEND-PROXY (★★ faktisk eksportstream) ─────────────────────────────────
# Felt som sammenblandes og eksporteres som én navngitt blend.
# Formatet: (assay_grade, "name of actual export blend")
BLEND_PROXY = {
    "BØYLA":           ("Alvheim",       "Alvheim FPSO-stream"),
    "SKOGUL":          ("Alvheim",       "Alvheim FPSO-stream"),
    "VOLUND":          ("Alvheim",       "Alvheim FPSO-stream"),
    "VILJE":           ("Alvheim",       "Alvheim FPSO-stream"),
    "EDVARD GRIEG":    ("Grane",         "Grane Blend (Sture)"),
    "IVAR AASEN":      ("Grane",         "Grane Blend via EG"),
    "SOLVEIG":         ("Grane",         "Grane Blend via EG"),
    "SVERDRUP PHASE 2":("Johan Sverdrup","Johan Sverdrup-stream"),
    "VALHALL":         ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "HOD":             ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "ELDFISK":         ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "EMBLA":           ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "ULA":             ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "TAMBAR":          ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "TAMBAR ØST":      ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "GYDA":            ("Ekofisk",       "Ekofisk Blend (Norpipe/Teesside)"),
    "OSEBERG ØST":     ("Oseberg",       "Oseberg Blend (Sture)"),
    "OSEBERG SØR":     ("Oseberg",       "Oseberg Blend (Sture)"),
    "FRAM":            ("Troll",         "Troll Blend (Mongstad)"),
    "FRAM H-NORD":     ("Troll",         "Troll Blend (Mongstad)"),
    "STATFJORD NORD":  ("Statfjord",     "Statfjord Blend"),
    "STATFJORD ØST":   ("Statfjord",     "Statfjord Blend"),
    "SYGNA":           ("Statfjord",     "Statfjord Blend"),
    "SNORRE":          ("Statfjord",     "Statfjord Blend (proxy)"),
    "VIGDIS":          ("Statfjord",     "Statfjord-stream"),
    "TORDIS":          ("Statfjord",     "Statfjord-stream"),
    "VISUND":          ("Gullfaks",      "Gullfaks Blend (Mongstad)"),
    "KVITEBJØRN":      ("Gullfaks",      "Gullfaks-area stream"),
    "VALEMON":         ("Gullfaks",      "Gullfaks-area stream"),
    "MARULK":          ("Norne",         "Norne FPSO-stream"),
    "SKULD":           ("Norne",         "Norne FPSO-stream"),
    "URD":             ("Norne",         "Norne FPSO-stream"),
    "AASTA HANSTEEN":  ("Skarv",         "kondensat/NGL-proxy"),
    "MARIA":           ("Heidrun",       "Heidrun-stream"),
}

# ── Whitelist av virkelig store forward-utbygginger (med kuratert assay) ──
# Disse vises i "Forward-felt"-laget med hand-tunet kvalitetsestimat
MAJOR_FORWARD_FIELDS = {
    "HUGIN", "MUNIN", "FULLA",      # Yggdrasil hub (~2027)
    "WISTING",                       # Barentshavet (2028+)
    "BALDER FUTURE", "JOTUN",        # Balder/Jotun re-development (2027)
    "DVALIN",                        # Tied-back to Heidrun
    "HALTEN EAST",                   # Norwegian Sea tie-back
    "NOAKA",                         # gammelt navn for Yggdrasil
    "OPHELIA",                       # tidlig fase
    "KING LEAR",                     # ConocoPhillips
    "SYMRA",                         # Aker BP/Equinor
    "TYRVING",                       # tied to Alvheim
    "DRAUPNE",                       # alternativ Yggdrasil-navn
}

# ── Funn-statuser vi viser i 'Funn under utvikling'-laget (NAV-upside) ─────
# Disse er IKKE besluttet utbygd, men har realistisk sjanse → relevant for NAV
DEVELOPMENT_CANDIDATE_STATUSES = {
    "Production in clarification phase",   # ~12 — neste i køen
    "Production likely but unclarified",   # ~38 — sannsynlig utvikling
    "Production not evaluated",            # ~41 — under tidlig vurdering
}

# Eksempler vi vil sikre er med (kjente navn fra equity-research)
NOTABLE_DISCOVERY_NAMES = {
    "KING LEAR", "PANDORA", "TONGUE", "LUPA", "BASK", "GODDO",
    "CARMEN", "SLAGUGLE", "EIRIN", "GRIND", "RØMER", "TROLDHAUGEN",
    "OFELIA", "OPHIDIAN",
}

# ── Forward-felt med antatt kvalitet (estimater) ─────────────────────────────
FORWARD_FIELDS_QUALITY = {
    "HUGIN":   dict(api=37.0, sulfur=0.22, vac_res=13.0, ccr=2.8, mid_dist=42.0,
                    v_ni=4.5, label="Yggdrasil – Hugin (Brent Group, first oil 2027)",
                    proxy="Statfjord/Oseberg Brent Group analog"),
    "MUNIN":   dict(api=37.0, sulfur=0.22, vac_res=13.0, ccr=2.8, mid_dist=42.0,
                    v_ni=4.5, label="Yggdrasil – Munin (Brent Group, first oil 2027)",
                    proxy="Statfjord/Oseberg Brent Group analog"),
    "FULLA":   dict(api=50.0, sulfur=0.05, vac_res=2.0,  ccr=0.3, mid_dist=46.0,
                    v_ni=0.7, label="Yggdrasil – Fulla (kondensat, first oil 2027)",
                    proxy="Skarv kondensat-analog"),
    "WISTING": dict(api=37.0, sulfur=0.34, vac_res=14.0, ccr=2.5, mid_dist=40.0,
                    v_ni=3.5, label="Wisting (Barentshavet, planlagt 2028+)",
                    proxy="Goliat-analog (samme havområde)"),
    "TYRVING": dict(api=34.5, sulfur=0.40, vac_res=8.7,  ccr=1.1, mid_dist=44.8,
                    v_ni=3.6, label="Tyrving (tied-back Alvheim, 2025+)",
                    proxy="Alvheim FPSO-stream"),
}


# ────────────────────────────────────────────────────────────────────────────
# DATA-INNLASTING
# ────────────────────────────────────────────────────────────────────────────

def load_geojson(name: str) -> dict:
    path = GEO_DIR / f"{name}.geojson"
    return json.loads(path.read_text())


def load_model() -> tuple[dict, list, dict]:
    """Returnerer coefs, features, metrics."""
    m = json.loads(MODEL_JSON.read_text())
    return m["coefficients"], m["features"], m["metrics"]


def load_assay_database() -> pd.DataFrame:
    """Hent assay-data per grade fra regresjons-panelet (statisk per grade)."""
    df = pd.read_csv(PROC_DIR / "regression_panel.csv")
    cols = ["grade", "api_gravity", "sulfur_pct", "vacuum_resid_pct",
            "ccr_wt_pct", "middle_distillate_pct", "log_v_ni"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("grade")[cols[1:]].mean().reset_index()
    return g


# ────────────────────────────────────────────────────────────────────────────
# GEOGRAFISK PROXY: finn nærmeste felt med assay (samme havområde)
# ────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def polygon_centroid(geom: dict) -> tuple[float, float] | None:
    """Approksimer polygon-sentroid som middel av alle koordinater."""
    coords = []
    def walk(c):
        if isinstance(c, (list, tuple)):
            if c and isinstance(c[0], (int, float)) and len(c) >= 2:
                coords.append((c[0], c[1]))
            else:
                for sub in c:
                    walk(sub)
    walk(geom.get("coordinates", []))
    if not coords:
        return None
    lon = sum(c[0] for c in coords) / len(coords)
    lat = sum(c[1] for c in coords) / len(coords)
    return (lat, lon)


def build_centroid_index(fields_fc: dict) -> dict:
    """Returner dict: feltnavn → (lat, lon, main_area)."""
    idx = {}
    for f in fields_fc["features"]:
        name = (f.get("properties", {}).get("fldName") or "").upper()
        area = f.get("properties", {}).get("fldMainArea", "")
        cen = polygon_centroid(f.get("geometry", {}))
        if name and cen:
            idx[name] = (cen[0], cen[1], area)
    return idx


def find_nearest_proxy(target_name: str, centroids: dict,
                        candidate_names: set) -> tuple[str, float] | None:
    """
    Finn nærmeste felt blant `candidate_names` (felt vi har assay for)
    til `target_name`. Foretrekker samme havområde, faller tilbake til hvilket
    som helst hvis ingen i samme område er innen 300 km.
    """
    if target_name not in centroids:
        return None
    t_lat, t_lon, t_area = centroids[target_name]

    same_area, other = [], []
    for cand in candidate_names:
        if cand == target_name or cand not in centroids:
            continue
        c_lat, c_lon, c_area = centroids[cand]
        d = haversine_km(t_lat, t_lon, c_lat, c_lon)
        if c_area == t_area:
            same_area.append((cand, d))
        else:
            other.append((cand, d))

    if same_area:
        same_area.sort(key=lambda x: x[1])
        return same_area[0]
    if other:
        other.sort(key=lambda x: x[1])
        # Tillat kun cross-area hvis avstanden er rimelig (<500 km)
        if other[0][1] < 500:
            return other[0]
    return None


def load_normpris_latest() -> dict:
    """Siste 4Q-gjennomsnitt av normpris-differensial per felt."""
    df = pd.read_csv(PROC_DIR / "normpris_differentials_long.csv")
    df["quarter"] = ((df["month"].astype(int) - 1) // 3 + 1).astype(int)
    df["qstr"]    = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    nq = (df.groupby(["field", "qstr"])["differential_usd"].mean()
            .reset_index())
    nq = nq.sort_values("qstr")
    out = {}
    for field, grp in nq.groupby("field"):
        last4 = grp.tail(4)
        if len(last4) >= 2:
            out[field.upper()] = {
                "value":    float(last4["differential_usd"].mean()),
                "n":        len(last4),
                "qstr_end": last4["qstr"].iloc[-1],
            }
    return out


# ────────────────────────────────────────────────────────────────────────────
# MODELL-PREDIKSJON
# ────────────────────────────────────────────────────────────────────────────

def build_field_features(api, sulfur, vac_res, ccr, mid_dist, v_ni,
                          brent=75.0, refutil=92.0, is_fpso=0) -> dict:
    """Bygg en feature-dict for NCS-felt med 'standard' markedsforhold.
    is_fpso: 1 hvis grade lastes via FPSO, 0 hvis pipeline (NY i v3).
    """
    return {
        # Statiske kvalitets-features
        "api_gravity":           api,
        "sulfur_pct":            sulfur,
        "api2":                  api ** 2,
        "reg_NorthAfrica":       0,
        "reg_NorthSea":          1,
        "reg_WestAfrica":        0,
        "vacuum_resid_pct":      vac_res,
        "middle_distillate_pct": mid_dist,
        "ccr_wt_pct":            ccr,
        "log_v_ni":              np.log1p(v_ni),
        # Logistikk
        "d_distance_long":       0,
        "is_fpso":               is_fpso,   # NY i v3: -1.98 USD/bbl for FPSO
        # Markeds-features (baseline-snitt)
        "brent_price":                       brent,
        "diesel_minus_gasoline_crack":        15.0,
        "brent_dubai_spread":                 1.5,
        "us_refinery_util_pct":               refutil,
        "us_crude_stocks_kbbl_dev_5y_pct":    0.0,
        "cushing_stocks_kbbl_dev_5y_pct":     0.0,
        "d_refinery_slack":                   0,
        "fc_slope_4m":                        0.0,
        "cos_month":                          0.0,
        # Interaksjoner
        "sulfur_x_brent":         sulfur * brent,
        "vacuum_resid_x_brent":   vac_res * brent,
        "ccr_x_brent":            ccr * brent,
        "api_x_contango":         0.0,
        "sulfur_x_refinery_util": sulfur * refutil,
        # Politiske dummies
        "d_russia_sanctions":     1,
        "d_iran_sanctions_v1":    0,
        "d_iran_sanctions_v2":    1,
        "d_us_shale_boom":        0,
        "d_covid":                0,
        "d_opec_plus_cuts_2023":  1,
    }


def predict(feat: dict, coefs: dict, features: list) -> float:
    """Bare predikert differensial (sum av coef × feature)."""
    pred = coefs.get("const", 0.0)
    for f in features:
        if f in feat and f in coefs:
            pred += coefs[f] * feat[f]
    return pred


def compute_ncs_baseline(assay_db: pd.DataFrame, ncs_grades: list[str]) -> dict:
    """Snitt-verdier på tvers av NCS-grades = referansen vi måler hver felt mot.
    NaN-verdier ekskluderes fra snittet (.mean() i pandas ignorerer NaN by default).
    """
    sub = assay_db[assay_db["grade"].isin(ncs_grades)]

    def mean_or(col: str, default: float) -> float:
        if col not in sub.columns:
            return float(default)
        v = sub[col].mean()
        return float(v) if not pd.isna(v) else float(default)

    log_vni = mean_or("log_v_ni", float(np.log1p(5.0)))
    return dict(
        api      = mean_or("api_gravity",           35.0),
        sulfur   = mean_or("sulfur_pct",             0.50),
        vac_res  = mean_or("vacuum_resid_pct",      15.0),
        ccr      = mean_or("ccr_wt_pct",             2.0),
        mid_dist = mean_or("middle_distillate_pct", 40.0),
        v_ni     = float(np.expm1(log_vni)),
    )


def quality_impact_decomposition(assay: dict, baseline: dict, coefs: dict,
                                   brent: float = 75.0,
                                   refutil: float = 92.0) -> list:
    """
    For hver kvalitetsegenskap: regn ut hvor mye den drar dette feltets differensial
    OPP eller NED relativt til et gjennomsnittlig NCS-felt.

    Aggregerer hovedeffekt + alle interaksjoner som involverer egenskapen.
    Returnerer liste sortert etter abs(USD/bbl-utslag), top 5.
    """
    impacts = []

    # ── API gravity (kombinerer api_gravity + api2) ─────────────────────────
    api_f, api_b = assay["api"], baseline["api"]
    a1 = coefs.get("api_gravity", 0)
    a2 = coefs.get("api2", 0)
    delta_api = (a1 * api_f + a2 * api_f**2) - (a1 * api_b + a2 * api_b**2)
    impacts.append({
        "label":     "API gravity",
        "field_v":   f"{api_f:.1f}°",
        "base_v":    f"{api_b:.1f}°",
        "impact":    delta_api,
        "explain":   "tung/lett crude",
    })

    # ── Svovel (sulfur_pct + sulfur_x_brent + sulfur_x_refinery_util) ───────
    s_f, s_b = assay["sulfur"], baseline["sulfur"]
    s_marg = (
        coefs.get("sulfur_pct", 0)
        + coefs.get("sulfur_x_brent", 0) * brent
        + coefs.get("sulfur_x_refinery_util", 0) * refutil
    )
    delta_s = s_marg * (s_f - s_b)
    impacts.append({
        "label":     "Svovel-innhold",
        "field_v":   f"{s_f:.2f}%",
        "base_v":    f"{s_b:.2f}%",
        "impact":    delta_s,
        "explain":   "søt vs sur crude",
    })

    # ── Vacuum residue (vacuum_resid_pct + vacuum_resid_x_brent) ───────────
    vr_f, vr_b = assay["vac_res"], baseline["vac_res"]
    vr_marg = (
        coefs.get("vacuum_resid_pct", 0)
        + coefs.get("vacuum_resid_x_brent", 0) * brent
    )
    delta_vr = vr_marg * (vr_f - vr_b)
    impacts.append({
        "label":     "Vacuum residue",
        "field_v":   f"{vr_f:.1f}%",
        "base_v":    f"{vr_b:.1f}%",
        "impact":    delta_vr,
        "explain":   "bunnfraksjon — mer = lavere verdi",
    })

    # ── Middle distillate yield ─────────────────────────────────────────────
    md_f, md_b = assay["mid_dist"], baseline["mid_dist"]
    md_marg = coefs.get("middle_distillate_pct", 0)
    delta_md = md_marg * (md_f - md_b)
    impacts.append({
        "label":     "Middle distillate yield",
        "field_v":   f"{md_f:.1f}%",
        "base_v":    f"{md_b:.1f}%",
        "impact":    delta_md,
        "explain":   "diesel/jet-utbytte",
    })

    # ── Conradson carbon (ccr_wt_pct + ccr_x_brent) ────────────────────────
    ccr_f, ccr_b = assay["ccr"], baseline["ccr"]
    ccr_marg = coefs.get("ccr_wt_pct", 0) + coefs.get("ccr_x_brent", 0) * brent
    delta_ccr = ccr_marg * (ccr_f - ccr_b)
    impacts.append({
        "label":     "Conradson carbon",
        "field_v":   f"{ccr_f:.2f}%",
        "base_v":    f"{ccr_b:.2f}%",
        "impact":    delta_ccr,
        "explain":   "koksdannelse — vanskelig å raffinere",
    })

    # ── Metaller (V+Ni) ─────────────────────────────────────────────────────
    vni_f, vni_b = assay["v_ni"], baseline["v_ni"]
    vni_marg = coefs.get("log_v_ni", 0)
    delta_vni = vni_marg * (np.log1p(vni_f) - np.log1p(vni_b))
    impacts.append({
        "label":     "Metaller (V+Ni)",
        "field_v":   f"{vni_f:.1f} ppm",
        "base_v":    f"{vni_b:.1f} ppm",
        "impact":    delta_vni,
        "explain":   "katalysator-forgiftning",
    })

    impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return impacts[:5]


# ────────────────────────────────────────────────────────────────────────────
# POPUP-HTML
# ────────────────────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "api_gravity":           "API gravity",
    "sulfur_pct":            "Svovel (%)",
    "api2":                  "API² (ikke-lineær)",
    "reg_NorthSea":          "Nordsjø-region",
    "reg_WestAfrica":        "Vest-Afrika",
    "reg_NorthAfrica":       "Nord-Afrika",
    "vacuum_resid_pct":      "Vacuum resid (%)",
    "middle_distillate_pct": "Middle distillate (%)",
    "ccr_wt_pct":            "Conradson Carbon (%)",
    "log_v_ni":              "log(V+Ni)",
    "d_distance_long":       "Long-distance dummy",
    "brent_price":           "Brent-pris",
    "diesel_minus_gasoline_crack": "Diesel-gasolin crack",
    "brent_dubai_spread":    "Brent-Dubai spread",
    "us_refinery_util_pct":  "US raffinerings-util",
    "us_crude_stocks_kbbl_dev_5y_pct": "US lager-dev",
    "cushing_stocks_kbbl_dev_5y_pct":  "Cushing lager-dev",
    "d_refinery_slack":      "Raffinerings-slakk",
    "fc_slope_4m":           "Forward-kurve 4M",
    "cos_month":             "Sesong (cos)",
    "sulfur_x_brent":        "Svovel × Brent",
    "vacuum_resid_x_brent":  "VacResid × Brent",
    "ccr_x_brent":           "CCR × Brent",
    "api_x_contango":        "API × contango",
    "sulfur_x_refinery_util": "Svovel × US util",
    "d_russia_sanctions":    "Russia-sanksjoner",
    "d_iran_sanctions_v1":   "Iran-sanksj. v1",
    "d_iran_sanctions_v2":   "Iran-sanksj. v2",
    "d_us_shale_boom":       "US shale boom",
    "d_covid":               "COVID",
    "d_opec_plus_cuts_2023": "OPEC+ kutt 2023",
}


def source_badge(source_type: str, source_label: str) -> str:
    """Tydelig merkelapp for hvor assay-dataen kommer fra."""
    cfg = {
        "DIRECT":  ("★★★", "#27AE60", "EGEN ASSAY"),
        "BLEND":   ("★★",  "#2980B9", "BLEND-PROXY"),
        "IMPUTED": ("★★",  "#16A085", "SCRIPT 63 (MEDIAN + SODIR-DST)"),
        "PROXY":   ("★",   "#E67E22", "GEOGRAFISK PROXY (ESTIMAT)"),
        "FORWARD": ("◆",   "#8E44AD", "FORWARD-ESTIMAT"),
    }
    icon, color, tag = cfg.get(source_type, ("?", "#999", "UKJENT"))
    return (
        f"<div style='background:white;padding:3px 6px;border-left:3px solid {color};"
        f"font-size:9px;color:{color};font-weight:bold;margin:3px 0 4px 0;"
        f"display:flex;justify-content:space-between;align-items:center;'>"
        f"<span>{icon} {tag}</span>"
        f"<span style='color:#666;font-weight:normal;font-style:italic;font-size:8.5px;'>{source_label}</span>"
        f"</div>"
    )


# ────────────────────────────────────────────────────────────────────────────
# SCRIPT 63 KVALITETS-FALLBACK + V5.1 DECLINE-KURVER
# ────────────────────────────────────────────────────────────────────────────
# main_area → log(avstand til Rotterdam), avledet fra hardkodede assay-verdier
AREA_LOG_DIST = {"North sea": 6.50, "Norwegian sea": 7.30, "Barents sea": 7.70, "Unknown": 6.60}

def load_imputed_quality() -> dict:
    """Script 63: felt → kvalitets-assay-dict (samme format som get_assay_values)."""
    if not IMPUTED_CSV.exists():
        return {}
    df = pd.read_csv(IMPUTED_CSV)
    out = {}
    for _, r in df.iterrows():
        out[str(r["field"]).upper().strip()] = dict(
            api=float(r["api_gravity"]), sulfur=float(r["sulfur_pct"]),
            vac_res=float(r["vacuum_resid_pct"]), ccr=float(r["ccr_pct"]),
            mid_dist=float(r["middle_distillate_pct"]),
            v_ni=float(r["vanadium_ppm"]) + float(r["nickel_ppm"]),
            tier=str(r["tier"]),
        )
    return out

def load_production_profiles() -> dict:
    """Script 64: felt → produksjonsprofil (historikk + predikert / forward forecast)."""
    if not PROFILES_JSON.exists():
        return {}
    with open(PROFILES_JSON) as f:
        return json.load(f)


def indicative_discovery_profile(D_pred: float, analog_field: str,
                                  ramp_mo: float = 9, plat_mo: float = 9,
                                  horizon_yr: float = 18) -> dict:
    """
    Indikativ produksjons-TYPEKURVE for et ikke-besluttet funn (normalisert til peak%).
    Ressursvolum er ukjent for funn → vi viser FORMEN (ramp→platå→decline) med
    decline-raten fra nærmeste analog-felt. Tydelig merket som indikativ.
    """
    def shape(ti):
        m = ti * 12
        if m < ramp_mo:
            return 100.0 / (1 + np.exp(-0.6 * (m - ramp_mo / 2)))   # logistisk ramp
        if m < ramp_mo + plat_mo:
            return 100.0
        return 100.0 * np.exp(-D_pred * (ti - (ramp_mo + plat_mo) / 12.0))
    # årlige forecast-søyler
    fcst_bars = [[y, round(float(shape(y)), 1)] for y in range(0, int(horizon_yr) + 1)]
    # jevn stiplet linje
    t = np.linspace(0, horizon_yr, 50)
    dline = [[round(float(ti), 2), round(float(shape(ti)), 1)] for ti in t]
    return {
        "type": "discovery",
        "stage": "funn (ikke besluttet utbygd)",
        "D_pred": round(float(D_pred), 4),
        "analog_field": analog_field,
        "fcst_bars": fcst_bars,
        "decline_line": dline,
    }


def production_chart_svg(profile: dict, w: int = 280, h: int = 120) -> str:
    """
    Inline SVG SØYLEDIAGRAM: produksjon (% av peak) per år.
      · Mørke søyler = historisk faktisk produksjon
      · Lyse søyler  = forecast (predikert decline anvendt på siste faktiske prod.)
      · Stiplet linje = predikert decline-kurve over søylene
    Forward/funn: alle søyler er forecast (lyse), stiplet linje = lifecycle-kurve.
    """
    padL, padR, padT, padB = 26, 8, 8, 16
    plot_w, plot_h = w - padL - padR, h - padT - padB

    ptype = profile.get("type", "producing")
    hist_bars = profile.get("hist_bars") or []
    fcst_bars = profile.get("fcst_bars") or []
    dline = profile.get("decline_line") or []

    all_bars = hist_bars + fcst_bars
    if not all_bars:
        return ""
    years = [b[0] for b in all_bars]
    x0, x1 = min(years), max(years)
    span = max(1, x1 - x0 + 1)
    ymax = max(110.0, max(b[1] for b in all_bars), max((p[1] for p in dline), default=0))

    bw = plot_w / span                       # søylebredde per år
    def bx(yr): return padL + (yr - x0) / span * plot_w
    def sy(y): return padT + (1 - y / ymax) * plot_h

    # fargevalg per type
    if ptype == "forward":
        hist_col, fcst_col, line_col = "#2471a3", "#a9dfbf", "#16a085"
    elif ptype == "discovery":
        hist_col, fcst_col, line_col = "#2471a3", "#d7bde2", "#8e44ad"
    else:
        hist_col, fcst_col, line_col = "#2471a3", "#aed6f1", "#e74c3c"

    # gridlinjer + y-akse-labels (0/50/100)
    grids = ""
    for yv in (0, 50, 100):
        yy = sy(yv)
        grids += (f"<line x1='{padL}' y1='{yy:.0f}' x2='{w-padR}' y2='{yy:.0f}' "
                  f"stroke='#eee' stroke-width='1'/>"
                  f"<text x='{padL-3}' y='{yy+3:.0f}' font-size='7' fill='#bbb' text-anchor='end'>{yv}</text>")

    # søyler
    bars_svg = ""
    gap = bw * 0.16
    for yr, val in hist_bars:
        x = bx(yr) + gap / 2
        y = sy(val)
        bh = (padT + plot_h) - y
        bars_svg += f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw-gap:.1f}' height='{max(bh,0):.1f}' fill='{hist_col}' opacity='0.9'/>"
    for yr, val in fcst_bars:
        x = bx(yr) + gap / 2
        y = sy(val)
        bh = (padT + plot_h) - y
        bars_svg += f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw-gap:.1f}' height='{max(bh,0):.1f}' fill='{fcst_col}' opacity='0.85'/>"

    # forecast-skille (vertikal markør der historikk slutter)
    sep = ""
    if hist_bars and fcst_bars:
        xs = bx(fcst_bars[0][0])
        sep = (f"<line x1='{xs:.1f}' y1='{padT}' x2='{xs:.1f}' y2='{padT+plot_h}' "
               f"stroke='#ccc' stroke-width='1' stroke-dasharray='2,2'/>")

    # stiplet decline-linje (sentrert over søylene → +bw/2)
    line_svg = ""
    if dline:
        d = " ".join(f"{bx(x)+bw/2:.1f},{sy(y):.1f}" for x, y in dline)
        line_svg = (f"<polyline points='{d}' fill='none' stroke='{line_col}' "
                    f"stroke-width='1.6' stroke-dasharray='4,2' opacity='0.9'/>")

    return (f"<svg width='{w}' height='{h}' style='display:block;margin-top:2px;'>"
            f"{grids}{bars_svg}{sep}{line_svg}</svg>")


def decline_block_html(profile: dict | None) -> str:
    """HTML-blokk: produksjonssøyler + spesifikk modell-predikert decline rate."""
    if not profile:
        return ""
    D_pred = profile.get("D_pred")
    # Historikk-only (for nye felt uten decline-estimat ennå)
    if D_pred is None:
        if not profile.get("hist_bars"):
            return ""
        chart = production_chart_svg(profile)
        reason = profile.get("decline_src", "ingen decline-estimat")
        note = ("for tidlig for decline-forecast (ennå i ramp/platå)"
                if reason == "ennå ikke i decline"
                else "decline-rate ikke estimert (utenfor modellens datagrunnlag)")
        return f"""
      <div style='font-size:10px;font-weight:bold;color:#444;border-top:1px solid #ddd;padding-top:4px;margin-top:6px;'>
        📊 PRODUKSJONSHISTORIKK
      </div>
      <div style='font-size:9px;color:#888;margin-top:2px;'>Fase: <b>{profile.get('stage','')}</b>
        — <i>{note}</i></div>
      {chart}
      <div style='font-size:8.5px;'><span style='color:#2471a3;'>▮</span> historisk produksjon (% av peak)</div>
    """
    ptype = profile.get("type")
    is_fwd = ptype == "forward"
    is_disc = ptype == "discovery"
    stage = profile.get("stage", "")
    chart = production_chart_svg(profile)

    half_life = np.log(2) / D_pred if D_pred and D_pred > 0 else float("inf")
    hl_str = f"{half_life:.1f} år" if np.isfinite(half_life) else "—"
    rem5 = np.exp(-D_pred * 5) * 100 if D_pred else 0

    if is_disc:
        header = "📐 INDIKATIV PRODUKSJONS-TYPEKURVE (analog)"
    elif is_fwd:
        header = "📈 PRODUKSJONS-FORECAST (V5.1 + lifecycle)"
    else:
        header = "📉 PRODUKSJON: HISTORIKK vs MODELL"

    # eksplisitt decline-setning
    decline_sentence = (
        f"Modellen predikerer en decline rate på <b style='color:#e74c3c;'>{D_pred*100:.1f}%/år</b> "
        f"etter platå.")
    if not is_fwd and not is_disc and profile.get("D_actual") is not None:
        d_act = profile["D_actual"]
        decline_sentence += (f" Observert hittil: <b style='color:#9b59b6;'>{d_act*100:.1f}%/år</b>.")

    if is_disc:
        analog = profile.get("analog_field", "?")
        legend = ("<span style='color:#d7bde2;'>▮</span> indikativ forecast &nbsp; "
                  "<span style='color:#8e44ad;'>┄ decline</span>")
        decline_sentence = (
            f"Analog-felt: <b>{analog}</b>. Modellen predikerer ~"
            f"<b style='color:#8e44ad;'>{D_pred*100:.1f}%/år</b> decline for denne klassen.")
        extra = (f"<div style='font-size:8px;color:#aaa;font-style:italic;margin-top:1px;'>"
                 f"⚠ Ikke-besluttet funn — ressursvolum ukjent. Søylene viser forventet "
                 f"PRODUKSJONSFORM (% av peak), ikke volum. Halveringstid ~{hl_str}.</div>")
    elif is_fwd:
        analog = profile.get("analog")
        if analog:
            decline_sentence = (
                f"Decline-rate satt fra analog-felt <b>{analog}</b>: "
                f"<b style='color:#16a085;'>{D_pred*100:.1f}%/år</b> etter platå "
                f"<span style='color:#999;'>(ramp/platå fra lifecycle-modellen)</span>.")
        legend = ("<span style='color:#a9dfbf;'>▮</span> forecast-produksjon &nbsp; "
                  "<span style='color:#16a085;'>┄ lifecycle</span>")
        extra = (f"<div style='font-size:8.5px;color:#888;margin-top:1px;'>"
                 f"Peak ~{profile.get('peak_p50_msm3',0)*209.67:.0f} kboe/d · "
                 f"ramp {profile.get('ramp_p50',0):.0f} mnd · platå {profile.get('plateau_p50',0):.0f} mnd · "
                 f"first oil {profile.get('first_oil','?')}</div>")
    else:
        legend = ("<span style='color:#2471a3;'>▮</span> historisk &nbsp; "
                  "<span style='color:#aed6f1;'>▮</span> forecast &nbsp; "
                  "<span style='color:#e74c3c;'>┄ predikert decline</span>")
        extra = (f"<div style='font-size:8.5px;color:#888;margin-top:1px;'>"
                 f"halveringstid {hl_str} · ~{rem5:.0f}% av peak igjen etter 5 år</div>")

    return f"""
      <div style='font-size:10px;font-weight:bold;color:#444;border-top:1px solid #ddd;padding-top:4px;margin-top:6px;'>
        {header}
      </div>
      <div style='font-size:9.5px;color:#555;margin-top:2px;'>{decline_sentence}</div>
      <div style='font-size:8.5px;color:#888;margin-top:1px;'>Fase: <b>{stage}</b></div>
      {chart}
      <div style='font-size:8.5px;'>{legend}</div>
      {extra}
    """


def popup_html_field(props: dict, assay: dict | None, pred_diff: float | None,
                      top5: list | None, normpris: dict | None,
                      source_type: str = "", source_label: str = "",
                      is_forward: bool = False, forward_label: str = "",
                      baseline_pred: float | None = None,
                      decline: dict | None = None) -> str:
    """Generer HTML for popup når man klikker på et felt."""
    name      = props.get("fldName") or props.get("dscName") or "Ukjent"
    operator  = props.get("cmpLongName") or props.get("dscOwnerName") or "—"
    status    = (props.get("fldCurrentActivitySatus")
                 or props.get("dscCurrentActivityStatus") or "—")
    hctype    = props.get("fldHcType") or props.get("dscHcType") or "—"
    main_area = props.get("fldMainArea") or props.get("dscMainArea") or "—"
    disc_year = props.get("fldDiscoveryYear") or props.get("dscDiscoveryYear") or ""

    # Header
    forward_tag = ""
    if is_forward:
        forward_tag = (f"<div style='background:#FFF4E5;padding:3px 6px;border-left:3px solid #E67E22;"
                       f"font-size:10px;color:#E67E22;font-weight:bold;margin-bottom:4px;'>"
                       f"FORWARD — {forward_label}</div>")

    year_str = f" · oppdaget {disc_year}" if disc_year else ""

    html = f"""
    <div style='font-family: -apple-system, sans-serif; max-width: 340px;'>
      {forward_tag}
      <div style='font-size:13px;font-weight:bold;color:#1a1a2e;margin-bottom:2px;'>{name}</div>
      <div style='font-size:10px;color:#666;margin-bottom:4px;'>
        {operator}<br>
        <span style='color:#888;'>Status: {status} · {hctype} · {main_area}{year_str}</span>
      </div>
    """

    # Kilde-badge (★★★ / ★★ / ★)
    if source_type:
        html += source_badge(source_type, source_label)

    # Kvalitets-tabell
    if assay:
        is_estimat = source_type in ("PROXY", "FORWARD", "IMPUTED")
        header_label = "OLJEKVALITET (estimert fra proxy)" if is_estimat else "OLJEKVALITET (assay)"
        html += f"""
      <div style='font-size:10px;font-weight:bold;color:#444;border-top:1px solid #ddd;padding-top:4px;margin-top:4px;'>
        ⛽ {header_label}
      </div>
      <table style='font-size:10px;width:100%;border-collapse:collapse;margin-top:2px;'>
        <tr><td style='color:#666;'>API gravity</td><td style='text-align:right;font-family:monospace;'>{assay['api']:.1f}°</td></tr>
        <tr><td style='color:#666;'>Svovel</td><td style='text-align:right;font-family:monospace;'>{assay['sulfur']:.2f}%</td></tr>
        <tr><td style='color:#666;'>Vacuum resid</td><td style='text-align:right;font-family:monospace;'>{assay['vac_res']:.1f}%</td></tr>
        <tr><td style='color:#666;'>Conradson carbon</td><td style='text-align:right;font-family:monospace;'>{assay['ccr']:.2f}%</td></tr>
        <tr><td style='color:#666;'>Middle distillate</td><td style='text-align:right;font-family:monospace;'>{assay['mid_dist']:.1f}%</td></tr>
      </table>
        """
    else:
        html += ("<div style='font-size:10px;color:#999;font-style:italic;margin-top:4px;'>"
                 "Ingen assay-data — modell ikke kjørt.</div>")

    # Modellprediksjon — m/ math-dekomposisjon: NCS-baseline + kvalitets-justering
    if pred_diff is not None:
        color = "#27AE60" if pred_diff > 0 else "#C0392B" if pred_diff < -0.5 else "#7F8C8D"
        sign  = "+" if pred_diff > 0 else ""
        is_estimat = source_type in ("PROXY", "FORWARD", "IMPUTED")
        header_lbl = "MODELL-ESTIMAT (basert på proxy)" if is_estimat else "MODELLPREDIKERT DIFFERENSIAL"
        disclaimer = ""
        if is_estimat:
            disclaimer = ("<div style='font-size:9px;color:#999;font-style:italic;margin-top:2px;'>"
                          "Usikkerhet: ±2–3 USD/bbl. Estimatet hviler på proxy-assay.</div>")

        html += f"""
      <div style='font-size:10px;font-weight:bold;color:#444;border-top:1px solid #ddd;padding-top:4px;margin-top:6px;'>
        📊 {header_lbl}
      </div>
      <div style='font-size:18px;font-weight:bold;color:{color};margin-top:2px;line-height:1.1;'>
        {sign}{pred_diff:.2f} <span style='font-size:10px;color:#888;font-weight:normal;'>USD/bbl vs. Dated Brent</span>
      </div>
      {disclaimer}
        """

        # Math-dekomposisjon: NCS-baseline + kvalitets-justering = total
        if baseline_pred is not None and top5 is not None:
            quality_delta = sum(c["impact"] for c in top5)
            bp_sign = "+" if baseline_pred > 0 else ""
            qd_sign = "+" if quality_delta > 0 else ""
            qd_color = "#1D6A39" if quality_delta > 0 else "#C0392B" if quality_delta < 0 else "#888"
            html += f"""
      <div style='font-size:9px;color:#888;margin-top:3px;font-family:monospace;
                   background:#FAFAFA;padding:3px 5px;border-radius:3px;'>
        NCS-baseline:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{bp_sign}{baseline_pred:.2f}</b><br>
        Kvalitets-justering: <b style='color:{qd_color};'>{qd_sign}{quality_delta:.2f}</b><br>
        <span style='color:#aaa;'>─────────────────────</span><br>
        <b style='color:{color};'>= Total: {sign}{pred_diff:.2f}</b>
      </div>
        """

    # Normpris (siste 4Q)
    if normpris:
        np_color = "#27AE60" if normpris["value"] > 0 else "#C0392B" if normpris["value"] < -0.5 else "#7F8C8D"
        np_sign  = "+" if normpris["value"] > 0 else ""
        html += f"""
      <div style='font-size:10px;color:#666;margin-top:3px;'>
        Normpris (4Q snitt → {normpris['qstr_end']}):
        <span style='font-weight:bold;color:{np_color};font-family:monospace;'>
          {np_sign}{normpris['value']:.2f} USD/bbl
        </span>
      </div>
        """

    # Top 5 kvalitets-drivere (vs. NCS-snitt)
    if top5:
        total_quality_delta = sum(c["impact"] for c in top5)
        html += """
      <div style='font-size:10px;font-weight:bold;color:#444;border-top:1px solid #ddd;padding-top:4px;margin-top:6px;'>
        🔍 TOP 5 KVALITETSDRIVERE
        <span style='font-weight:normal;color:#888;font-size:9px;'>(USD/bbl vs. NCS-snitt)</span>
      </div>
      <table style='font-size:9.5px;width:100%;border-collapse:collapse;margin-top:3px;'>
        """
        max_abs = max(abs(c["impact"]) for c in top5) or 1.0
        for c in top5:
            imp       = c["impact"]
            bar_pct   = abs(imp) / max_abs * 100
            bar_color = "#1D6A39" if imp > 0.05 else "#C0392B" if imp < -0.05 else "#888888"
            sign      = "+" if imp > 0 else ""
            html += f"""
        <tr>
          <td style='color:#333;padding:1px 4px 1px 0;font-weight:600;'>{c['label']}</td>
          <td style='text-align:right;font-family:monospace;color:{bar_color};font-weight:bold;white-space:nowrap;'>
            {sign}{imp:.2f}
          </td>
        </tr>
        <tr>
          <td colspan='2' style='padding:0 0 1px 0;'>
            <span style='font-size:8.5px;color:#888;font-style:italic;'>
              {c['field_v']} <span style='color:#aaa;'>(snitt {c['base_v']}) · {c['explain']}</span>
            </span>
          </td>
        </tr>
        <tr><td colspan='2' style='padding-bottom:3px;'>
          <div style='background:#eee;height:3px;border-radius:2px;'>
            <div style='background:{bar_color};width:{bar_pct:.0f}%;height:100%;border-radius:2px;'></div>
          </div>
        </td></tr>
            """
        html += "</table>"

        # Sumlinje for kvalitets-utslag
        sum_color = "#1D6A39" if total_quality_delta > 0 else "#C0392B" if total_quality_delta < 0 else "#666"
        sum_sign  = "+" if total_quality_delta > 0 else ""
        html += f"""
      <div style='font-size:9.5px;color:#444;margin-top:2px;border-top:1px dashed #ddd;padding-top:2px;'>
        Sum top 5 kvalitets-utslag:
        <span style='font-weight:bold;color:{sum_color};font-family:monospace;'>
          {sum_sign}{total_quality_delta:.2f} USD/bbl
        </span>
      </div>
        """

    # Estimert decline-kurve (V5.1-modell)
    html += decline_block_html(decline)

    html += """
      <div style='font-size:8.5px;color:#999;font-style:italic;border-top:1px solid #eee;padding-top:3px;margin-top:6px;'>
        Pris: Brent-linked OLS · Decline: V5.1 (nested CV R²=0.66) · Kvalitet: Script 63-fallback
      </div>
    </div>
    """
    return html


# ────────────────────────────────────────────────────────────────────────────
# FARGE-SKALA (heatmap for kvalitets-differensial)
# ────────────────────────────────────────────────────────────────────────────

def make_colormap() -> LinearColormap:
    """Diverging fargeskala: rød (rabatt) → grå (null) → grønn (premium)."""
    cm = LinearColormap(
        colors=["#C0392B", "#E67E22", "#F4F4F4", "#52BE80", "#1D6A39"],
        vmin=-5, vmax=5,
        caption="Modellpredikert differensial vs. Dated Brent (USD/bbl)",
    )
    return cm


def style_field(pred_diff: float | None, status: str, cm: LinearColormap,
                 source_type: str = "DIRECT") -> dict:
    """
    Style for et felt-polygon. Kantstil koder kvalitetsnivået på prediksjonen:
       DIRECT  — Solid svart kant, full opacity   (egen assay)
       BLEND   — Solid mørkblå kant, full opacity (faktisk eksport-blend)
       PROXY   — Stiplet grå kant, redusert opac. (geografisk estimat)
    """
    if pred_diff is None:
        return {
            "fillColor":   "#cccccc",
            "color":       "#888888",
            "weight":      0.6,
            "fillOpacity": 0.35,
        }

    fill_color = cm(max(-5, min(5, pred_diff)))

    border_cfg = {
        "DIRECT": dict(color="#222222", weight=1.1, dash=None,   fillOp=0.82),
        "BLEND":  dict(color="#2874A6", weight=0.9, dash=None,   fillOp=0.75),
        "PROXY":  dict(color="#7F7F7F", weight=0.8, dash="4, 3", fillOp=0.55),
    }
    cfg = border_cfg.get(source_type, border_cfg["DIRECT"])

    style = {
        "fillColor":   fill_color,
        "color":       cfg["color"],
        "weight":      cfg["weight"],
        "fillOpacity": cfg["fillOp"],
    }
    if cfg["dash"]:
        style["dashArray"] = cfg["dash"]

    # Nedlagte felt blekes
    if status and "shut" in status.lower():
        style["fillOpacity"] = 0.25
        style["color"]       = "#999999"

    return style


def style_forward(props: dict) -> dict:
    """Stiplet oransje polygon for forward-felt (approved for production)."""
    return {
        "fillColor":   "#F39C12",
        "color":       "#E67E22",
        "weight":      1.5,
        "dashArray":   "5, 5",
        "fillOpacity": 0.45,
    }


def style_discovery(pred_diff: float | None, hctype: str = "") -> dict:
    """
    Stiplet lilla polygon for funn under utvikling (NAV-upside).
    Bruker fyllfarge basert på predikert kvalitet — men med lavere opacity
    og en annen kantfarge for å signalisere usikkerheten.
    """
    # Lilla-toner — tydelig forskjellig fra produserende felt og forward-felt
    fill = "#9B59B6"
    if pred_diff is not None:
        if pred_diff > 1.0:
            fill = "#52BE80"   # grønnaktig for premium-funn
        elif pred_diff < -1.0:
            fill = "#E74C3C"   # rødaktig for discount-funn
    return {
        "fillColor":   fill,
        "color":       "#8E44AD",
        "weight":      1.2,
        "dashArray":   "3, 4",
        "fillOpacity": 0.40,
    }


def style_licence(_: dict) -> dict:
    return {
        "fillColor":   "#3498DB",
        "color":       "#2874A6",
        "weight":      0.4,
        "fillOpacity": 0.08,
    }


def style_block(_: dict) -> dict:
    return {
        "fillColor":   "#ffffff",
        "color":       "#aaaaaa",
        "weight":      0.3,
        "fillOpacity": 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  SCRIPT 49: Interaktivt NCS-kart")
    print("=" * 70)

    print("\n[1] Laster data...")
    coefs, features, metrics = load_model()
    print(f"  Modell: R²={metrics['r2']:.3f}, OOT R²={metrics['r2_oot']:.3f}, RMSE={metrics['rmse']:.2f}")

    assay_db = load_assay_database()
    print(f"  Assay-database: {len(assay_db)} grades")

    normpris = load_normpris_latest()
    print(f"  Normpris (siste 4Q): {len(normpris)} felt")

    imputed_quality = load_imputed_quality()
    print(f"  Script 63 kvalitets-fallback: {len(imputed_quality)} felt")

    prod_profiles = load_production_profiles()
    n_prod_prof = sum(1 for p in prod_profiles.values() if p.get("type") == "producing")
    n_fwd_prof = sum(1 for p in prod_profiles.values() if p.get("type") == "forward")
    print(f"  Produksjonsprofiler: {n_prod_prof} produserende + {n_fwd_prof} forward")

    # field → V5.1 D_pred (for indikative funn-typekurver via analog)
    d_pred_by_field = {n: p["D_pred"] for n, p in prod_profiles.items()
                       if p.get("type") == "producing" and p.get("D_pred")}

    fields_fc      = load_geojson("fields")
    discoveries_fc = load_geojson("discoveries")
    licences_fc    = load_geojson("licences")
    blocks_fc      = load_geojson("blocks")
    print(f"  Felt: {len(fields_fc['features'])}, "
          f"Funn: {len(discoveries_fc['features'])}, "
          f"Lisenser: {len(licences_fc['features'])}, "
          f"Blokker: {len(blocks_fc['features'])}")

    # ── Forhåndsberegn prediksjon per felt ───────────────────────────────────
    print("\n[2] Beregner predikert differensial per felt...")

    # Bygg sentroid-indeks for nærmeste-nabo-søk
    centroids = build_centroid_index(fields_fc)

    # NCS-baseline (gjennomsnittlig NCS-felt — referansen for kvalitets-utslag)
    NCS_GRADES = list(set(DIRECT_ASSAY.values()))   # 21 NCS-assays
    ncs_baseline = compute_ncs_baseline(assay_db, NCS_GRADES)
    print(f"  NCS-baseline (snitt 21 grades): API {ncs_baseline['api']:.1f}°, "
          f"S {ncs_baseline['sulfur']:.2f}%, CCR {ncs_baseline['ccr']:.2f}%, "
          f"mid-dist {ncs_baseline['mid_dist']:.1f}%")

    # Forhåndsberegn NCS-baseline-prediksjonen (referanse-differensialet til
    # et 'snitt-NCS-felt' uten kvalitets-avvik). Dette lar oss vise math:
    #   pred(felt) = pred(NCS-baseline) + sum(quality_deltas)
    baseline_feat = build_field_features(
        ncs_baseline["api"], ncs_baseline["sulfur"], ncs_baseline["vac_res"],
        ncs_baseline["ccr"], ncs_baseline["mid_dist"], ncs_baseline["v_ni"]
    )
    baseline_pred = predict(baseline_feat, coefs, features)
    print(f"  NCS-baseline prediksjon: {baseline_pred:+.2f} USD/bbl vs. Brent "
          f"(dette er referansen kvalitets-utslagene måles mot)")

    # Felt som har DIRECT eller BLEND mapping (kan brukes som proxy-kandidat)
    proxy_candidates = set(DIRECT_ASSAY) | set(BLEND_PROXY)

    # Fall-back-snitt fra NCS-grades brukes hvis enkeltkolonner er NaN
    # (noen grades har f.eks. NaN i CCR — bruk NCS-snittet for å unngå NaN i output)
    def safe_value(r, col: str, fallback: float) -> float:
        v = r.get(col)
        if v is None or pd.isna(v):
            return float(fallback)
        return float(v)

    def get_assay_values(grade_name: str) -> dict | None:
        row = assay_db[assay_db["grade"] == grade_name]
        if len(row) == 0:
            return None
        r = row.iloc[0]

        api      = safe_value(r, "api_gravity",           35.0)
        sulfur   = safe_value(r, "sulfur_pct",             0.50)
        vac_res  = safe_value(r, "vacuum_resid_pct",      15.0)
        ccr      = safe_value(r, "ccr_wt_pct",             2.0)
        mid_dist = safe_value(r, "middle_distillate_pct", 40.0)
        log_vni  = safe_value(r, "log_v_ni",              float(np.log1p(5.0)))
        v_ni     = float(np.expm1(log_vni))

        return dict(api=api, sulfur=sulfur, vac_res=vac_res,
                    ccr=ccr, mid_dist=mid_dist, v_ni=v_ni)

    field_predictions = {}
    n_direct = n_blend = n_proxy = n_none = 0

    for f in fields_fc["features"]:
        name = (f.get("properties", {}).get("fldName") or "").upper()
        if not name:
            continue

        source_type = None
        source_label = None
        assay_grade = None

        # Nivå 1: DIRECT (egen assay)
        if name in DIRECT_ASSAY:
            assay_grade = DIRECT_ASSAY[name]
            source_type = "DIRECT"
            source_label = "Equinor lab-assay (direkte)"
            n_direct += 1

        # Nivå 2: BLEND (eksportert via kjent blend)
        elif name in BLEND_PROXY:
            assay_grade, blend_name = BLEND_PROXY[name]
            source_type = "BLEND"
            source_label = f"Blend-proxy: {blend_name}"
            n_blend += 1

        # Nivå 3: SCRIPT 63 IMPUTERING (område-median + Sodir-DST API)
        #         Erstatter tidligere geografisk-proxy-fallback.
        assay = None
        if assay_grade is None and name in imputed_quality:
            assay = imputed_quality[name]
            tier = assay.get("tier", "")
            source_type = "IMPUTED"
            tier_lbl = {"3_MEDIAN+DST": "median + Sodir-DST",
                        "2_BLEND": "blend-assay", "1_STANDALONE": "standalone"}.get(tier, tier)
            source_label = f"Script 63 ({tier_lbl})"
            n_proxy += 1

        # Nivå 4: GEOGRAFISK PROXY (siste utvei — kun hvis ikke i Script 63)
        if assay_grade is None and assay is None:
            sodir_candidates = {n for n in proxy_candidates if n in centroids}
            nearest = find_nearest_proxy(name, centroids, sodir_candidates)
            if nearest:
                proxy_name, dist_km = nearest
                if proxy_name in DIRECT_ASSAY:
                    assay_grade = DIRECT_ASSAY[proxy_name]
                else:
                    assay_grade = BLEND_PROXY[proxy_name][0]
                source_type = "PROXY"
                source_label = f"Geografisk proxy: {proxy_name.title()} ({dist_km:.0f} km)"
                n_proxy += 1

        # Hent assay-verdier fra grade hvis vi gikk via DIRECT/BLEND/PROXY
        if assay is None:
            if assay_grade is None:
                n_none += 1
                continue
            assay = get_assay_values(assay_grade)
            if assay is None:
                n_none += 1
                continue

        # is_fpso-flagg basert på Sodir-feltnavn (NY i v3)
        is_fpso = 1 if name in SODIR_FPSO_FIELDS else 0

        feat = build_field_features(
            assay["api"], assay["sulfur"], assay["vac_res"],
            assay["ccr"], assay["mid_dist"], assay["v_ni"],
            is_fpso=is_fpso,
        )
        pred = predict(feat, coefs, features)
        top5 = quality_impact_decomposition(assay, ncs_baseline, coefs)
        field_predictions[name] = {
            "pred":          pred,
            "top5":          top5,
            "assay":         assay,
            "source_type":   source_type,
            "source_label":  source_label,
            "proxy_grade":   assay_grade,
            "is_fpso":       is_fpso,
            "profile":       prod_profiles.get(name),
        }

    n_decline = sum(1 for n in field_predictions if prod_profiles.get(n))
    print(f"  ★★★ DIRECT (egen assay):       {n_direct:>3}")
    print(f"  ★★  BLEND-PROXY:                {n_blend:>3}")
    print(f"  ★★  SCRIPT 63 / proxy:          {n_proxy:>3}")
    print(f"  —   uten data:                  {n_none:>3}")
    print(f"  TOTAL m/ modellprediksjon:    {n_direct + n_blend + n_proxy:>3} / {len(fields_fc['features'])}")
    print(f"  📈  m/ produksjonsprofil:       {n_decline:>3}")

    # ── Bygg Folium-kart ─────────────────────────────────────────────────────
    print("\n[3] Bygger kart...")
    # Senter omtrent på Stavanger/Nordsjøen
    fmap = folium.Map(
        location=[61.0, 4.5],
        zoom_start=6,
        tiles=None,
        prefer_canvas=True,
        control_scale=True,
    )
    # Lyst kart som standard. Mørkt kart legges med show=False så det kun er
    # et valg i lag-kontrollen og ikke rendres oppå det lyse ved innlasting.
    folium.TileLayer(
        "CartoDB positron", name="Lyst kart (Positron)", control=True
    ).add_to(fmap)
    folium.TileLayer(
        "CartoDB dark_matter", name="Mørkt kart (Dark Matter)", control=True, show=False
    ).add_to(fmap)

    # Fargeskala brukes til felt-farger. Branca-legenden (Leaflet-kontroll,
    # låst til topright) erstattes av en egen HTML-legend nederst til høyre
    # lenger ned, så den blir større og plassert der vi vil ha den.
    cm = make_colormap()

    # NB: Leaflet legger sist-tilføyde lag ØVERST i z-orden, så vi MÅ legge
    # bakgrunnslag (blokker, lisenser) først og felt-laget til slutt.
    # Bakgrunnslagene gjøres også non-interactive (ingen klikk-fangst).

    # ── BAKGRUNNSLAG 1: NCS-blokker (nederst, ekte non-interactive) ────────
    # Eget Leaflet-pane med pointer_events=False og lav z-index. Da fanger
    # blokk-rutene ALDRI klikk, og de ligger alltid under feltene, uansett
    # hvilken rekkefølge lagene hukes av/på. Feltene forblir klikkbare.
    # Huket av som standard når lenken åpnes (vises som referansegrid).
    CustomPane("ncs_bg", z_index=350, pointer_events=False).add_to(fmap)
    blocks_layer = folium.FeatureGroup(name="🗺  NCS-blokkruter", show=True)
    folium.GeoJson(
        blocks_fc,
        style_function=style_block,
        pane="ncs_bg",          # klikk passerer rett gjennom til feltene
        zoom_on_click=False,
    ).add_to(blocks_layer)
    blocks_layer.add_to(fmap)

    # ── BAKGRUNNSLAG 2: Lisensblokker (ikke-interaktivt) ────────────────────
    licences_layer = folium.FeatureGroup(name="📑  Produksjonslisenser", show=False)
    folium.GeoJson(
        licences_fc,
        style_function=style_licence,
        # Tooltip ved hover (fanger ikke klikk på samme måte)
        tooltip=folium.GeoJsonTooltip(
            fields=["prlName", "cmpLongName"],
            aliases=["Lisens:", "Operatør:"],
            sticky=False,
        ),
        zoom_on_click=False,
    ).add_to(licences_layer)
    licences_layer.add_to(fmap)

    # ── HOVEDLAG 1: Forward-funn (kun store utbygginger) ────────────────────
    forward_layer = folium.FeatureGroup(name="🟠  Forward-felt (planlagte utbygginger)", show=True)
    n_forward = 0
    for f in discoveries_fc["features"]:
        props = f.get("properties", {})
        status = (props.get("dscCurrentActivityStatus") or "").lower()
        if "approved" not in status and "production" not in status:
            continue
        dsc_name = (props.get("dscName") or "").upper()
        fld_name = (props.get("fldName") or "").upper()

        # Filter: kun whitelist-felt
        in_whitelist = (
            fld_name in MAJOR_FORWARD_FIELDS
            or any(key in dsc_name for key in MAJOR_FORWARD_FIELDS if len(key) > 3)
        )
        if not in_whitelist:
            continue

        n_forward += 1

        # Forward kvalitet hvis vi har et eksplisitt estimat
        fwd_q = FORWARD_FIELDS_QUALITY.get(fld_name)
        if fwd_q is None:
            for prefix, q in FORWARD_FIELDS_QUALITY.items():
                if prefix in fld_name or prefix in dsc_name:
                    fwd_q = q
                    break

        if fwd_q:
            feat = build_field_features(
                fwd_q["api"], fwd_q["sulfur"], fwd_q["vac_res"],
                fwd_q["ccr"], fwd_q["mid_dist"], fwd_q["v_ni"]
            )
            pred = predict(feat, coefs, features)
            assay_dict = dict(api=fwd_q["api"], sulfur=fwd_q["sulfur"],
                              vac_res=fwd_q["vac_res"], ccr=fwd_q["ccr"],
                              mid_dist=fwd_q["mid_dist"],
                              v_ni=fwd_q["v_ni"])
            top5 = quality_impact_decomposition(assay_dict, ncs_baseline, coefs)
            popup_html = popup_html_field(
                props, assay_dict, pred, top5, None,
                source_type="FORWARD",
                source_label=fwd_q.get("proxy", "estimat"),
                is_forward=True,
                forward_label=fwd_q.get("label", fld_name.title()),
                baseline_pred=baseline_pred,
                decline=prod_profiles.get(fld_name),
            )
        else:
            popup_html = popup_html_field(
                props, None, None, None, None,
                is_forward=True, forward_label="Under utbygging",
                decline=prod_profiles.get(fld_name),
            )

        folium.GeoJson(
            f,
            style_function=style_forward,
            highlight_function=lambda x: {"weight": 3, "color": "#D35400"},
            tooltip=folium.Tooltip(props.get("dscName", "?") + " (forward)", sticky=True),
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(forward_layer)
    forward_layer.add_to(fmap)
    print(f"  Forward-felt lagt til: {n_forward}")

    # ── HOVEDLAG 1B: Funn under utvikling (NAV upside) ──────────────────────
    # Bruker geografisk proxy for kvalitet — nærmeste felt med assay.
    # Alle popups markeres som ESTIMAT med klar usikkerhets-disclaimer.
    discovery_layer = folium.FeatureGroup(
        name="🔮  Funn under utvikling", show=False
    )
    n_disc = 0
    # Bygg sentroid-indeks også for funn (vi har den allerede for felt)
    field_centroid_set = {n for n in centroids if n in proxy_candidates}

    for f in discoveries_fc["features"]:
        props = f.get("properties", {})
        status = props.get("dscCurrentActivityStatus") or ""
        dsc_name = (props.get("dscName") or "").upper()
        fld_name = (props.get("fldName") or "").upper()
        hctype   = (props.get("dscHcType") or "").upper()

        # Filter 1: kun NAV-relevante statuser (ikke 'Production unlikely' eller 'Included in other')
        if status not in DEVELOPMENT_CANDIDATE_STATUSES:
            continue

        # Filter 2: ekskluder rene gass-funn (modellen er for crude oil)
        if hctype in {"GAS", ""}:
            continue

        # Filter 3: skip funn som allerede vises i 'Forward'-laget
        if any(key in fld_name or key in dsc_name for key in MAJOR_FORWARD_FIELDS if len(key) > 3):
            continue

        n_disc += 1

        # Geografisk proxy: finn sentroid for funnet og nærmeste felt med assay
        disc_cen = polygon_centroid(f.get("geometry", {}))
        proxy_label = None
        assay = None
        pred = None
        top5 = None

        disc_profile = None
        if disc_cen:
            # Lag en temp dict-entry for å bruke find_nearest_proxy
            disc_key = f"__DISC__{dsc_name}"
            temp_centroids = {**centroids, disc_key: (disc_cen[0], disc_cen[1], props.get("dscMainArea", ""))}
            nearest = find_nearest_proxy(disc_key, temp_centroids, field_centroid_set)
            if nearest:
                proxy_name, dist_km = nearest
                proxy_grade = DIRECT_ASSAY.get(proxy_name) or BLEND_PROXY.get(proxy_name, (None,))[0]
                if proxy_grade:
                    assay = get_assay_values(proxy_grade)
                    if assay:
                        feat = build_field_features(
                            assay["api"], assay["sulfur"], assay["vac_res"],
                            assay["ccr"], assay["mid_dist"], assay["v_ni"]
                        )
                        pred = predict(feat, coefs, features)
                        top5 = quality_impact_decomposition(assay, ncs_baseline, coefs)
                        proxy_label = f"Geografisk proxy: {proxy_name.title()} ({dist_km:.0f} km unna)"

                # Indikativ type-kurve: bruk analog-feltets V5.1 decline-rate
                D_analog = d_pred_by_field.get(proxy_name)
                if D_analog is None:
                    # nærmeste analog manglet D_pred → bruk klasse-typisk subsea-tieback
                    D_analog = 0.22
                    analog_lbl = f"{proxy_name.title()} (klasse-typisk decline)"
                else:
                    analog_lbl = proxy_name.title()
                disc_profile = indicative_discovery_profile(D_analog, analog_lbl)

        if proxy_label is None:
            proxy_label = "Ingen proxy funnet"

        # Lag NAV-fokusert label
        disc_year = props.get("dscDiscoveryYear") or "?"
        operator  = props.get("cmpLongName") or "?"
        nav_label = (
            f"{dsc_name.title()} · oppdaget {disc_year} · "
            f"{operator[:25]} · status: {status}"
        )

        popup_html = popup_html_field(
            props, assay, pred, top5, None,
            source_type="PROXY",
            source_label=proxy_label,
            is_forward=True,
            forward_label=nav_label,
            baseline_pred=baseline_pred,
            decline=disc_profile,
        )

        folium.GeoJson(
            f,
            style_function=lambda x, p=pred: style_discovery(p),
            highlight_function=lambda x: {"weight": 2.5, "color": "#5B2C6F"},
            tooltip=folium.Tooltip(
                f"{props.get('dscName', '?')} (under utvikling)",
                sticky=True,
            ),
            popup=folium.Popup(popup_html, max_width=380),
        ).add_to(discovery_layer)
    discovery_layer.add_to(fmap)
    print(f"  Funn under utvikling lagt til: {n_disc}")

    # ── HOVEDLAG 2: Felt-polygoner (kvalitets-heatmap) — TOPPLAGET ──────────
    # Legges sist så det alltid er øverst i z-orden og fanger alle klikk.
    fields_layer = folium.FeatureGroup(name="🛢️  Felt — kvalitets-heatmap", show=True)
    for f in fields_fc["features"]:
        props = f.get("properties", {})
        name  = (props.get("fldName") or "").upper()
        pred_info = field_predictions.get(name)
        status = props.get("fldCurrentActivitySatus", "")

        if pred_info:
            style = style_field(pred_info["pred"], status, cm, pred_info["source_type"])
            np_data = normpris.get(name)
            popup_html = popup_html_field(
                props, pred_info["assay"], pred_info["pred"],
                pred_info["top5"], np_data,
                source_type=pred_info["source_type"],
                source_label=pred_info["source_label"],
                baseline_pred=baseline_pred,
                decline=pred_info.get("profile"),
            )
        else:
            style = style_field(None, status, cm)
            popup_html = popup_html_field(props, None, None, None, normpris.get(name),
                                          decline=prod_profiles.get(name))

        folium.GeoJson(
            f,
            style_function=lambda x, s=style: s,
            highlight_function=lambda x: {"weight": 2.5, "color": "#000"},
            tooltip=folium.Tooltip(props.get("fldName", "?"), sticky=True),
            popup=folium.Popup(popup_html, max_width=380),
        ).add_to(fields_layer)
    fields_layer.add_to(fmap)

    # ── Lag-kontroll og logo/tittel ──────────────────────────────────────────
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    # Tittel-overlay + datakilde-legend
    title_html = """
    <div style='position: fixed; top: 12px; left: 60px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.18);
                font-family: -apple-system, sans-serif; max-width: 400px;'>
      <div style='font-size: 14px; font-weight: bold; color: #1a1a2e;'>
        NCS Crude Oil Quality Map
      </div>
      <div style='font-size: 10px; color: #666; margin-top: 3px; margin-bottom: 8px;'>
        Klikk på et felt for assay, modellpredikert Brent-differensial og top 5 forklaringsvariabler.
        Farge = predikert differensial (rød = rabatt, grønn = premium).
      </div>

      <div style='font-size: 9.5px; font-weight: bold; color: #444;
                  border-top: 1px solid #eee; padding-top: 5px; margin-bottom: 3px;'>
        Datakvalitet (kantstil på polygon):
      </div>

      <div style='font-size: 9px; color: #444; line-height: 1.5;'>
        <span style='color:#27AE60;font-weight:bold;'>★★★ DIRECT</span> &mdash;
        Egen Equinor lab-assay <span style='color:#888;'>(solid svart kant)</span><br>
        <span style='color:#2980B9;font-weight:bold;'>★★ BLEND-PROXY</span> &mdash;
        Eksportert via kjent blend, f.eks. Ekofisk Blend <span style='color:#888;'>(solid blå kant)</span><br>
        <span style='color:#E67E22;font-weight:bold;'>★ GEOGRAFISK PROXY</span> &mdash;
        Estimert fra nærmeste nabofelt <span style='color:#888;'>(stiplet grå kant)</span><br>
        <span style='color:#E67E22;font-weight:bold;'>◆ FORWARD-FELT</span> &mdash;
        Approved for production (Yggdrasil etc.) <span style='color:#888;'>(oransje stiplet)</span><br>
        <span style='color:#8E44AD;font-weight:bold;'>◇ FUNN UNDER UTVIKLING</span> &mdash;
        NAV-upside, geografisk proxy <span style='color:#888;'>(lilla stiplet, toggle av/på)</span>
      </div>

      <div style='font-size: 8.5px; color: #999; margin-top: 6px; font-style: italic;
                  border-top: 1px solid #eee; padding-top: 4px;'>
        Kilder: Sodir FactMaps (geometri), Equinor lab-assays (kvalitet),
        Petroleumsprisrådet (normpris).<br>
        Modell: Brent-linked OLS, 32 grades, OOT R² = 0.34, RMSE = 2.95 USD/bbl.
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    # Heatmap-legend (kvalitets-differensial) — nederst til høyre, større.
    # Erstatter branca sin lille topright-kontroll.
    legend_html = """
    <div style='position: fixed; bottom: 22px; right: 14px; z-index: 9998;
                background: rgba(255,255,255,0.94); padding: 11px 14px;
                border-radius: 6px; box-shadow: 0 2px 10px rgba(0,0,0,0.22);
                font-family: -apple-system, sans-serif; width: 320px;'>
      <div style='font-size: 11.5px; font-weight: bold; color: #333; margin-bottom: 7px;'>
        Modellpredikert differensial vs. Dated Brent
      </div>
      <div style='height: 18px; border-radius: 3px; border: 1px solid #ddd;
                  background: linear-gradient(to right,
                  #C0392B 0%, #E67E22 25%, #F4F4F4 50%, #52BE80 75%, #1D6A39 100%);'></div>
      <div style='display: flex; justify-content: space-between;
                  font-size: 10.5px; color: #555; margin-top: 4px;'>
        <span>&minus;5 rabatt</span><span>0</span><span>+5 premium</span>
      </div>
      <div style='font-size: 9.5px; color: #999; margin-top: 3px;'>USD/bbl</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # ── Lagre ────────────────────────────────────────────────────────────────
    fmap.save(str(OUT_HTML))
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"\n  ✓ Lagret kart: {OUT_HTML} ({size_mb:.1f} MB)")
    print(f"\n  Åpne i nettleser: file://{OUT_HTML}")


if __name__ == "__main__":
    main()
