"""
Script 63: NCS Field → Crude Quality Imputation
====================================================================

Bygger en best-estimat kvalitetsvektor (API, svovel, vacuum resid, CCR,
metaller, middeldestillat) for HVERT NCS oljefelt, med eksplisitt
provenance-tier per felt.

HIERARKI (høyest → lavest):
  Tier 1  STANDALONE_ASSAY  — feltet har egen publisert assay
  Tier 2  BLEND_ASSAY       — feltet selges inn i navngitt blend m/ assay
                              (KORREKT for pris — det er salgsvaren)
  Tier 3  ANALOG            — ingen assay; lab-variabler fra nærmeste-API-felt
                              i samme hovedområde, API fra Sodir DST
  Tier 4  GEOGRAPHY         — siste utvei: hovedområde-median

VIKTIG NYANSE (pris vs decline):
  For PRIS teller salgsvaren. Felt i en navngitt blend prises som blenden
  uansett reservoar-API. Derfor er Tier 2 (blend-assay) korrekt, ikke
  reservoar-API. Sodir DST brukes kun for standalone-felt uten assay (Tier 3).

  Sodir wellbore gir KUN API/GOR/temp — ikke svovel/metaller/kutt.
  Lab-variablene må komme fra analog-felt (Tier 3) eller geografi (Tier 4).

VALIDERING:
  Hold ut hvert felt med ekte assay, imputer lab-variablene via
  nærmeste-API-samme-område-analog, mål feil per variabel.

Output:
  data/processed/63_ncs_field_quality.csv
  data/processed/63_imputation_validation.csv
  data/processed/63_field_quality_imputation.png
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DQ = ROOT / "analyses" / "decline_quality" / "data"
PROC = ROOT / "data" / "processed"
GEO = ROOT / "data" / "raw" / "sodir_geo" / "fields.geojson"

LAB_VARS = ["sulfur_pct", "vacuum_resid_pct", "ccr_pct",
            "vanadium_ppm", "nickel_ppm", "middle_distillate_pct"]

lines = []
def log(m=""):
    print(m); lines.append(m)

log("=" * 70)
log("SCRIPT 63: NCS FIELD → CRUDE QUALITY IMPUTATION")
log("=" * 70)

# ────────────────────────────────────────────────────────────────────────────
# 1. LAST DATA
# ────────────────────────────────────────────────────────────────────────────
assays = pd.read_csv(PROC / "unified_crude_assays.csv")
fs = pd.read_csv(DQ / "field_summary.csv")              # field → grade (blend)
dst = pd.read_csv(DQ / "fluid_enrichment" / "dst_derived_fluid.csv")
master = pd.read_csv(DQ / "master_fluid_library_v51.csv")

# main_area + grade per field fra GeoJSON
with open(GEO) as f:
    gj = json.load(f)
field_area, field_operator = {}, {}
for feat in gj["features"]:
    p = feat["properties"]
    name = (p.get("fldName") or "").strip().upper()
    if name:
        field_area[name] = (p.get("fldMainArea") or "").strip()
        field_operator[name] = (p.get("cmpLongName") or "").strip()

# assay-grade → kvalitetsvektor
assay_q = assays.set_index("grade")[["api_gravity"] + LAB_VARS]

log(f"\nAssay-bibliotek: {len(assays)} grades")
log(f"Field→blend-mapping: {fs.grade.notna().sum()} felt")
log(f"DST API: {len(dst)} felt")

# ────────────────────────────────────────────────────────────────────────────
# 2. UNIVERS: alle NCS oljefelt (fra master library, ekskl. gass)
# ────────────────────────────────────────────────────────────────────────────
universe = master[~master.is_gas_field.fillna(False)].copy()
universe["field_u"] = universe.field.str.upper().str.strip()
universe["main_area"] = universe.field_u.map(field_area).fillna("Unknown")
universe["operator"] = universe.field_u.map(field_operator).fillna("")

# field → blend grade
fs["field_u"] = fs.field.str.upper().str.strip()
field_to_grade = fs.set_index("field_u")["grade"].to_dict()
field_is_direct = fs.set_index("field_u")["is_direct_assay"].to_dict()

# DST API per field
dst["field_u"] = dst.field.str.upper().str.strip()
dst_api = dst.set_index("field_u")["api_median"].to_dict()

log(f"\nUnivers (NCS oljefelt): {len(universe)}")

# ────────────────────────────────────────────────────────────────────────────
# 3. IMPUTERINGS-FUNKSJON
# ────────────────────────────────────────────────────────────────────────────
# Bygg sett av felt som HAR en gyldig blend/standalone assay (for analog-pool)
def grade_has_assay(grade):
    return isinstance(grade, str) and grade in assay_q.index

def assign_quality(field_u, api_for_match, main_area, analog_pool, exclude=None,
                   force_analog=False):
    """
    Returnerer (quality_dict, tier, provenance).
    analog_pool: DataFrame med felt som har kjent assay [field_u, api, main_area, + LAB_VARS]
    force_analog: hopp over Tier 1/2 — brukes i validering for å teste analog-metoden.
    """
    grade = field_to_grade.get(field_u)
    is_direct = field_is_direct.get(field_u, False)

    if not force_analog:
        # Tier 1: standalone egen assay (feltnavnet ER grade, og direct)
        if grade_has_assay(grade) and is_direct:
            q = assay_q.loc[grade]
            return q.to_dict(), "1_STANDALONE", f"assay:{grade}"

        # Tier 2: blend-assay (selges som navngitt blend)
        if grade_has_assay(grade):
            q = assay_q.loc[grade]
            return q.to_dict(), "2_BLEND", f"blend:{grade}"

    # Tier 3: analog — MEDIAN av k=3 nærmeste API i samme område.
    # Validering viste at k=3-median slår både enkelt-nærmeste-nabo (for støyete)
    # og ren geografi-median (mister API-relevans) på 4 av 6 lab-variabler.
    K = 3
    pool = analog_pool
    if exclude is not None:
        pool = pool[pool.field_u != exclude]
    if api_for_match is not None and len(pool) > 0:
        same_area = pool[pool.main_area == main_area]
        cand = same_area if len(same_area) >= 1 else pool  # fall til hele NCS hvis tomt område
        cand = cand.copy()
        cand["api_dist"] = (cand.api_match - api_for_match).abs()
        knn = cand.sort_values("api_dist").head(K)
        q = {"api_gravity": api_for_match}  # API fra DST (feltspesifikk)
        for v in LAB_VARS:
            q[v] = knn[v].median()
        scope = "område" if len(same_area) >= 1 else "NCS"
        nearest = knn.iloc[0]
        return q, "3_ANALOG", f"analog-k{len(knn)}:{nearest.field_u}+(ΔAPI={nearest.api_dist:.1f},{scope})"

    # Tier 4: geografi-median
    geo = analog_pool[analog_pool.main_area == main_area]
    if len(geo) == 0:
        geo = analog_pool
    q = {"api_gravity": api_for_match if api_for_match else geo.api_match.median()}
    for v in LAB_VARS:
        q[v] = geo[v].median()
    return q, "4_GEOGRAPHY", f"geo-median:{main_area}"

# Bygg analog-pool: felt som har en blend/standalone assay (med deres assay-API + lab-vars)
pool_rows = []
for fu in universe.field_u:
    grade = field_to_grade.get(fu)
    if grade_has_assay(grade):
        q = assay_q.loc[grade]
        pool_rows.append({
            "field_u": fu,
            "api_match": q["api_gravity"],
            "main_area": field_area.get(fu, "Unknown"),
            **{v: q[v] for v in LAB_VARS},
        })
analog_pool = pd.DataFrame(pool_rows)
log(f"Analog-pool (felt m/ assay): {len(analog_pool)}")

# ────────────────────────────────────────────────────────────────────────────
# 4. VALIDERING: hold ut hvert assay-felt, imputer lab-vars via analog
# ────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("VALIDERING: nærmeste-API-analog for lab-variabler (leave-one-out)")
log("=" * 70)

# VIKTIG: valider på UNIKE assay-grades, ikke felt-duplikater.
# Mange felt deler samme blend-assay (alle Alvheim-felt = identisk) → tvilling-
# matching gir falsk 0 feil. Vi bygger grade-nivå pool med modal-område per grade.
grade_pool_rows = []
for grade in sorted(set(g for g in field_to_grade.values() if grade_has_assay(g))):
    # modal hovedområde blant felt som selger denne grade
    fields_w_grade = [fu for fu, g in field_to_grade.items() if g == grade]
    areas = [field_area.get(fu, "Unknown") for fu in fields_w_grade]
    modal_area = max(set(areas), key=areas.count) if areas else "Unknown"
    q = assay_q.loc[grade]
    grade_pool_rows.append({
        "field_u": grade, "api_match": q["api_gravity"], "main_area": modal_area,
        **{v: q[v] for v in LAB_VARS},
    })
grade_pool = pd.DataFrame(grade_pool_rows)
log(f"Unike assay-grades for validering: {len(grade_pool)}")

val_rows = []
for _, r in grade_pool.iterrows():
    # nærmeste-API ANNEN grade (samme område hvis mulig)
    q_imp, tier, prov = assign_quality(
        r.field_u, api_for_match=r.api_match, main_area=r.main_area,
        analog_pool=grade_pool, exclude=r.field_u, force_analog=True,
    )
    if tier.startswith("3"):
        row = {"field": r.field_u, "analog_provenance": prov}
        for v in LAB_VARS:
            row[f"{v}_true"] = r[v]
            row[f"{v}_imp"] = q_imp[v]
        val_rows.append(row)

val = pd.DataFrame(val_rows)
log(f"\nTestbare analog-imputeringer: {len(val)} felt")
log(f"\n{'Variabel':22s} {'median |feil|':>14s} {'median rel.feil':>16s} {'n':>4s}")
log("-" * 60)
val_summary = {}
for v in LAB_VARS:
    sub = val.dropna(subset=[f"{v}_true", f"{v}_imp"])
    if len(sub) < 3:
        continue
    abs_err = (sub[f"{v}_imp"] - sub[f"{v}_true"]).abs()
    rel_err = (abs_err / sub[f"{v}_true"].abs().clip(lower=0.01)) * 100
    val_summary[v] = {"mae": abs_err.median(), "mape": rel_err.median(), "n": len(sub)}
    log(f"  {v:22s} {abs_err.median():13.3f} {rel_err.median():15.0f}% {len(sub):4d}")

# Sammenlign mot geografi-median som baseline
log(f"\n  Sammenligning: analog vs ren geografi-median (baseline)")
log(f"  {'Variabel':22s} {'analog MAE':>12s} {'geo MAE':>10s} {'forbedring':>12s}")
log("  " + "-" * 56)
for v in LAB_VARS:
    sub = val.dropna(subset=[f"{v}_true"])
    if len(sub) < 3:
        continue
    analog_mae = (sub[f"{v}_imp"] - sub[f"{v}_true"]).abs().median()
    # geografi-baseline: median av alle ANDRE grades i samme område
    gp_area = grade_pool.set_index("field_u")["main_area"].to_dict()
    geo_errs = []
    for _, rr in sub.iterrows():
        g = rr.field
        area = gp_area.get(g, "Unknown")
        others = grade_pool[(grade_pool.field_u != g) & (grade_pool.main_area == area)]
        if len(others) == 0:
            others = grade_pool[grade_pool.field_u != g]
        geo_errs.append(abs(others[v].median() - rr[f"{v}_true"]))
    geo_mae = np.median(geo_errs)
    imp = (1 - analog_mae / geo_mae) * 100 if geo_mae > 0 else 0
    log(f"  {v:22s} {analog_mae:11.3f} {geo_mae:9.3f} {imp:+11.0f}%")

# ────────────────────────────────────────────────────────────────────────────
# 5. ANVEND PÅ ALLE NCS-FELT
# ────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("ANVENDER PÅ ALLE NCS OLJEFELT")
log("=" * 70)

out_rows = []
for _, r in universe.iterrows():
    fu = r.field_u
    api_match = dst_api.get(fu, r.api_gravity)  # DST-API foretrukket for analog-match
    q, tier, prov = assign_quality(fu, api_match, r.main_area, analog_pool)
    out_rows.append({
        "field": fu, "main_area": r.main_area, "operator": r.operator,
        "tier": tier, "provenance": prov,
        "api_gravity": q["api_gravity"],
        **{v: q[v] for v in LAB_VARS},
    })

out = pd.DataFrame(out_rows).sort_values(["tier", "field"])
log(f"\nTier-fordeling:")
for tier, n in out.tier.value_counts().sort_index().items():
    log(f"  {tier:18s}  {n} felt")

out.to_csv(PROC / "63_ncs_field_quality.csv", index=False)
val.to_csv(PROC / "63_imputation_validation.csv", index=False)
log(f"\nLagret: 63_ncs_field_quality.csv ({len(out)} felt)")

# ── Aker BP-felt: vis hvilke som endres ──
log(f"\n── Aker BP-felt — tier per felt ──")
akbp = out[out.operator.str.contains("Aker BP", na=False)]
log(f"\n  {'Felt':18s} {'Tier':14s} {'API':>6s} {'Svovel':>7s} {'Provenance':30s}")
for _, r in akbp.sort_values("tier").iterrows():
    log(f"  {r.field:18s} {r.tier:14s} {r.api_gravity:6.1f} {r.sulfur_pct:6.2f}% {r.provenance[:30]:30s}")

# ────────────────────────────────────────────────────────────────────────────
# 6. FIGUR
# ────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# Panel 1: validation — sulfur imputed vs true
ax = fig.add_subplot(gs[0, 0])
sub = val.dropna(subset=["sulfur_pct_true", "sulfur_pct_imp"])
ax.scatter(sub.sulfur_pct_imp, sub.sulfur_pct_true, s=55, alpha=0.75, c="#1565C0", edgecolors="white")
lim = [0, max(sub.sulfur_pct_true.max(), sub.sulfur_pct_imp.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=0.7, alpha=0.5)
ax.set_xlabel("Imputert svovel (analog)"); ax.set_ylabel("Faktisk svovel %")
ax.set_title(f"Svovel-imputering\nmedian |feil| {val_summary.get('sulfur_pct',{}).get('mae',0):.2f}%-poeng",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 2: middle distillate
ax = fig.add_subplot(gs[0, 1])
sub = val.dropna(subset=["middle_distillate_pct_true", "middle_distillate_pct_imp"])
ax.scatter(sub.middle_distillate_pct_imp, sub.middle_distillate_pct_true, s=55, alpha=0.75, c="#2E7D32", edgecolors="white")
lim = [sub.middle_distillate_pct_true.min()*0.9, sub.middle_distillate_pct_true.max()*1.1]
ax.plot(lim, lim, "k--", lw=0.7, alpha=0.5)
ax.set_xlabel("Imputert"); ax.set_ylabel("Faktisk middeldestillat %")
ax.set_title(f"Middeldestillat-imputering\nmedian |feil| {val_summary.get('middle_distillate_pct',{}).get('mae',0):.1f}%-poeng",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 3: analog vs geo MAE (relative improvement bars)
ax = fig.add_subplot(gs[0, 2])
varnames, analog_maes, geo_maes = [], [], []
for v in LAB_VARS:
    sub = val.dropna(subset=[f"{v}_true"])
    if len(sub) < 3: continue
    am = (sub[f"{v}_imp"] - sub[f"{v}_true"]).abs().median()
    gp_area2 = grade_pool.set_index("field_u")["main_area"].to_dict()
    ge = []
    for _, rr in sub.iterrows():
        area = gp_area2.get(rr.field, "Unknown")
        others = grade_pool[(grade_pool.field_u != rr.field) & (grade_pool.main_area == area)]
        if len(others) == 0: others = grade_pool[grade_pool.field_u != rr.field]
        ge.append(abs(others[v].median() - rr[f"{v}_true"]))
    varnames.append(v.replace("_pct","").replace("_ppm","")[:10])
    analog_maes.append(am); geo_maes.append(np.median(ge))
x = np.arange(len(varnames))
ax.bar(x-0.2, analog_maes, 0.4, color="#1565C0", alpha=0.85, label="Analog")
ax.bar(x+0.2, geo_maes, 0.4, color="#bbb", alpha=0.85, label="Geografi")
ax.set_xticks(x); ax.set_xticklabels(varnames, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Median |feil|"); ax.set_title("Analog vs geografi-baseline", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

# Panel 4: tier distribution
ax = fig.add_subplot(gs[1, 0])
tier_counts = out.tier.value_counts().sort_index()
tier_colors = {"1_STANDALONE":"#2E7D32","2_BLEND":"#1565C0","3_ANALOG":"#FF9800","4_GEOGRAPHY":"#C62828"}
ax.bar(range(len(tier_counts)), tier_counts.values,
       color=[tier_colors.get(t,"#999") for t in tier_counts.index], alpha=0.85)
ax.set_xticks(range(len(tier_counts)))
ax.set_xticklabels([t.split("_")[1] for t in tier_counts.index], fontsize=9)
ax.set_ylabel("Antall felt"); ax.set_title(f"Tier-fordeling ({len(out)} felt)", fontsize=10, fontweight="bold")
for i, v in enumerate(tier_counts.values):
    ax.text(i, v+0.5, str(v), ha="center", fontsize=9, fontweight="bold")

# Panel 5: API vs sulfur colored by tier (the imputed universe)
ax = fig.add_subplot(gs[1, 1:])
for tier, color in tier_colors.items():
    sub = out[out.tier == tier]
    ax.scatter(sub.api_gravity, sub.sulfur_pct, s=45, alpha=0.7, c=color,
               label=tier.split("_")[1], edgecolors="white", lw=0.4)
ax.set_xlabel("API gravity"); ax.set_ylabel("Svovel %")
ax.set_title("NCS-felt kvalitetskart (farge = imputerings-tier)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("NCS Field → Crude Quality Imputation — analog (nærmeste API + område) for lab-variabler",
             fontsize=13, fontweight="bold", y=0.99)
plt.savefig(PROC / "63_field_quality_imputation.png", dpi=170, bbox_inches="tight")
log(f"\nLagret figur: 63_field_quality_imputation.png")

with open(PROC / "63_imputation_summary.txt", "w") as f:
    f.write("\n".join(lines))
