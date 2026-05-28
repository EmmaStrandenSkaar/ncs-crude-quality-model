"""
Script 45 — Feltspesifikk sammenligning: modellpredikert differensial vs. offisiell normpris

Petroleumsprisrådet setter kvartalsvise normpriser per NCS-felt (differensial vs. Dated Brent).
Dette er den mest offisielle markedsprisen som finnes per felt — brukt for skatteformål.

Mapping:
  ALVHEIM   normpris → modell-prediksjon for ALVHEIM / BØYLA / SKOGUL
  EKOFISK   normpris → modell-prediksjon for VALHALL, HOD, ULA, TAMBAR, TAMBAR ØST
  GRANE     normpris → modell-prediksjon for EDVARD GRIEG, IVAR AASEN
  JOHAN SVERDRUP → direkte
  SKARV     → direkte

Sammenligner:
  - Modellpredikert differensial vs. Brent (fra script 42)
  - Offisiell normpris-differensial (fra Petroleumsprisrådet)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUT_PNG      = DATA_PROC / "45_field_level_normpris_vs_model.png"

# ---------------------------------------------------------------------------
# Mapping: normpris-felt → liste av AKRBP-felt som selger inn i samme eksportstrøm
# ---------------------------------------------------------------------------
NORMPRIS_TO_AKRBP = {
    "ALVHEIM":         ["ALVHEIM", "BØYLA", "SKOGUL"],
    "EKOFISK":         ["VALHALL", "HOD", "ULA", "TAMBAR", "TAMBAR ØST"],
    "GRANE":           ["EDVARD GRIEG", "IVAR AASEN"],
    "JOHAN SVERDRUP":  ["JOHAN SVERDRUP"],
    "SKARV":           ["SKARV"],
}

# Visningsnavn for plottet
DISPLAY_NAME = {
    "ALVHEIM":        "Alvheim  (→ ALVHEIM / BØYLA / SKOGUL)",
    "EKOFISK":        "Ekofisk Blend  (→ VALHALL / HOD / ULA / TAMBAR)",
    "GRANE":          "Grane Blend  (→ EDVARD GRIEG / IVAR AASEN)",
    "JOHAN SVERDRUP": "Johan Sverdrup",
    "SKARV":          "Skarv",
}

# Farger per felt
COLORS = {
    "ALVHEIM":        "#2E86AB",   # blå
    "EKOFISK":        "#A23B72",   # lilla
    "GRANE":          "#F18F01",   # oransje
    "JOHAN SVERDRUP": "#C73E1D",   # rød
    "SKARV":          "#3B1F2B",   # mørk
}


def load_normpris_quarterly() -> pd.DataFrame:
    """Last inn og aggreger normpris-differensialer til kvartalsvis."""
    df = pd.read_csv(DATA_PROC / "normpris_differentials_long.csv")

    # Bygg year-month og quarter
    df["year"]    = df["year"].astype(int)
    df["month"]   = df["month"].astype(int)
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
    df["qstr"]    = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)

    nq = (df.groupby(["field", "qstr", "year", "quarter"])["differential_usd"]
            .mean()
            .reset_index()
            .rename(columns={"differential_usd": "normpris_diff"}))
    return nq


def load_model_field_quarterly() -> pd.DataFrame:
    """Last inn modellpredikerte differensialer per felt per kvartal."""
    df = pd.read_csv(DATA_PROC / "42_akrbp_field_breakdown.csv")
    df["year"]    = df["qstr"].str[:4].astype(int)
    df["quarter"] = df["qstr"].str[6].astype(int)
    return df


def main():
    np_q   = load_normpris_quarterly()
    mod_q  = load_model_field_quarterly()

    # Felles start: 2017 (Skarv mangler normpris før, JS mangler før 2021)
    START_Q = "2017-Q1"
    END_Q   = "2025-Q4"

    # ---------------------------------------------------------------------------
    # Bygg sammenlignings-DF per normpris-felt
    # ---------------------------------------------------------------------------
    records = []
    for np_field, akrbp_fields in NORMPRIS_TO_AKRBP.items():
        # Normpris for dette feltet
        nf = np_q[np_q["field"] == np_field][["qstr", "normpris_diff"]].copy()

        # Modell: vektet gjennomsnitt av AKRBP-felt (produksjonsvektet)
        mf = mod_q[mod_q["field"].isin(akrbp_fields)].copy()
        # Vektet differensial: sum(share_i * diff_i) / sum(share_i)
        mf_agg = (mf.groupby("qstr")
                    .apply(lambda g: pd.Series({
                        "model_diff_weighted": (
                            (g["diff_pred"] * g["share"]).sum() / g["share"].sum()
                            if g["share"].sum() > 0 else np.nan
                        ),
                        "model_diff_simple": g["diff_pred"].mean(),
                        "total_share": g["share"].sum(),
                    }))
                    .reset_index())

        merged = nf.merge(mf_agg, on="qstr", how="inner")
        merged["np_field"] = np_field
        merged = merged[(merged["qstr"] >= START_Q) & (merged["qstr"] <= END_Q)]
        records.append(merged)

    comp = pd.concat(records, ignore_index=True)

    # ---------------------------------------------------------------------------
    # Statistikk
    # ---------------------------------------------------------------------------
    print("=== Feltspesifikk modell vs. normpris (differensial vs. Brent) ===\n")
    print(f"{'Felt':<25} {'Corr':>6}  {'MAE':>5}  {'Bias (mod-np)':>13}  "
          f"{'np mean':>8}  {'mod mean':>9}")
    print("-" * 75)
    for np_field in NORMPRIS_TO_AKRBP:
        sub = comp[comp["np_field"] == np_field].dropna(subset=["normpris_diff", "model_diff_weighted"])
        if len(sub) < 4:
            continue
        corr = sub["model_diff_weighted"].corr(sub["normpris_diff"])
        mae  = (sub["model_diff_weighted"] - sub["normpris_diff"]).abs().mean()
        bias = (sub["model_diff_weighted"] - sub["normpris_diff"]).mean()
        np_m = sub["normpris_diff"].mean()
        md_m = sub["model_diff_weighted"].mean()
        print(f"  {DISPLAY_NAME[np_field]:<43} {corr:>6.3f}  {mae:>5.2f}  {bias:>+13.2f}  "
              f"{np_m:>8.2f}  {md_m:>9.2f}")

    # ---------------------------------------------------------------------------
    # Plot: 5 subplots (ett per felt/blend-gruppe)
    # ---------------------------------------------------------------------------
    n_panels = len(NORMPRIS_TO_AKRBP)
    fig = plt.figure(figsize=(16, 14), facecolor="white")
    fig.suptitle(
        "Feltspesifikk differensial vs. Brent — Modellpredikert vs. Offisiell normpris\n"
        "(Petroleumsprisrådet, endelige normpriser)",
        fontsize=13, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_panels)]
    # Siste subplot (5. felt) → midtstilt
    if n_panels == 5:
        axes[4].remove()
        axes[4] = fig.add_subplot(gs[2, :])

    for idx, (np_field, akrbp_fields) in enumerate(NORMPRIS_TO_AKRBP.items()):
        ax = axes[idx]
        color = COLORS[np_field]

        sub = comp[comp["np_field"] == np_field].dropna(subset=["normpris_diff", "model_diff_weighted"])
        sub = sub.sort_values("qstr")

        if len(sub) < 2:
            ax.text(0.5, 0.5, "For lite data", transform=ax.transAxes, ha="center")
            continue

        x     = range(len(sub))
        xlbl  = sub["qstr"].tolist()

        # Normpris (solid, grønn med markør)
        ax.plot(x, sub["normpris_diff"], color="#1D6A39", lw=2.2, marker="o", ms=3.5,
                label="Normpris (offisiell)")

        # Modell (solid, feltfarge)
        ax.plot(x, sub["model_diff_weighted"], color=color, lw=2.0, ls="--",
                label="Modell (predikert)")

        # Fyll mellom
        ax.fill_between(x, sub["normpris_diff"], sub["model_diff_weighted"],
                        alpha=0.12, color=color)

        # Nulllinje
        ax.axhline(0, color="gray", lw=0.8, ls=":")

        # Stats i øvre hjørne
        corr = sub["model_diff_weighted"].corr(sub["normpris_diff"])
        mae  = (sub["model_diff_weighted"] - sub["normpris_diff"]).abs().mean()
        bias = (sub["model_diff_weighted"] - sub["normpris_diff"]).mean()
        ax.text(0.98, 0.97,
                f"corr={corr:.3f}\nMAE={mae:.2f}\nbias={bias:+.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.9))

        # X-akse: vis bare hvert 4. kvartal
        tick_pos  = [i for i, q in enumerate(xlbl) if q.endswith("Q1")]
        tick_lbls = [xlbl[i][:4] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbls, fontsize=8)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylabel("Diff. vs. Brent (USD/fat)", fontsize=9)
        ax.set_title(DISPLAY_NAME[np_field], fontsize=10, fontweight="bold", color=color)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_facecolor("#FAFAFA")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Global fotnote
    fig.text(0.02, 0.005,
             "Kilde: Petroleumsprisrådet (normpris, endelige), modell (OLS-regresjon på 45 crude-grades, "
             "N=2 869 obs.).\n"
             "Normpris = offisiell Dated Brent-differensial brukt for skatteformål. "
             "Bøyla/Skogul selger via Alvheim FPSO; Edvard Grieg/Ivar Aasen via Grane Blend; "
             "Valhall/HOD/ULA/Tambar via Ekofisk Blend.",
             fontsize=7, color="#777777", style="italic")

    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\n  Graf lagret: {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
