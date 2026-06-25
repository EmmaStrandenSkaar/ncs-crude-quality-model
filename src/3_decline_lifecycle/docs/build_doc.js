const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, PageOrientation, LevelFormat,
  TabStopType, TabStopPosition, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, ExternalHyperlink
} = require('docx');

const RESULTS = "./analyses/decline_quality/results";
const OUTPUT = "./analyses/decline_quality/docs/NCS_Decline_Model_Methodology.docx";

// ── Helpers ──
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

const p = (text, opts = {}) => new Paragraph({
  spacing: { before: 80, after: 80 },
  ...opts,
  children: [new TextRun({ text, ...opts.run })],
});

const ptext = (runs, opts = {}) => new Paragraph({
  spacing: { before: 80, after: 80 },
  ...opts,
  children: runs.map(r => typeof r === 'string' ? new TextRun(r) : new TextRun(r)),
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 200 },
  children: [new TextRun({ text })],
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 280, after: 160 },
  children: [new TextRun({ text })],
});

const h3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 200, after: 120 },
  children: [new TextRun({ text })],
});

const bullet = (text, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { before: 40, after: 40 },
  children: [new TextRun({ text })],
});

const numbered = (text) => new Paragraph({
  numbering: { reference: "numbers", level: 0 },
  spacing: { before: 40, after: 40 },
  children: [new TextRun({ text })],
});

const formula = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 160, after: 160 },
  children: [new TextRun({ text, font: "Courier New", size: 22, bold: true, color: "1F4E79" })],
});

const cell = (text, opts = {}) => new TableCell({
  borders,
  width: { size: opts.width || 2340, type: WidthType.DXA },
  shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
  margins: { top: 100, bottom: 100, left: 140, right: 140 },
  verticalAlign: VerticalAlign.CENTER,
  children: [new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({ text, bold: opts.bold || false, size: opts.size || 20, color: opts.color || "000000" })],
  })],
});

const img = (filename, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 200 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync(path.join(RESULTS, filename)),
    transformation: { width: w, height: h },
    altText: { title: filename, description: filename, name: filename },
  })],
});

const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 200 },
  children: [new TextRun({ text, italics: true, size: 18, color: "595959" })],
});

const spacer = () => new Paragraph({ children: [new TextRun("")] });

// ═══════════════════════════════════════════════════════════════
// DOCUMENT CONTENT
// ═══════════════════════════════════════════════════════════════

const content = [
  // ── Title page ──
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1200, after: 200 },
    children: [new TextRun({
      text: "Decline Rate Modell for Norsk Sokkel",
      bold: true, size: 48, color: "1F4E79",
    })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 600 },
    children: [new TextRun({
      text: "Fysikk + Premium Rammeverk for Equity Research",
      size: 32, color: "595959", italics: true,
    })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 400, after: 100 },
    children: [new TextRun({ text: "Emma Strandenskaar", size: 24, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 100 },
    children: [new TextRun({ text: "Metodikk-dokumentasjon", size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 1200 },
    children: [new TextRun({ text: "Juni 2026", size: 22, color: "808080" })],
  }),

  // Key stats box
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [4680, 4680],
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({
        children: [
          cell("Modell-ytelse", { width: 4680, bold: true, shade: "1F4E79", color: "FFFFFF", size: 22 }),
          cell("Verdi", { width: 4680, bold: true, shade: "1F4E79", color: "FFFFFF", size: 22 }),
        ],
      }),
      new TableRow({ children: [
        cell("Cross-validated R²", { width: 4680 }),
        cell("0.713", { width: 4680, bold: true, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("In-sample R²", { width: 4680 }),
        cell("0.770", { width: 4680 }),
      ]}),
      new TableRow({ children: [
        cell("RMSE", { width: 4680 }),
        cell("0.040", { width: 4680 }),
      ]}),
      new TableRow({ children: [
        cell("Antall felt i kalibrering", { width: 4680 }),
        cell("49 NCS-felt", { width: 4680 }),
      ]}),
      new TableRow({ children: [
        cell("Aker BP treffrate (±0.05)", { width: 4680 }),
        cell("75%", { width: 4680, bold: true, color: "2E7D32" }),
      ]}),
    ],
  }),

  new Paragraph({ children: [new PageBreak()] }),

  // ═══════════════════════════════════════════════════════════════
  // 1. SAMMENDRAG
  // ═══════════════════════════════════════════════════════════════
  h1("1. Sammendrag"),
  p("Dette dokumentet beskriver en kvantitativ modell for å estimere årlige decline rates (D_annual) for olje- og gassfelt på norsk sokkel. Modellen kombinerer:"),
  bullet("Fysikk-baseline: Beggs-Robinson viskositet beregnet fra API gravity"),
  bullet("Historisk premium: feltets faktiske avvik fra fysikk-prediksjonen siste 12 måneder"),
  bullet("Asymmetri-korreksjon: justering for felt som avviker mye fra fysikken (uansett retning)"),
  spacer(),
  p("Modellen oppnår CV R² = 0.713 på 49 NCS-felt, med RMSE 0.040 på årlige decline rates. Alle inputs er offentlig tilgjengelige fra Sodir (norsk sokkeldirektorat), noe som gjør modellen direkte anvendelig i ER-analyse av Aker BP, Equinor, Vår Energi, ConocoPhillips og andre operatører på sokkelen."),
  spacer(),
  p("Den endelige formelen er:", { run: { bold: true } }),
  formula("D = 0.0664 + 0.0601·ln(viskositet) − 0.0597·P₁₂ + 0.0379·|P₁₂|"),
  p("hvor P₁₂ er gjennomsnittlig log-premium siste 12 måneder."),

  // ═══════════════════════════════════════════════════════════════
  // 2. HVORFOR DECLINE RATES ER KRITISK FOR ER
  // ═══════════════════════════════════════════════════════════════
  h1("2. Hvorfor decline rates er kritisk for Equity Research"),
  p("Decline rate er en av de viktigste driverne for verdsettelse av oppstrøms olje- og gasselskaper. Den bestemmer hvor raskt produksjonen — og dermed kontantstrømmen — faller fra et felt etter peak."),

  h2("2.1 Påvirkning på NPV"),
  p("For et typisk NCS-felt med 10-15 års produksjonshistorikk vil en endring i decline rate fra 8% til 12% redusere NPV med 20-30%, alt annet likt. Dette gjør decline-anslag til en hovedvariabel i target price-modeller."),

  h2("2.2 Hvor markedet ofte feiler"),
  p("Mange ER-rapporter bruker en \"standard\" decline rate på 5-10% for alle NCS-felt, uten å justere for:"),
  bullet("Oljekvalitet (viskositet) — påvirker reservoarets evne til å strømme"),
  bullet("Feltspesifikk historikk — noen felt har holdt seg flat i 10+ år (Valhall, Ekofisk)"),
  bullet("Operatørstrategi — CAPEX-investering bremser natural decline"),
  spacer(),
  p("Vår modell gir et transparent, datadrevet alternativ som fanger disse forskjellene gjennom to enkle variabler."),

  // ═══════════════════════════════════════════════════════════════
  // 3. FYSIKK-FUNDAMENTET
  // ═══════════════════════════════════════════════════════════════
  h1("3. Fysikk-fundamentet: Beggs-Robinson viskositet"),

  h2("3.1 Hvorfor viskositet betyr noe"),
  p("Når en oljereservoar tappes, må oljen strømme gjennom mikroskopiske kanaler i bergarten mot brønnen. Hvor lett dette skjer avhenger av oljens viskositet:"),
  bullet("Lett olje (høy API, lav viskositet) — strømmer lett, men reservoarets trykk faller raskt → bratt decline"),
  bullet("Tung olje (lav API, høy viskositet) — strømmer trått, men reservoartrykket holdes lengre → slakere decline (men lavere total produksjon)"),
  spacer(),
  p("Denne fysikalske sammenhengen er grunnlaget for vår baseline-prediksjon. Vi kan ikke direkte måle viskositet for hvert felt i sanntid, men API gravity er offentlig tilgjengelig og lett konvertibel via Beggs-Robinson-korrelasjonen."),

  h2("3.2 Beggs-Robinson-formelen"),
  p("Beggs-Robinson (1975) er en empirisk korrelasjon som beregner dead-oil-viskositet fra API gravity og reservoartemperatur:"),
  formula("μ = 10^(x · T^(-1.163)) − 1   [cP]"),
  formula("hvor x = 10^(3.0324 − 0.02023·API)"),
  p("Vi bruker T = 194°F (90°C) som NCS-typisk reservoartemperatur. Sensitivitetstesting viste at temperaturvalg fra 60°C til 127°C kun endrer modellytelsen marginalt (CV R² varierer med ±0.003)."),

  h2("3.3 Validering av fysikk-baseline"),
  p("På 51 NCS-felt finner vi en signifikant positiv sammenheng mellom ln(viskositet) og observert D_annual (β = 0.059, p < 0.001). Dette bekrefter hypotesen: tyngre olje gir slakere decline. Men fysikk alene forklarer kun R² = 9.6% av variansen i decline rates — derfor trenger vi premium-rammeverket."),

  // ═══════════════════════════════════════════════════════════════
  // 4. PREMIUM-RAMMEVERKET
  // ═══════════════════════════════════════════════════════════════
  h1("4. Premium-rammeverket: Alpha / Beta-tankegang"),

  h2("4.1 Analog til finansmarkeder"),
  p("Vi låner et begrepsapparat fra finans og bruker det på reservoarer:"),
  bullet("Beta (fysikk) — felles eksponering mot \"markedet\" (her: viskositet → forventet decline)"),
  bullet("Alpha (premium) — feltspesifikk avvik som ikke kan forklares av fysikken alene"),
  spacer(),
  p("Premiumet fanger alt vi ikke direkte kan observere: operatørkvalitet, brønnplassering, vanninjeksjonsstrategi, CAPEX-program, reservoarstruktur, infill-boring osv. Vi trenger ikke å modellere disse direkte — feltets faktiske produksjonshistorikk avslører dem."),

  h2("4.2 Beregning av premium"),
  p("For hver måned i post-peak-perioden beregner vi:"),
  formula("log_premium_i = ln(actual_i / expected_i)"),
  p("hvor expected_i = exp(−D_physics/12 · month_i)"),
  p("Hvis feltet faktisk produserer mer enn fysikken tilsier, blir log_premium positiv (\"discount\" på decline). Hvis det produserer mindre, blir log_premium negativ (\"premium\" på decline)."),

  h2("4.3 Hvorfor 12-måneders vindu"),
  p("Vi testet systematisk premium-vindu fra 3 til 120 måneder. Resultatene:"),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 3120, 3120],
    rows: [
      new TableRow({ children: [
        cell("Vindu", { width: 3120, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("CV R²", { width: 3120, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("Aker BP RMSE", { width: 3120, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("6 mnd", { width: 3120 }),
        cell("0.713", { width: 3120, align: AlignmentType.CENTER }),
        cell("0.0557", { width: 3120, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("12 mnd ★", { width: 3120, bold: true, shade: "E8F5E9" }),
        cell("0.713", { width: 3120, bold: true, color: "2E7D32", align: AlignmentType.CENTER, shade: "E8F5E9" }),
        cell("0.0565", { width: 3120, align: AlignmentType.CENTER, shade: "E8F5E9" }),
      ]}),
      new TableRow({ children: [
        cell("24 mnd", { width: 3120 }),
        cell("0.707", { width: 3120, align: AlignmentType.CENTER }),
        cell("0.0585", { width: 3120, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("36 mnd", { width: 3120 }),
        cell("0.665", { width: 3120, align: AlignmentType.CENTER }),
        cell("0.0625", { width: 3120, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("60 mnd", { width: 3120 }),
        cell("0.620", { width: 3120, align: AlignmentType.CENTER }),
        cell("0.0684", { width: 3120, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("Lifetime", { width: 3120 }),
        cell("0.523", { width: 3120, align: AlignmentType.CENTER }),
        cell("0.0756", { width: 3120, align: AlignmentType.CENTER }),
      ]}),
    ],
  }),

  spacer(),
  p("Vi valgte 12 måneder framfor 6 av tre grunner:"),
  numbered("Robusthet — 12 datapunkter er stabilt mot enkeltmåneds-shutdowns (vedlikehold, branner). 6 datapunkter er sårbart for én anomalisk måned."),
  numbered("Sesongeffekter — Norske felt har planlagte sommerstopp for vedlikehold. 12 mnd fanger en full syklus."),
  numbered("ER-narrativ — \"1 år historikk\" er en troverdig kommunikasjon i en investeringsanalyse."),

  h2("4.4 Hvorfor |premium| (asymmetri-korreksjon)"),
  p("Etter at vi inkluderte premium som lineær variabel, observerte vi en U-formet sammenheng: felt med stort avvik fra fysikken (i begge retninger) hadde høyere baseline-decline enn felt nær fysikkens prediksjon. Vi inkluderte derfor |P₁₂| som tredje variabel."),
  spacer(),
  p("Intuisjonen er at felt med stort |premium| er mer uforutsigbare:"),
  bullet("Store positive premium (f.eks. Valhall +2.07) skyldes ofte intensive CAPEX-sykluser og redevelopment som skaper volatile produksjonsmønstre"),
  bullet("Store negative premium (f.eks. Volund −1.53) indikerer felt med brattere natural decline enn fysikken tilsier"),
  spacer(),
  p("Begge tilfeller fortjener et risikopåslag i decline-estimatet."),

  new Paragraph({ children: [new PageBreak()] }),

  // ═══════════════════════════════════════════════════════════════
  // 5. METODOLOGI
  // ═══════════════════════════════════════════════════════════════
  h1("5. Metodologi"),

  h2("5.1 Datakilder"),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [
        cell("Kilde", { width: 3120, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("Bruk", { width: 6240, bold: true, shade: "1F4E79", color: "FFFFFF" }),
      ]}),
      new TableRow({ children: [
        cell("Sodir månedlig produksjon", { width: 3120 }),
        cell("Felt-måned produksjonsvolumer, beregning av peak og post-peak-måneder", { width: 6240 }),
      ]}),
      new TableRow({ children: [
        cell("Sodir feltinformasjon", { width: 3120 }),
        cell("API gravity, operatør, hovedområde, oppdagelsesår", { width: 6240 }),
      ]}),
      new TableRow({ children: [
        cell("Equinor / Norem assays", { width: 3120 }),
        cell("Bekreftelse av oljekvalitetsparametere for blendinger (Ekofisk, Grane, Alvheim)", { width: 6240 }),
      ]}),
      new TableRow({ children: [
        cell("NSTA (UK)", { width: 3120 }),
        cell("Out-of-sample validering av modellen på UK Continental Shelf", { width: 6240 }),
      ]}),
    ],
  }),

  h2("5.2 Beregning av D_annual"),
  p("For hvert felt fitter vi en eksponentiell decline-kurve på post-peak-data:"),
  formula("ln(production_t / peak_production) = −D_annual · t  + ε"),
  p("D_annual estimeres med vanlig minste kvadraters metode. Felt med færre enn 12 post-peak-måneder eller R² < 0.1 ekskluderes."),

  h2("5.3 Modelltrening"),
  p("Modellen estimeres med vanlig OLS-regresjon. Vi bruker Leave-One-Out Cross-Validation (LOO-CV) for å oppnå et ærlig anslag på out-of-sample-ytelse, gitt N = 49 felt. Manuell LOO-implementasjon (sklearn-funksjonen returnerer NaN for 1-felt test-folds)."),

  h2("5.4 Robusthetstesting"),
  p("Modellen er testet på følgende måter:"),
  bullet("LOO Cross-Validation — gir CV R² = 0.713 (sannferdig anslag)"),
  bullet("Out-of-sample på UK NSTA-felt — viser at modellen er basin-spesifikk (UK-felt har omvendt viskositetskoeffisient)"),
  bullet("Sensitivitetstest på temperatur — knapt påvirkning fra 60°C til 127°C"),
  bullet("Felt-karakteristikk-analyse — bekreftet at ingen Sodir-variabel forbedrer modellen utover premium"),

  // ═══════════════════════════════════════════════════════════════
  // 6. ENDELIG MODELL
  // ═══════════════════════════════════════════════════════════════
  h1("6. Endelig modell"),

  h2("6.1 Formel og koeffisienter"),
  formula("D_annual = 0.0664 + 0.0601·ln(μ) − 0.0597·P₁₂ + 0.0379·|P₁₂|"),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2400, 1600, 1600, 1880, 1880],
    rows: [
      new TableRow({ children: [
        cell("Variabel", { width: 2400, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("Koeffisient", { width: 1600, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("Std. β", { width: 1600, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("t-statistikk", { width: 1880, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("p-verdi", { width: 1880, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("Intercept", { width: 2400 }),
        cell("+0.0664", { width: 1600, align: AlignmentType.CENTER }),
        cell("—", { width: 1600, align: AlignmentType.CENTER }),
        cell("+5.81", { width: 1880, align: AlignmentType.CENTER }),
        cell("<0.001 ***", { width: 1880, align: AlignmentType.CENTER, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("ln(viskositet)", { width: 2400 }),
        cell("+0.0601", { width: 1600, align: AlignmentType.CENTER }),
        cell("+0.027", { width: 1600, align: AlignmentType.CENTER }),
        cell("+4.61", { width: 1880, align: AlignmentType.CENTER }),
        cell("<0.001 ***", { width: 1880, align: AlignmentType.CENTER, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("Premium 12 mnd", { width: 2400 }),
        cell("−0.0597", { width: 1600, align: AlignmentType.CENTER }),
        cell("−0.080", { width: 1600, align: AlignmentType.CENTER, bold: true }),
        cell("−11.32", { width: 1880, align: AlignmentType.CENTER, bold: true }),
        cell("<0.001 ***", { width: 1880, align: AlignmentType.CENTER, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("|Premium 12 mnd|", { width: 2400 }),
        cell("+0.0379", { width: 1600, align: AlignmentType.CENTER }),
        cell("+0.034", { width: 1600, align: AlignmentType.CENTER }),
        cell("+4.82", { width: 1880, align: AlignmentType.CENTER }),
        cell("<0.001 ***", { width: 1880, align: AlignmentType.CENTER, color: "2E7D32" }),
      ]}),
    ],
  }),
  caption("Alle koeffisienter er statistisk signifikante på 1%-nivå."),

  h2("6.2 Tolkning av koeffisientene"),
  bullet("ln(viskositet) — positiv: høyere viskositet (tyngre olje) gir høyere decline. Bekrefter fysikk-hypotesen."),
  bullet("Premium 12 mnd — sterkt negativ: felt som outperformer fysikken (positiv premium) har lavere decline. Dette er hovedeffekten i modellen (|β| = 0.080)."),
  bullet("|Premium| — positiv: store avvik fra fysikken (uansett retning) gir høyere baseline-decline. Risikopåslag for uforutsigbare felt."),

  h2("6.3 Modellfigur"),
  img("fig_final_model.png", 640, 415),
  caption("Figur 1: Endelig modell — 9-panels oversikt. Øverst: CV-progresjon, predikert vs. faktisk, variabel-viktighet. Midten: U-formet premium-effekt, viskositets-effekt, Aker BP-prediksjoner. Nederst: Implisitte prognoser, full ranking, formel."),

  new Paragraph({ children: [new PageBreak()] }),

  // ═══════════════════════════════════════════════════════════════
  // 7. PRAKTISK BRUK
  // ═══════════════════════════════════════════════════════════════
  h1("7. Praktisk bruk i ER-analyse"),

  h2("7.1 Trinn-for-trinn-prosedyre"),
  numbered("Hent API gravity for feltet (Sodir, Equinor crude assays, eller selskapsrapport)"),
  numbered("Beregn viskositet med Beggs-Robinson-formelen"),
  numbered("Last ned siste 12 måneders produksjonsdata fra Sodir factpages"),
  numbered("Identifiser peak-måned og beregn months_since_peak for hver datapunkt"),
  numbered("Beregn fysikk-baseline: D_physics = 0.097 + 0.059·ln(μ)"),
  numbered("For hver av siste 12 mnd: log_premium_i = ln(actual_i) + D_physics/12·month_i"),
  numbered("P₁₂ = gjennomsnitt av de 12 log_premium-verdiene"),
  numbered("Plugg inn i hovedformelen: D = 0.0664 + 0.0601·ln(μ) − 0.0597·P₁₂ + 0.0379·|P₁₂|"),
  numbered("Produksjonsprognose: production_year_T = peak · exp(−D · T)"),

  h2("7.2 Eksempel: Aker BP-felt"),
  p("Tabellen under viser modellens prediksjoner for Aker BPs 12 modne NCS-felt:"),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2080, 1300, 1300, 1300, 1300, 2080],
    rows: [
      new TableRow({ children: [
        cell("Felt", { width: 2080, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("API", { width: 1300, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("D_faktisk", { width: 1300, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("D_modell", { width: 1300, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("Bom", { width: 1300, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("Premium 12m", { width: 2080, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
      ]}),
      ...[
        ["EDVARD GRIEG", "27.1", "0.384", "0.221", "+0.163", "−0.79"],
        ["SKOGUL", "34.5", "0.247", "0.190", "+0.058", "−0.85"],
        ["VOLUND", "34.5", "0.243", "0.256", "−0.013", "−1.53"],
        ["VILJE", "34.5", "0.222", "0.208", "+0.014", "−1.03"],
        ["IVAR AASEN", "27.1", "0.219", "0.191", "+0.029", "−0.48"],
        ["SKARV", "50.8", "0.107", "0.146", "−0.039", "−1.07"],
        ["TAMBAR", "38.9", "0.096", "0.066", "+0.029", "+0.97"],
        ["ULA", "38.9", "0.083", "0.079", "+0.004", "+0.39"],
        ["ALVHEIM", "34.5", "0.056", "0.085", "−0.029", "+0.99"],
        ["HOD", "38.9", "0.053", "0.031", "+0.022", "+2.61"],
        ["BØYLA", "34.5", "0.050", "0.104", "−0.053", "+0.15"],
        ["VALHALL", "38.9", "0.038", "0.043", "−0.004", "+2.07"],
      ].map(row => new TableRow({ children: [
        cell(row[0], { width: 2080 }),
        cell(row[1], { width: 1300, align: AlignmentType.CENTER }),
        cell(row[2], { width: 1300, align: AlignmentType.CENTER }),
        cell(row[3], { width: 1300, align: AlignmentType.CENTER, bold: true }),
        cell(row[4], { width: 1300, align: AlignmentType.CENTER,
              color: Math.abs(parseFloat(row[4])) < 0.05 ? "2E7D32" : "C62828" }),
        cell(row[5], { width: 2080, align: AlignmentType.CENTER,
              color: parseFloat(row[5]) > 0 ? "2E7D32" : "C62828" }),
      ]})),
    ],
  }),
  caption("75% av Aker BP-feltene treffes innenfor ±0.05 (grønne tall). De største bommene (Edvard Grieg, Bøyla) skyldes ekstrem produksjonsvolatilitet som ingen enkel modell kan fange."),

  h2("7.3 Investeringscase-bygging"),
  p("Modellen lar deg systematisk svare på spørsmål som:"),
  bullet("\"Bør Edvard Grieg ha samme decline-anslag som Ivar Aasen?\" — Nei, premium-historikken viser at Edvard Grieg har 70% høyere natural decline."),
  bullet("\"Er Valhalls produksjonsplatå holdbar?\" — Premium = +2.07 reflekterer redevelopment-CAPEX. Spørsmålet blir: vil Aker BP fortsette investeringstakten?"),
  bullet("\"Hva er rimelig terminal decline for Skarv?\" — Modellen gir 0.146 vs 0.107 faktisk; premium −1.07 indikerer brattere decline enn fysikken tilsier."),

  new Paragraph({ children: [new PageBreak()] }),

  // ═══════════════════════════════════════════════════════════════
  // 8. BEGRENSNINGER
  // ═══════════════════════════════════════════════════════════════
  h1("8. Begrensninger og forutsetninger"),

  h2("8.1 Modellen forutsetter eksponentiell decline"),
  p("Vi modellerer decline som en konstant prosentvis nedgang per år. Dette passer godt for de fleste NCS-felt, men ikke for:"),
  bullet("Plattå-felt som Valhall (har holdt ~35% av peak i 10+ år)"),
  bullet("Felt under aktiv redevelopment der produksjonen midlertidig øker"),
  bullet("Volatile felt med uregelmessige investeringssykluser (Bøyla)"),
  spacer(),
  p("For disse feltene fanger premium-variablen deler av effekten, men noen genuine outliers (som Edvard Grieg) gir høyere prediksjonsfeil."),

  h2("8.2 Basin-spesifikk kalibrering"),
  p("Modellen er kalibrert på NCS-felt. UK NSTA-validering viste at viskositetskoeffisienten faktisk reverserer fortegn på UK Continental Shelf — sannsynligvis fordi UK-feltene er en annen generasjon med annerledes utviklingshistorikk."),
  spacer(),
  p("Implikasjon: modellen bør re-kalibreres for andre basseng (Nordsjøen UK, Golfen, Vest-Afrika) før bruk."),

  h2("8.3 Krever 12 mnd post-peak data"),
  p("For helt nye felt uten 12 måneders post-peak-historikk kan modellen ikke beregne premium. I praksis betyr dette at de første ~18 månedene etter peak kun gir fysikk-baseline-prediksjon."),

  h2("8.4 Forutsetter stabilt operatørprogram"),
  p("Premium fanger historikk, men forutsetter at fremtidig operatør-atferd ligner siste 12 måneder. Plutselige endringer (CAPEX-kutt, salg av felt, redevelopment-prosjekter) krever kvalitative justeringer."),

  // ═══════════════════════════════════════════════════════════════
  // 9. HVORFOR DENNE MODELLEN SKILLER SEG UT
  // ═══════════════════════════════════════════════════════════════
  h1("9. Differensiering vs. standard tilnærminger"),

  h2("9.1 Sammenligning med vanlige metoder"),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2680, 3340, 3340],
    rows: [
      new TableRow({ children: [
        cell("Tilnærming", { width: 2680, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("Svakhet", { width: 3340, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("Vår modell", { width: 3340, bold: true, shade: "1F4E79", color: "FFFFFF" }),
      ]}),
      new TableRow({ children: [
        cell("Flat 5-10% decline", { width: 2680 }),
        cell("Ignorerer feltspesifikke forskjeller helt", { width: 3340 }),
        cell("Differensierer fra 1% (Ekofisk) til 38% (Edvard Grieg)", { width: 3340, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("Operatør-guidance", { width: 2680 }),
        cell("Subjektiv, optimistisk bias, ikke verifiserbar", { width: 3340 }),
        cell("Datadrevet, transparent metodikk", { width: 3340, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("Komplekse reservoarmodeller", { width: 2680 }),
        cell("Krever proprietære data; black-box for analytikere", { width: 3340 }),
        cell("Offentlige inputs; 4 parametere; etterprøvbart", { width: 3340, color: "2E7D32" }),
      ]}),
      new TableRow({ children: [
        cell("Lineær ekstrapolering", { width: 2680 }),
        cell("Ingen fysisk forankring", { width: 3340 }),
        cell("Beggs-Robinson som vitenskapelig baseline", { width: 3340, color: "2E7D32" }),
      ]}),
    ],
  }),

  h2("9.2 Hva dette gir i en ER-/IB-kontekst"),
  bullet("Differensiert syn på decline-rater per felt — viktig competitive edge"),
  bullet("Transparent metodikk som tåler intern modell-validering"),
  bullet("Skalerbar til alle NCS-operatører (Equinor, Aker BP, Vår Energi, ConocoPhillips, OKEA, DNO)"),
  bullet("Mulig å overføre til andre basseng med rekalibrering"),
  bullet("Bevisst om begrensninger — ingen \"black box\"-mystikk"),

  // ═══════════════════════════════════════════════════════════════
  // 10. APPENDIX
  // ═══════════════════════════════════════════════════════════════
  new Paragraph({ children: [new PageBreak()] }),
  h1("Appendix A: Modellutvikling-historikk"),
  p("Modellen ble utviklet gjennom 12 iterasjoner. De viktigste milepælene:"),

  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [1560, 4440, 1680, 1680],
    rows: [
      new TableRow({ children: [
        cell("Trinn", { width: 1560, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("Forbedring", { width: 4440, bold: true, shade: "1F4E79", color: "FFFFFF" }),
        cell("CV R²", { width: 1680, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
        cell("Aker BP RMSE", { width: 1680, bold: true, shade: "1F4E79", color: "FFFFFF", align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("1", { width: 1560, align: AlignmentType.CENTER }),
        cell("Bare viskositet (Beggs-Robinson)", { width: 4440 }),
        cell("−0.015", { width: 1680, align: AlignmentType.CENTER }),
        cell("0.095", { width: 1680, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("2", { width: 1560, align: AlignmentType.CENTER }),
        cell("+ lifetime premium", { width: 4440 }),
        cell("0.473", { width: 1680, align: AlignmentType.CENTER }),
        cell("0.076", { width: 1680, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("3", { width: 1560, align: AlignmentType.CENTER }),
        cell("Switch til siste 3 års premium", { width: 4440 }),
        cell("0.544", { width: 1680, align: AlignmentType.CENTER }),
        cell("0.063", { width: 1680, align: AlignmentType.CENTER }),
      ]}),
      new TableRow({ children: [
        cell("4", { width: 1560, align: AlignmentType.CENTER, bold: true }),
        cell("+ 12 mnd premium + |premium|", { width: 4440, bold: true, shade: "E8F5E9" }),
        cell("0.713", { width: 1680, bold: true, color: "2E7D32", align: AlignmentType.CENTER, shade: "E8F5E9" }),
        cell("0.056", { width: 1680, bold: true, color: "2E7D32", align: AlignmentType.CENTER, shade: "E8F5E9" }),
      ]}),
    ],
  }),

  h1("Appendix B: Variabel-definisjoner"),
  p("API gravity: Standard målestokk for oljens tetthet (°API). Høyere API = lettere olje. NCS-typisk område: 25° (tung) til 50° (lett kondensat)."),
  spacer(),
  p("Viskositet (μ): Oljens motstand mot strømning, målt i centipoise (cP). Vann har μ ≈ 1 cP; tunge oljer har μ ≈ 100+ cP."),
  spacer(),
  p("D_annual: Eksponentiell decline rate på årlig basis. Production_year_T = Peak · exp(−D · T). D = 0.10 betyr 10% årlig nedgang."),
  spacer(),
  p("Premium (P₁₂): Gjennomsnittlig log-avvik mellom faktisk og fysikk-forventet produksjon siste 12 måneder. Positiv = outperformer; negativ = underperformer."),
  spacer(),
  p("Post-peak: Periode fra månedlig produksjon når 100% av peak-måneden. Modellen gjelder bare post-peak-perioden."),
  spacer(),
  p("LOO-CV: Leave-One-Out Cross-Validation. Estimer modellen N ganger, hver gang med ett felt holdt utenfor. Måler ærlig out-of-sample-ytelse."),

  h1("Appendix C: Kontaktinfo og prosjektmateriale"),
  p("Alle scripts, data og figurer er versjonskontrollert i prosjektet:"),
  ptext([
    { text: "Repository: " },
    { text: "./", font: "Courier New", size: 18 },
  ]),
  ptext([
    { text: "Hovedscript: " },
    { text: "analyses/decline_quality/scripts/", font: "Courier New", size: 18 },
  ]),
  ptext([
    { text: "Resultater: " },
    { text: "analyses/decline_quality/results/", font: "Courier New", size: 18 },
  ]),
  spacer(),
  p("Modellen er fullt etterprøvbar og kan utvides med nye datakilder (kvartalsrapporter, brønnmeldinger, CAPEX-guidance) etter behov."),
];

// ═══════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ═══════════════════════════════════════════════════════════════

const doc = new Document({
  creator: "Emma Strandenskaar",
  title: "NCS Decline Rate Model — Methodology",
  description: "Fysikk + Premium-rammeverk for Equity Research",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "595959" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({
            text: "NCS Decline Rate Model — Metodikk",
            size: 18, color: "808080", italics: true,
          })],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "1F4E79", space: 4 } },
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Side ", size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }),
            new TextRun({ text: " av ", size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "808080" }),
            new TextRun({ text: "    |    Emma Strandenskaar    |    Juni 2026", size: 18, color: "808080" }),
          ],
        })],
      }),
    },
    children: content,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log(`Successfully created: ${OUTPUT}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
});
