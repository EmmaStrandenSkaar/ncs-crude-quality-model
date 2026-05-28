# Aker BP — Quality-Driven Valuation & M&A Framework

**Prosjekt-eier:** Emma Strandenskaar
**Formål:** Søknadsanalyse for sommer-internship, equity research / investment banking
**Tidshorisont:** 6-8 uker (mai-juli 2026)
**Status:** Fase 1 påbegynt

---

## Investment thesis (working draft)

> Aker BP handler til ~19x EV/2P — samme nivå som ved Hess Norge (2017) og Lundin (2021).
> Markedet undervurderer at AKRBP's NCS-portefølje genererer en strukturell **kvalitetspremium**
> mot Brent som er kvantifiserbar via destillasjonsutbytter, logistikk og refining-økonomi.
> Vi modellerer denne premien, anvender den på forward Brent, og rangerer M&A-kandidater
> etter "quality-adjusted accretion" — kombinert reservepris-rabatt og porteføljekvalitet.

---

## Integrasjon med eksisterende arbeid

| Eksisterende script | Hva det gjør | Integrasjon med kvalitetsmodellen |
|---------------------|--------------|------------------------------------|
| `13_pre_earnings_aker_bp.py` | Pre-Q2 2026 prising, Hormuz-premium | Erstatte Hormuz-proxy med modell-predikert NCS-premium |
| `14_aker_bp_ma_window.py` | EV/2P-vinduanalyse, Citi-tesen | Beholdes som intro / markedsetting |
| `15_aker_bp_target_universe.py` | M&A-kandidater på NCS | Legge til "quality-adjusted accretion" per target |
| `16-18_post_deal_performance` | Historiske deals + backtest | Beholdes som validering |
| `33-34_extended/parsimonious_model` | Differensial-modellen | Kjerne i den nye analysen |

---

## Faseinndeling og leveranser

### Fase 1 — Data foundation (uke 1-2)
**Mål:** Bytte estimerte assay-verdier med verifiserte verdier fra primærkilder.

- [ ] Skrape Equinor crude assays (PDF-er) for NCS-felt:
  Johan Sverdrup, Troll, Oseberg, Statfjord, Gullfaks, Heidrun, Grane, Åsgard, Norne
- [ ] Hente BP/Shell-assays for internasjonale benchmarks:
  Brent, Forties, Mars, Arab Light, Bonny Light
- [ ] Spore Aker BP-opererte felt:
  Alvheim, Edvard Grieg, Ivar Aasen, Valhall, Hod, Skarv, Ula
- [ ] Bygge `verified_crude_assays.csv` med eksplisitt source URL + dato per rad
- [ ] Hente Aker BP field-by-field produksjonshistorikk fra Sodir (har allerede)
- [ ] Hente Aker BP kvartals-finansielle (realisert pris, OPEX, produksjon)
- [ ] Samle sell-side konsensus (Bloomberg / publiske rapporter) for Q3'26-Q4'27

**Leveranse:** Verifisert dataset + datavedlikeholds-script

### Fase 2 — Modell-refinement (uke 3-4)
- [ ] Re-kjøre regresjon med verifiserte assays — sammenligne koeffisienter
- [ ] Bygge **Aker BP realized-price decomposition**:
  ```
  Realized = Brent + Σ(production_share_i × differential_i)
  ```
  Hvor differential_i kommer fra modellen per felt.
- [ ] Validere modell mot **rapportert** realisert pris siste 8 kvartaler
- [ ] Tracking error analyse — hvor mye av Aker BPs realisering forklarer modellen?

**Leveranse:** Realized-price tracker (faktisk vs predikert per kvartal)

### Fase 3 — Forward-prediksjon (uke 4-5)
- [ ] Forward Brent curve (ICE)
- [ ] Aker BPs produksjonsguiding per felt (fra Q-rapporter + CMD)
- [ ] Modell forward differensial per felt → forward realized price
- [ ] Bygge revenue waterfall Q3'26-Q4'28
- [ ] Yggdrasil/NOAKA mix-shift effekt (production starts 2027)
- [ ] Sammenligne mot sell-side konsensus

**Leveranse:** Forward realized price chart + revenue/EBITDA-prognose vs konsensus

### Fase 4 — M&A-integrasjon (uke 5-6)
- [ ] Per kandidat fra `15_aker_bp_target_universe`:
  - Kalkulere implisitt porteføljepremium/-rabatt
  - "Quality-adjusted accretion" = EV/2P-rabatt + porteføljekvalitet-løft
- [ ] Pro-forma realisert pris hvis AKRBP gjør deal X
- [ ] Ranking: hvilke targets er strategisk verdt fokus utover ren EV/2P-akkresjon?
- [ ] Historisk validering: gjorde Hess Norge / Lundin AKRBPs portefølje bedre eller dårligere?

**Leveranse:** M&A ranking-matrise (quality fit × EV accretion × execution risk)

### Fase 5 — Valuation (uke 6-7)
- [ ] DCF med quality-adjusted realisering:
  - Replace flat realized-Brent diff med modell-predikert per felt per kvartal
  - WACC, terminalverdi, declining-curves
- [ ] SOTP per felt (NPV-bidrag per asset)
- [ ] Sensitivitet: Brent-pris × differensial-scenarier × NOAKA-timing

**Leveranse:** Valuation table med price target og confidence interval

### Fase 6 — Deliverable (uke 7-8)
- [ ] **One-pager** (investment summary) — den viktigste fila
- [ ] **Full ER-rapport** (15-20 sider): thesis, framework, valuation, risks
- [ ] **Pitch deck** (10-12 slides) — for muntlig presentasjon i intervju
- [ ] **GitHub README** med metode + reproducibility-instruksjoner
- [ ] **Appendix:** dataquellt, modell-spesifikasjoner, robusthet-tester

**Leveranse:** Internship-klar pakke (1 PDF + 1 deck + GitHub link)

---

## Suksesskriterier

For at dette skal *imponere skikkelig* i en internship-søknad:

1. **Sluttproduktet ser ut som ekte ER** — ikke et skoleprosjekt
2. **Alle datapoint er sporbare** — kilde og dato for hver tall
3. **Modellen brukes til å si noe handlbart** — buy/sell-konklusjon med tall
4. **Norsk dybde + internasjonal kontekst** — viser at du forstår NCS-spesifikt
5. **Kvantitative funn er forklart kvalitativt** — viser at du tenker som analytiker

---

## Foreløpig differensial-modell (oppdateres fortløpende)

Per nå (M10/M34 parsimonious):
- N = 3 421 observasjoner, 43 grades, 38 features
- In-sample R² = 0.694, CV R² = 0.683, **OOT R² = 0.578**
- RMSE = 4.28 USD/fat (MAE = 3.01 USD/fat)
- Kjente svakheter: assay-data delvis estimert (Fase 1 adresserer dette)
