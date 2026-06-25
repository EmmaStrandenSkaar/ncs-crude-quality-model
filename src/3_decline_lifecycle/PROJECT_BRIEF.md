# Nytt prosjekt: Decline rate vs. oljekvalitet

## Forskningsspørsmål

**Er det en sammenheng mellom oljekvalitet (assay-egenskaper) og hvor raskt felt
produksjonen faller måned-over-måned?**

Konkret: kan API, sulfur, CCR, vacuum residue, V+Ni, og andre kvalitets-features
forklare variasjon i månedlige decline rates på tvers av NCS-felt?

## Hypoteser å teste

1. **Tunge crudes (lav API) deklinerer raskere** — høyere viskositet betyr mer
   resistens i reservoaret når trykket faller
2. **Sour crudes (høy svovel) har raskere decline** — kjemisk korrosjon på
   produksjons-utstyr → flere shut-downs
3. **Høy CCR/Metaller** — proxy for komplekse reservoarer som er vanskelige
   å produsere optimalt
4. **Interaksjon med felt-alder** — kvalitets-effekten øker over tid (modne
   felt med vanskelig olje sliter mer)
5. **GOR (gas-oil ratio)** — gass-rike felt har annet decline-mønster

## Tilgjengelig data (i samme repo)

### Produksjon (Sodir)
- `data/raw/sodir/sodir_field_production_monthly.csv`
- ~25 års månedlig produksjon per felt (1985-2026)
- Kolonner: prfYear, prfMonth, prfPrdOilNetMillSm3, prfPrdGasNetBillSm3,
  prfInformationCarrier (feltnavn), m.fl.

### Kvalitet (assays)
- `data/processed/unified_crude_assays.csv` — 170 grades med API, sulfur,
  vacuum residue, CCR, V+Ni, naphtha/kerosene/diesel/gasoil yields
- Equinor lab-assays for de 21 viktigste NCS-feltene
- Mapping: Sodir-feltnavn → assay-grade (se script 49 i prosjektet for
  `DIRECT_ASSAY` og `BLEND_PROXY` dicts)

### Felt-metadata (Sodir geo)
- `data/raw/sodir_geo/fields.geojson` — 142 felt med:
  - First discovery year (fldDiscoveryYear)
  - Status (Producing/Shut down/Approved)
  - Operatør (cmpLongName)
  - Hovedområde (North Sea/Norwegian Sea/Barents Sea)
  - Geometri (sentroid kan beregnes for distanse)

### Markedsdata (kan kontrolleres for)
- `data/processed/regression_panel.csv` — Brent-priser, crack spreads,
  refinery util etc. per måned (4735 obs)
- Useful som controls hvis prisene påvirker decline (lave priser → produsenter
  reduserer produksjon i lave felt)

## Bakgrunn — hva eksisterer allerede

Vi har bygget en omfattende **kvalitets-prismodell** for NCS-crudes
(Brent-linked OLS, OOT R²=0.38, RMSE 2.78 USD/bbl) som forklarer hvordan
kvalitet driver pris-differensial vs. Dated Brent. Den modellen ligger i
hovedrepoet og skal IKKE endres av dette nye prosjektet.

Decline-analysen er en **separat studie** som ser på et helt annet spørsmål:
hvordan påvirker kvalitet selve produksjons-banen (ikke prisen).

## Foreslått tilnærming (du kan endre)

### Fase 1: Datapreparering
1. Last Sodir månedlig produksjon
2. Beregn månedlig decline rate per felt: (prod_t - prod_(t-1)) / prod_(t-1)
3. Filter til kun PRODUSERENDE felt (ikke nedlagte)
4. Behandle outliers (shut-down-måneder, oppstart, vedlikehold)
5. Join med assay-data (via Sodir→assay mapping)

### Fase 2: Eksplorerende analyse
1. Scatter: API vs. snitt decline rate per felt
2. Scatter: sulfur vs. snitt decline rate
3. Heatmap: alle kvalitets-features vs. decline metrics
4. Distribusjon: er decline normalt eller skewed?

### Fase 3: Regresjon
- y = monthly_decline_rate
- X = quality features + field age + controls (Brent price, year)
- Panel-fixed-effects per felt for å fjerne field-specific bias
- Eventuelt: AR(1) struktur (decline er autokorrelert)

### Fase 4: Robusthetstester
- Med vs. uten outliers
- Subperioder (pre/post-2014 oljepris-krasj)
- Per hovedområde (Nordsjø vs Norske Sjø vs Barentshavet)

### Fase 5: Visualisering + tolkning
- En clean graf til presentasjonen din: "kvalitets-decline-sammenheng"
- Forklare økonomiske mekanismer

## Output-konvensjon

Hold alt nytt i `analyses/decline_quality/`:
```
analyses/decline_quality/
├── scripts/
│   ├── 01_compute_decline_rates.py
│   ├── 02_explore_quality_decline.py
│   ├── 03_panel_regression.py
│   └── 04_visualization.py
├── data/
│   └── (processed outputs)
├── results/
│   └── (figures, tables)
└── PROJECT_BRIEF.md (denne filen)
```

På den måten påvirker det aldri hovedmodellen.

## Klar til å starte?

Start gjerne med å lese denne filen, deretter inspisere Sodir-data og
assay-data for å forstå strukturen. Foreslå en konkret plan før vi koder.
