# Data

All data in this project comes from **public sources**. Raw vendor files are
included for reproducibility; they remain the property of their publishers.

## Sources

| Source | What | Used for |
|--------|------|----------|
| **Sokkeldirektoratet (Sodir/NPD)** | Monthly field production, DST wellbore tests, field geometry (GeoJSON) | Decline model, fluid library, interactive map |
| **Petroleumsprisrådet** | Official normpris differentials | Price-model training target |
| **Equinor / ExxonMobil / TotalEnergies** | Published crude oil assays | Crude-quality features (API, sulfur, distillation, metals) |
| **EIA (U.S. Energy Information Administration)** | Refinery utilisation, stocks, crack spreads, product demand | Market-regime controls |
| **Aker BP ASA quarterly reports** | Reported realised liquids price | Out-of-sample validation |

## Layout

```
data/
├── raw/         External inputs. Re-fetchable via src/1_data_ingestion/ scripts.
│                (Several large folders are git-ignored; run the fetch scripts to rebuild.)
└── processed/   Model outputs: trained models (.pkl/.json), predictions (.csv), figures (.png).
```

Crude assays published by Equinor, ExxonMobil and TotalEnergies are reproduced
here under their public availability for non-commercial, academic use. All
analysis, code and derived results are the author's own work.
