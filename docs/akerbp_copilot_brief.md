# Brief til Copilot i Excel — oppdater Aker BP-modellen

**Mål:** Oppdater Aker BP NAV/verdsettelsesmodellen med (A) nye feltspesifikke decline-rater
og (B) realiserte feltvise oljepriser, fra en ekstern decline- og prismodell.

Gjør endringene per felt. Behold modellens eksisterende struktur — bare bytt ut/oppdater
decline-rate-inputene og prisinputene med verdiene under. Marker oppdaterte celler med en
kommentar "Oppdatert fra V5.1 decline-modell / realisert-pris-modell, juni 2026".

---

## DEL A — Decline-rater per felt

Bruk **"Effektiv årlig decline"**-kolonnen som den årlige nedgangsraten i produksjonsforecasten.

> **Viktig:** Modellens decline-rate er en *kontinuerlig* eksponentiell rate (D). Den effektive
> årlige nedgangen er `1 − EXP(−D)`. Jeg har allerede regnet om — bruk "Eff. årlig decline"-kolonnen
> direkte. Produksjonsformel: **Produksjon(år t+1) = Produksjon(år t) × (1 − eff_decline)**.
> (Tilsvarer `Produksjon(t) = Peak × EXP(−D × t)` hvis du heller bruker kontinuerlig form.)

| Felt | Aker BP WI | Eff. årlig decline | (kont. D) | Datakvalitet |
|------|-----------:|-------------------:|----------:|--------------|
| Volund | 65 %* | **23.7 %** | 27.0 % | estimert |
| Vilje | 65 %* | **19.6 %** | 21.8 % | estimert |
| Edvard Grieg | 65 % | **18.0 %** | 19.9 % | operatør-data |
| Skogul | 65 % | **17.4 %** | 19.2 % | estimert |
| Ivar Aasen | 34.8 % | **16.2 %** | 17.7 % | estimert |
| Skarv | 23.8 % | **13.8 %** | 14.9 % | direkte assay |
| Bøyla | 65 % | **9.5 %** | 10.0 % | estimert |
| Alvheim | 65 % | **8.1 %** | 8.5 % | direkte assay |
| Ula | 80 % | **8.1 %** | 8.4 % | direkte assay |
| Tambar | 55 % | **7.0 %** | 7.2 % | operatør-data |
| Valhall | 90 % | **5.0 %** | 5.1 % | direkte assay |
| Hod | 90 % | **3.8 %** | 3.9 % | direkte assay |

\* Volund/Vilje: verifiser eierandel (antatt 65 % som Alvheim-området).

**Instruksjon til Copilot:**
1. Finn raden/kolonnen for hvert felts årlige decline-rate i produksjonsforecast-arket.
2. Erstatt med "Eff. årlig decline"-verdien over.
3. Sørg for at produksjonsforecasten bruker formelen `= forrige_år × (1 − decline_rate)`.
4. For felt som ikke er i listen (rene gass-/NGL-felt eller felt utenfor porteføljen): behold eksisterende.

---

## DEL B — Realiserte feltvise oljepriser

Realisert pris per felt = **Brent-antakelse + differensial**. Differensialen er feltets
kvalitetsjusterte prisforskjell mot Dated Brent (USD/bbl), fra realisert-pris-modellen.

| Felt | Differensial vs Brent (USD/bbl) |
|------|--------------------------------:|
| Tambar Øst | **+0.86** |
| Tambar | **+0.66** |
| Ula | **+0.63** |
| Valhall | **+0.62** |
| Hod | **+0.62** |
| Alvheim | **+0.46** |
| Bøyla | **+0.46** |
| Skogul | **+0.44** |
| Ivar Aasen | **−0.16** |
| Edvard Grieg | **−0.17** |
| Johan Sverdrup | **−0.92** |
| Skarv | **−4.21** |

**Instruksjon til Copilot:**
1. I prisforutsetnings-arket: lag/oppdater en kolonne "Differensial vs Brent" per felt med verdiene over.
2. Realisert feltpris-formel: **= Brent_antakelse + Differensial_felt**.
   (Eksempel: hvis Brent-scenario = 75 USD/bbl, blir Valhall = 75 + 0.62 = 75.62; Skarv = 75 − 4.21 = 70.79.)
3. Selskapets blendede realiserte pris = produksjonsvektet snitt:
   **= SUMPRODUKT(felt_produksjon × felt_realisert_pris) / SUM(felt_produksjon)**
   (vekt med netto produksjon = brutto × Aker BP WI).
4. Skarv er kondensat/NGL — stor negativ differensial er korrekt (lett, lav verdi per fat olje-ekvivalent).

---

## DEL C — Forward-felt (Yggdrasil/Wisting) for NAV-upside

Hvis modellen har en upside-/forward-seksjon, bruk disse fra lifecycle-modellen:

| Felt | First oil | Peak (kboe/d) | Eff. årlig decline |
|------|----------:|--------------:|-------------------:|
| Wisting | 2028 | ~164 | ~15.5 % |
| Fulla (Yggdrasil) | 2028 | ~50 | ~32 % |
| Hugin (Yggdrasil) | 2027 | ~41 | ~33 % |
| Munin (Yggdrasil) | 2027 | ~39 | ~33 % |
| Symra | 2029 | ~38 | ~33 % |
| Tyrving | 2025 | ~31 | ~33 % |
| Dvalin | 2027 | ~34 | ~33 % |

Forward-felt bruker en full lifecycle-profil (ramp → platå → decline). Decline-raten over gjelder
*etter* platået. Små subsea-tiebacks (Hugin/Munin/Symra) har høy decline (~33 %) — de tappes raskt.
Prisdifferensial for forward-felt: bruk Alvheim-/Brent-blend-nivå (~+0.4 USD/bbl) inntil egne assays finnes.

---

## Tekniske notater

- **Decline-kilde:** V5.1-modell (nested CV R² = 0.66). "Direkte assay"/"operatør-data" = høyere
  tillit; "estimert" = modell-prediksjon med større usikkerhet.
- **Pris-kilde:** Brent-linket differensial-modell, validert mot Aker BPs rapporterte realiserte
  priser (R² = 0.989, MAE 1.56 USD/bbl).
- **Decline-formen:** eksponentiell. For felt fortsatt i platå-fase, start decline først etter
  platå-slutt (Valhall/Hod holder seg flate lenger — lav decline reflekterer dette).
- **Edvard Grieg:** modellen predikerer 18 % vs observert 38 % — feltet decliner raskere enn
  modellen fanger. Vurder en manuell oppjustering hvis du vil være konservativ.
- **Enheter:** differensialer i USD/bbl; legges til Brent-scenarioet ditt (uansett nivå).

---

*Kildefil med rådata: `docs/akerbp_model_inputs.csv`. Generert fra decline-modell V5.1 og
realisert-pris-modell (script 42), juni 2026.*
