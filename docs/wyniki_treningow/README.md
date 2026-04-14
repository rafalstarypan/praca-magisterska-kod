# Wyniki treningów — indeks

Każdy podkatalog odpowiada jednemu modelowi. Raporty zawierają pełne metryki,
hiperparametry, dane z TensorBoard i porównania z literaturą.

## Struktura

```
wyniki_treningow/
├── README.md                          ← ten plik
├── bert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=81.1, F1=88.3)
│   ├── bioasq_bez_pretreningu.md      ← BioASQ bez SQuAD (Test F1=50.2)
│   ├── bioasq_z_pretreningiem.md      ← BioASQ z SQuAD (Test F1=65.7)
│   ├── covidqa_bez_pretreningu.md     ← COVID-QA bez SQuAD (Test F1=39.7)
│   └── covidqa_z_pretreningiem.md     ← COVID-QA z SQuAD (Test F1=55.5)
├── biobert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=81.4, F1=88.7)
│   ├── bioasq_bez_pretreningu.md      ← BioASQ bez SQuAD (Test F1=61.8)
│   ├── bioasq_z_pretreningiem.md      ← BioASQ z SQuAD (Test F1=69.2)
│   ├── covidqa_bez_pretreningu.md     ← COVID-QA bez SQuAD (Test F1=45.0)
│   └── covidqa_z_pretreningiem.md     ← COVID-QA z SQuAD (Test F1=57.1)
├── pubmedbert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=82.2, F1=89.6)
│   ├── bioasq_bez_pretreningu.md      ← BioASQ bez SQuAD (Test F1=71.4)
│   ├── bioasq_z_pretreningiem.md      ← BioASQ z SQuAD (Test F1=73.5)
│   ├── covidqa_bez_pretreningu.md     ← COVID-QA bez SQuAD (Test F1=54.7)
│   ├── covidqa_z_pretreningiem.md     ← COVID-QA z SQuAD (Test F1=62.8)
│   ├── bioasq_lora_r8.md             ← E2: BioASQ LoRA r=8 (Test F1=73.4)
│   ├── covidqa_lora_r8.md            ← E2: COVID-QA LoRA r=8 (oczekiwany)
│   ├── bioasq_lora_r4.md             ← E2: BioASQ LoRA r=4 (oczekiwany)
│   └── bioasq_lora_r16.md            ← E2: BioASQ LoRA r=16 (oczekiwany)
├── roberta/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=86.1, F1=92.3)
│   ├── bioasq_bez_pretreningu.md      ← BioASQ bez SQuAD (Test F1=62.3)
│   ├── bioasq_z_pretreningiem.md      ← BioASQ z SQuAD (Test F1=69.0)
│   ├── covidqa_bez_pretreningu.md     ← COVID-QA bez SQuAD (Test F1=53.4)
│   ├── covidqa_z_pretreningiem.md     ← COVID-QA z SQuAD (Test F1=61.5)
│   ├── bioasq_lora_r8.md             ← E2: BioASQ LoRA r=8 (oczekiwany)
│   └── covidqa_lora_r8.md            ← E2: COVID-QA LoRA r=8 (oczekiwany)
├── distilbert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=77.4, F1=85.6)
│   ├── bioasq_bez_pretreningu.md      ← BioASQ bez SQuAD (Test F1=38.9)
│   ├── bioasq_z_pretreningiem.md      ← BioASQ z SQuAD (Test F1=60.8)
│   ├── covidqa_bez_pretreningu.md     ← COVID-QA bez SQuAD (Test F1=35.0)
│   └── covidqa_z_pretreningiem.md     ← COVID-QA z SQuAD (Test F1=52.0)
└── clinicalbert/
    ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=76.8, F1=85.4)
    ├── bioasq_bez_pretreningu.md      ← BioASQ bez SQuAD (Test F1=50.7)
    ├── bioasq_z_pretreningiem.md      ← BioASQ z SQuAD (Test F1=65.9)
    ├── covidqa_bez_pretreningu.md     ← COVID-QA bez SQuAD (Test F1=39.4)
    └── covidqa_z_pretreningiem.md     ← COVID-QA z SQuAD (Test F1=56.8)
```

## Konwencja nazewnictwa plików

### E1 (Macierz porównawcza)
- `squad_pretrening.md` — wynik pretreningu na SQuAD v1.1 (faza 1)
- `bioasq_z_pretreningiem.md` — fine-tuning na BioASQ z wagami z fazy 1
- `bioasq_bez_pretreningu.md` — fine-tuning na BioASQ bezpośrednio z HuggingFace
- `covidqa_z_pretreningiem.md` — COVID-QA 5-fold CV z wagami z fazy 1
- `covidqa_bez_pretreningu.md` — COVID-QA 5-fold CV bezpośrednio z HuggingFace

### E2 (Full FT vs LoRA)
- `bioasq_lora_r8.md` — LoRA r=8 na BioASQ
- `covidqa_lora_r8.md` — LoRA r=8 na COVID-QA (5-fold CV)
- `bioasq_lora_r4.md` — LoRA r=4 ablacja (tylko PubMedBERT)
- `bioasq_lora_r16.md` — LoRA r=16 ablacja (tylko PubMedBERT)

## Źródła danych w raportach

- `results.json` — metryki końcowe (EM, F1, train loss, parametry, hardware)
- TensorBoard events — pełne krzywe train/loss i eval/F1 co krok
- Logi terminala — dla treningów bez TensorBoard (SQuAD pretrening)
