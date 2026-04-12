# Wyniki treningów — indeks

Każdy podkatalog odpowiada jednemu modelowi. Raporty zawierają pełne metryki,
hiperparametry, dane z TensorBoard i porównania z literaturą.

## Struktura

```
wyniki_treningow/
├── README.md                          ← ten plik
├── distilbert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=77.4, F1=85.6)
│   ├── bioasq_bez_pretreningu.md      ← Baseline: BioASQ bez SQuAD (Test F1=38.9)
│   └── bioasq_z_pretreningiem.md      ← Faza 2: BioASQ z SQuAD (Test F1=60.8)
├── bert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=81.1, F1=88.3)
│   ├── bioasq_bez_pretreningu.md      ← Baseline: BioASQ bez SQuAD (Test F1=50.2)
│   └── bioasq_z_pretreningiem.md      ← Faza 2: BioASQ z SQuAD (Test F1=65.7)
├── biobert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=81.4, F1=88.7)
│   ├── bioasq_bez_pretreningu.md      ← Baseline: BioASQ bez SQuAD (Test F1=61.8)
│   └── bioasq_z_pretreningiem.md      ← Faza 2: BioASQ z SQuAD (Test F1=69.2)
├── pubmedbert/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=82.2, F1=89.6)
│   ├── bioasq_bez_pretreningu.md      ← Baseline: BioASQ bez SQuAD (Test F1=71.4)
│   └── bioasq_z_pretreningiem.md      ← Faza 2: BioASQ z SQuAD (Test F1=73.5)
├── roberta/
│   ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=86.1, F1=92.3)
│   ├── bioasq_bez_pretreningu.md      ← Baseline: BioASQ bez SQuAD (Test F1=62.3)
│   └── bioasq_z_pretreningiem.md      ← Faza 2: BioASQ z SQuAD (Test F1=69.0)
└── clinicalbert/
    ├── squad_pretrening.md            ← Faza 1: SQuAD v1.1 (EM=76.8, F1=85.4)
    ├── bioasq_bez_pretreningu.md      ← Baseline: BioASQ bez SQuAD (Test F1=50.7)
    └── bioasq_z_pretreningiem.md      ← Faza 2: BioASQ z SQuAD (Test F1=65.9)
```

## Konwencja nazewnictwa plików

- `squad_pretrening.md` — wynik pretreningu na SQuAD v1.1 (faza 1)
- `bioasq_z_pretreningiem.md` — fine-tuning na BioASQ z wagami z fazy 1
- `bioasq_bez_pretreningu.md` — fine-tuning na BioASQ bezpośrednio z HuggingFace
- `covidqa_fold_N.md` — wynik na COVID-QA, fold N
- `covidqa_podsumowanie.md` — agregacja 5 foldów (mean ± std)

## Źródła danych w raportach

- `results.json` — metryki końcowe (EM, F1, train loss, parametry)
- TensorBoard events — pełne krzywe train/loss i eval/F1 co krok
- Logi terminala — dla treningów bez TensorBoard (SQuAD pretrening)
