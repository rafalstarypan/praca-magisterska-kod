# Wynik treningu: ClinicalBERT — COVID-QA 5-fold CV (Z pretreningiem SQuAD)

Data: 2026-04-10
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | `results/squad/clinicalbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 31.68% | 57.59% | 3892 MB | 36 min |
| 1 | 33.00% | 55.84% | 3891 MB | 36 min |
| 2 | 31.02% | 55.77% | 3891 MB | 35 min |
| 3 | 32.01% | 55.95% | 3891 MB | 35 min |
| 4 | 35.97% | 58.63% | 3892 MB | 36 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **32.74%** | **±1.94** |
| **F1** | **56.76%** | **±1.29** |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Test EM | 19.21 ± 2.51 | 32.74 ± 1.94 | +13.53pp |
| Test F1 | 39.44 ± 2.16 | 56.76 ± 1.29 | +17.32pp |

### Uwaga

ClinicalBERT zyskuje najwięcej ze wszystkich 6 modeli z pretreningu SQuAD
na COVID-QA (+17.32pp F1). Bez SQuAD model był w dużej mierze bezużyteczny
(F1=39.44), ponieważ jego pretrening na notatkach klinicznych (MIMIC-III)
słabo transferuje na artykuły naukowe z CORD-19. Pretrening na SQuAD
dostarcza ogólnych umiejętności extractive QA, które kompensują ten
domain mismatch.

Mimo dużej poprawy, ClinicalBERT+SQuAD (F1=56.76) nadal zajmuje ostatnie
miejsce wśród 6 modeli z pretreningiem SQuAD na COVID-QA — PubMedBERT
(62.81), RoBERTa (61.45), BioBERT (57.06) i BERT (55.53) pozostają
lepsze lub porównywalne.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 3892 MB |
| Czas per fold | ~35-36 min |
| Batch size (train) | 16 |

## Artefakty

- Wyniki per fold: `results/covidqa/clinicalbert/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/clinicalbert/fold_{0-4}/{eval,test}_predictions.json`
