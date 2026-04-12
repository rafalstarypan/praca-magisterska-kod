# Wynik treningu: PubMedBERT — COVID-QA 5-fold CV (Z pretreningiem SQuAD)

Data: 2026-04-09
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 37.62% | 63.80% | 3906 MB | 30 min |
| 1 | 35.64% | 60.69% | 3906 MB | 30 min |
| 2 | 37.95% | 61.85% | 3906 MB | 30 min |
| 3 | 36.30% | 64.26% | 3906 MB | 30 min |
| 4 | 37.62% | 63.46% | 3906 MB | 30 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **37.03%** | **±0.96** |
| **F1** | **62.81%** | **±1.49** |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Test EM | 32.01 ± 0.81 | 37.03 ± 0.96 | +5.02pp |
| Test F1 | 54.65 ± 0.72 | 62.81 ± 1.49 | +8.16pp |

## Artefakty

- Wyniki per fold: `results/covidqa/pubmedbert/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/pubmedbert/fold_{0-4}/{eval,test}_predictions.json`
