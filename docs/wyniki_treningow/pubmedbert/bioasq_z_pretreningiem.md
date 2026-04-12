# Wynik treningu: PubMedBERT — BioASQ (Z pretreningiem SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 68 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 57.53% | 71.10% |
| **Test** | **59.76%** | **73.52%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 68.5 s |
| train_loss (średni) | 0.657 |
| train_samples_per_second | 53.33 |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Dev EM | 42.47% | 57.53% | +15.07pp |
| Dev F1 | 64.99% | 71.10% | +6.10pp |
| Test EM | 53.66% | 59.76% | +6.10pp |
| Test F1 | 71.39% | 73.52% | +2.13pp |

### Uwaga

Poprawa z SQuAD (+2.13pp Test F1) jest najniższa spośród wszystkich modeli.
PubMedBERT ma już silny baseline dzięki domain-specific pretreningowi from scratch
na PubMed — dodanie SQuAD daje marginalną poprawę.

## Artefakty

- Model: `results/bioasq/pubmedbert/best_model/`
- Metryki: `results/bioasq/pubmedbert/results.json`
- Predykcje: `results/bioasq/pubmedbert/{eval,test}_predictions.json`
