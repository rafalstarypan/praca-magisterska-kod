# Wynik treningu: ClinicalBERT — BioASQ (Z pretreningiem SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 62 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | `results/squad/clinicalbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 45.21% | 65.45% |
| **Test** | **46.34%** | **65.90%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 62.0 s |
| train_loss (średni) | 0.766 |
| train_samples_per_second | 59.04 |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Dev EM | 35.62% | 45.21% | +9.59pp |
| Dev F1 | 50.35% | 65.45% | +15.10pp |
| Test EM | 35.37% | 46.34% | +10.98pp |
| Test F1 | 50.70% | 65.90% | +15.20pp |

## Artefakty

- Model: `results/bioasq/clinicalbert/best_model/`
- Metryki: `results/bioasq/clinicalbert/results.json`
- Predykcje: `results/bioasq/clinicalbert/{eval,test}_predictions.json`
