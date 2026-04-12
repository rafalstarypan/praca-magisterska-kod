# Wynik treningu: RoBERTa — BioASQ (Z pretreningiem SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 64 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | RoBERTa-base (FacebookAI/roberta-base) |
| Parametry (total) | 124,056,578 |
| Parametry (trainable) | 124,056,578 (100% — full fine-tuning) |
| Źródło wag | `results/squad/roberta/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 56.16% | 71.94% |
| **Test** | **50.00%** | **69.03%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 64.1 s |
| train_loss (średni) | 0.825 |
| train_samples_per_second | 56.98 |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Dev EM | 42.47% | 56.16% | +13.70pp |
| Dev F1 | 59.64% | 71.94% | +12.30pp |
| Test EM | 42.68% | 50.00% | +7.32pp |
| Test F1 | 62.30% | 69.03% | +6.73pp |

## Artefakty

- Model: `results/bioasq/roberta/best_model/`
- Metryki: `results/bioasq/roberta/results.json`
- Predykcje: `results/bioasq/roberta/{eval,test}_predictions.json`
