# Wynik treningu: BERT — BioASQ (Z pretreningiem SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 72 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BERT-base-uncased (google-bert/bert-base-uncased) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | `results/squad/bert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 45.21% | 63.36% |
| **Test** | **50.00%** | **65.65%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 72.5 s |
| train_loss (średni) | 0.812 |
| train_samples_per_second | 50.41 |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Dev EM | 30.14% | 45.21% | +15.07pp |
| Dev F1 | 47.21% | 63.36% | +16.15pp |
| Test EM | 34.15% | 50.00% | +15.85pp |
| Test F1 | 50.24% | 65.65% | +15.41pp |

## Artefakty

- Model: `results/bioasq/bert/best_model/`
- Metryki: `results/bioasq/bert/results.json`
- Predykcje: `results/bioasq/bert/{eval,test}_predictions.json`
