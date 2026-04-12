# Wynik treningu: BERT — COVID-QA 5-fold CV (Bez pretreningu SQuAD)

Data: 2026-04-08
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BERT-base-uncased (google-bert/bert-base-uncased) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM, bez SQuAD) |

## Wyniki per fold

| Fold | Eval EM | Eval F1 | Test EM | Test F1 |
|------|---------|---------|---------|---------|
| 0 | 21.80% | 42.72% | 20.13% | 39.54% |
| 1 | 19.83% | 38.70% | 18.48% | 40.06% |
| 2 | 20.70% | 39.92% | 20.46% | 40.27% |
| 3 | 27.41% | 48.00% | 19.80% | 41.34% |
| 4 | 19.53% | 44.63% | 18.81% | 37.10% |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **19.54%** | **±0.87** |
| **F1** | **39.66%** | **±1.52** |

## Artefakty

- Wyniki per fold: `results/covidqa/bert_no_squad/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/bert_no_squad/fold_{0-4}/{eval,test}_predictions.json`
