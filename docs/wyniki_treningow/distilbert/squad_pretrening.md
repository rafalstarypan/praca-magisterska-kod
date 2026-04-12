# Wynik treningu: DistilBERT — pretrening na SQuAD v1.1

Data: 2026-04-05
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 28 min 34 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | DistilBERT (distilbert/distilbert-base-uncased) |
| Parametry (total) | 66,364,418 |
| Parametry (trainable) | 66,364,418 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM) |

## Dataset

| Parametr | Wartość |
|----------|---------|
| Dataset | SQuAD v1.1 (rajpurkar/squad) |
| Train | 87,599 przykładów |
| Dev | 10,570 przykładów |
| Test | Brak (SQuAD test set nie jest publiczny) |
| Typ pytań | Ekstrakcyjne QA (answer span w kontekście) |

## Hiperparametry

| Parametr | Wartość | Źródło |
|----------|---------|--------|
| learning_rate | 3e-5 | Devlin et al. (2019) |
| batch_size | 16 | Devlin et al. (2019), zakres {16, 32} |
| epochs | 2 | Devlin et al. (2019), Appendix A.3 |
| max_seq_length | 384 | Devlin et al. (2019), Section 4.2 |
| doc_stride | 128 | Devlin et al. (2019), Section 4.2 |
| weight_decay | 0.01 | Devlin et al. (2019) |
| warmup_ratio | 0.1 | ~10% warmup, standard BERT |
| fp16 | True | Mixed precision (Micikevicius et al., 2018) |
| optimizer | AdamW | HuggingFace Trainer default |
| lr_scheduler | linear decay with warmup | HuggingFace Trainer default |

## Wyniki

### Metryki na SQuAD v1.1 dev

| Metryka | Wynik | Oczekiwane (Sanh et al., 2019) |
|---------|-------|-------------------------------|
| **Exact Match** | 77.41% | ~77.7% |
| **F1** | 85.55% | ~85.8% |

Wyniki zgodne z literaturą — pipeline działa poprawnie.

### Przebieg treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (końcowy) | 0.87 |
| Train loss (średni) | 1.32 |
| Throughput | 103.3 samples/s |
| Kroki (total) | 11,066 |
| VRAM (peak) | ~3.2 GB |
| GPU utilization | ~94% |

### Krzywa loss (wybrane punkty)

| Epoch | Loss |
|-------|------|
| 0.00 | 5.94 |
| 0.10 | 2.82 |
| 0.20 | 1.81 |
| 0.50 | 1.38 |
| 1.00 | 1.09 |
| 1.50 | 0.91 |
| 2.00 | 0.87 |

## Porównanie z literaturą

| Model | EM | F1 | Źródło |
|-------|-----|-----|--------|
| BERT-base (Devlin et al.) | 80.8 | 88.5 | Devlin et al. (2019) |
| DistilBERT (Sanh et al.) | 77.7 | 85.8 | Sanh et al. (2019) |
| **DistilBERT (nasz wynik)** | **77.4** | **85.6** | Niniejsza praca |

Różnica <0.3pp F1 względem oryginalnego DistilBERT paper.

## Artefakty

- Model: `results/squad/distilbert/best_model/`
- Metryki: `results/squad/distilbert/results.json`
- Predykcje: `results/squad/distilbert/eval_predictions.json`
