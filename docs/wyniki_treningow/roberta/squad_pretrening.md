# Wynik treningu: RoBERTa — pretrening na SQuAD v1.1

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 44 min 47 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | RoBERTa-base (FacebookAI/roberta-base) |
| Parametry (total) | 124,056,578 |
| Parametry (trainable) | 124,056,578 (100% — full fine-tuning) |
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

| Metryka | Wynik | Oczekiwane (Liu et al., 2019) |
|---------|-------|-------------------------------|
| **Exact Match** | 86.15% | ~83.9% |
| **F1** | 92.27% | ~91.5% |

Wynik powyżej referencji. RoBERTa osiągnął najwyższe metryki spośród
wszystkich 6 modeli na SQuAD — co jest spójne z jego lepszą strategią
pretreningu (więcej danych, dłuższy trening, dynamic masking).

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (średni) | 0.952 |
| train_runtime | 2687.4 s |
| train_samples_per_second | 65.91 |
| train_steps_per_second | 4.12 |
| total_flos | 3.47e+16 |

## Uwaga techniczna

Tokenizer RoBERTa (BPE) generuje więcej sub-tokenów niż WordPiece (BERT)
dla niektórych pytań. Wymusiło to dodanie pre-truncation pytań w `data.py`
(limit: max_seq_length // 2 = 192 tokeny) — dotyczy pojedynczych przykładów
w SQuAD i nie wpływa na wyniki.

## Porównanie z literaturą

| Model | EM | F1 | Źródło |
|-------|-----|-----|--------|
| RoBERTa-base (Liu et al.) | 83.9 | 91.5 | Liu et al. (2019) |
| **RoBERTa-base (nasz wynik)** | **86.1** | **92.3** | Niniejsza praca |

Różnica +0.8pp F1 względem referencji. Możliwe wyjaśnienie: nowsza wersja
wag na HuggingFace Hub lub różnice w postprocessingu predykcji.

## Artefakty

- Model: `results/squad/roberta/best_model/`
- Metryki: `results/squad/roberta/results.json`
- Predykcje: `results/squad/roberta/eval_predictions.json`
- TensorBoard: `results/squad/roberta/runs/`
