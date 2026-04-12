# Wynik treningu: ClinicalBERT — pretrening na SQuAD v1.1

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 44 min 38 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (BioBERT + MIMIC-III clinical notes) |

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

| Metryka | Wynik | Oczekiwane |
|---------|-------|------------|
| **Exact Match** | 76.76% | ~83–86% |
| **F1** | 85.41% | ~83–86% |

F1 mieści się w oczekiwanym zakresie. EM niższy niż przewidywany — ClinicalBERT
był dalej pretrenowany na wąskiej domenie (notatki kliniczne MIMIC-III), co
osłabia transfer na ogólny SQuAD. Rozbieżność EM vs F1 sugeruje, że model
częściej daje częściowo poprawne odpowiedzi (wysoki overlap, ale nie exact match).

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (średni) | 1.273 |
| train_runtime | 2678.2 s |
| train_samples_per_second | 66.73 |
| train_steps_per_second | 4.17 |
| total_flos | 3.50e+16 |

## Porównanie z literaturą

| Model | EM | F1 | Źródło |
|-------|-----|-----|--------|
| BioBERT v1.1 (Lee et al.) | 82.1 | 89.1 | Lee et al. (2020) |
| ClinicalBERT (Alsentzer et al.) | — | — | Brak wyników SQuAD w paper |
| **ClinicalBERT (nasz wynik)** | **76.8** | **85.4** | Niniejsza praca |

Alsentzer et al. (2019) nie raportują wyników SQuAD. W porównaniu z BioBERT
(na którym ClinicalBERT bazuje) widać -3.7pp F1, co potwierdza hipotezę
o catastrophic forgetting po dodatkowym pretreningu na wąskiej domenie klinicznej.

## Artefakty

- Model: `results/squad/clinicalbert/best_model/`
- Metryki: `results/squad/clinicalbert/results.json`
- Predykcje: `results/squad/clinicalbert/eval_predictions.json`
- TensorBoard: `results/squad/clinicalbert/runs/`
