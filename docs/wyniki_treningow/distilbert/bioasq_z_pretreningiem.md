# Wynik treningu: DistilBERT — BioASQ (Z pretreningiem SQuAD)

Data: 2026-04-05
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 44 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | DistilBERT (distilbert/distilbert-base-uncased) |
| Parametry (total) | 66,364,418 |
| Parametry (trainable) | 66,364,418 (100% — full fine-tuning) |
| Źródło wag | `results/squad/distilbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 45.21% | 66.01% |
| **Test** | **42.68%** | **60.83%** |

### Przebieg treningu (eval po każdej epoce)

| Epoka | Train Loss | Eval Loss | Eval EM | Eval F1 | Uwagi |
|-------|-----------|-----------|---------|---------|-------|
| 1 | 1.466 | 1.351 | 39.73% | 62.18% | |
| 2 | 0.843 | 1.380 | 45.21% | 66.01% | **Najlepszy model (wg F1)** |
| 3 | 0.602 | 1.468 | 43.84% | 64.87% | Overfitting: train loss spada, eval F1 spada |

Model z epoki 2 został wybrany jako najlepszy (`load_best_model_at_end=True`,
`metric_for_best_model="f1"`).

### Dane z TensorBoard (pełna rozdzielczość)

**Train loss** (logowany co 50 kroków):

| Step | Epoch | Loss | Grad Norm | Learning Rate |
|------|-------|------|-----------|---------------|
| 50 | 0.65 | 1.466 | 13.29 | 2.667e-05 |
| 100 | 1.30 | 1.124 | 13.19 | 1.942e-05 |
| 150 | 1.95 | 0.844 | 12.52 | 1.217e-05 |
| 200 | 2.60 | 0.602 | 16.69 | 4.928e-06 |

**Eval metrics** (logowane co epokę):

| Step | Epoch | Eval Loss | Eval EM | Eval F1 | Samples/s |
|------|-------|-----------|---------|---------|-----------|
| 77 | 1.0 | 1.351 | 39.73% | 62.18% | 200.3 |
| 154 | 2.0 | 1.380 | 45.21% | 66.01% | 167.1 |
| 231 | 3.0 | 1.468 | 43.84% | 64.87% | 167.5 |

**Metryki końcowe treningu**:

| Metryka | Wartość |
|---------|---------|
| train_runtime | 44.1 s |
| train_samples_per_second | 82.85 |
| train_steps_per_second | 5.24 |
| total_flos | 3.58e+14 |
| train_loss (średni) | 0.947 |

Źródło: `results/bioasq/distilbert/runs/Apr05_22-36-19_rafal-alienware/events.out.tfevents.*`

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Dev EM | 32.88% | 45.21% | +12.33pp |
| Dev F1 | 50.02% | 66.01% | +15.99pp |
| Test EM | 24.39% | 42.68% | +18.29pp |
| Test F1 | 38.91% | 60.83% | +21.92pp |

### Wnioski

1. **Pretrening na SQuAD daje +22pp F1 na test set** — największa pojedyncza
   poprawa w pipeline. Potwierdza standardową praktykę z literatury (Yoon et al., 2020).

2. **Overfitting po epoce 2** — train loss spada (0.84→0.60), ale eval F1 spada
   (66.0→64.9). Mechanizm `load_best_model_at_end` poprawnie wybiera epokę 2.

3. **Test F1=60.8% dla DistilBERT** — realistyczny wynik dla najsłabszego modelu.
   Modele domenowe (BioBERT, PubMedBERT) powinny uzyskać F1 65-75%.

## Artefakty

- Model: `results/bioasq/distilbert/best_model/`
- Metryki: `results/bioasq/distilbert/results.json`
- Predykcje: `results/bioasq/distilbert/{eval,test}_predictions.json`
- TensorBoard: `results/bioasq/distilbert/tensorboard/`

**Uwaga**: Ten run nadpisał poprzednie wyniki (bez SQuAD pretreningu) w tym samym
katalogu. Wyniki bez pretreningu zachowane w pliku
`distilbert_bioasq_bez_pretreningu.md`.
