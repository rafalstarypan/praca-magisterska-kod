# Wynik treningu: DistilBERT — BioASQ (BEZ pretreningu SQuAD)

Data: 2026-03-31
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 38 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | DistilBERT (distilbert/distilbert-base-uncased) |
| Parametry (total) | 66,364,418 |
| Parametry (trainable) | 66,364,418 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM, **bez SQuAD**) |

## Dataset

| Parametr | Wartość |
|----------|---------|
| Dataset | BioASQ Task B factoid |
| Train | 1,217 przykładów |
| Dev | 73 przykłady (BioASQ 12b golden enriched) |
| Test | 82 przykłady (BioASQ 13b golden enriched) |

## Hiperparametry

Identyczne jak w pretreningu SQuAD, z wyjątkiem epochs=3 (zamiast 2).

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 32.88% | 50.02% |
| **Test** | **24.39%** | **38.91%** |

### Metryki treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (średni) | 2.46 |
| Throughput | 95.0 samples/s |
| Czas | 38 s |

## Znaczenie tego runu

Ten run stanowi **baseline bez pretreningu na SQuAD**. Po uruchomieniu wariantu
z pretreningiem (SQuAD → BioASQ), porównanie pokaże wpływ transferu wiedzy
z ogólnego QA na domenę biomedyczną.

## Artefakty

- Model: `results/bioasq/distilbert/best_model/`
- Metryki: `results/bioasq/distilbert/results.json`
- Predykcje: `results/bioasq/distilbert/{eval,test}_predictions.json`
