# Wynik treningu: PubMedBERT — BioASQ LoRA r=4 (ablacja ranku)

Data: 2026-04-14
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 — ablacja ranku LoRA (r=4)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 109,042,180 |
| Parametry (trainable) | 148,994 (0.14% — LoRA) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | **4** (ablacja — domyślny r=8) |
| Alpha (α) | 16 (α/r = 4.0) |
| Dropout | 0.1 |
| Target modules | query, value |
| Learning rate | 1e-4 |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| **Test** | **53.66%** | **73.77%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 54 s |
| train_loss (średni) | 1.069 |

## Ablacja ranku — porównanie

| Strategia | Trainable | % total | Test EM | Test F1 | VRAM | α/r |
|-----------|-----------|---------|---------|---------|------|-----|
| Full FT | 108,893,186 | 100% | 59.76 | 73.52 | ~3900 MB | — |
| LoRA r=4 | 148,994 | 0.14% | 53.66 | **73.77** | 2518 MB | 4.0 |
| LoRA r=8 | 296,450 | 0.27% | 53.66 | 73.38 | 2522 MB | 2.0 |
| LoRA r=16 | ~590K | ~0.54% | *oczekiwany* | *oczekiwany* | — | 1.0 |

### Komentarz

LoRA r=4 osiąga **najwyższy F1** (73.77) spośród wszystkich wariantów,
w tym Full FT (73.52) i LoRA r=8 (73.38). Przy zaledwie 149K trenowalnych
parametrów (0.14% modelu) wynik jest lepszy niż Full FT ze 109M parametrów.

Potwierdza to hipotezę niskorangowej adaptacji (Aghajanyan et al. 2021):
na małym zbiorze jak BioASQ (~1.2k treningowych) adaptacja wymaga bardzo
niskiej wymiarowości. Wyższy mnożnik α/r=4.0 (vs 2.0 dla r=8) kompensuje
mniejszą pojemność silniejszym skalowaniem aktualizacji.

EM jest identyczny dla r=4 i r=8 (53.66), co sugeruje, że precyzja granic
spanów jest ograniczona przez inne czynniki niż ranga LoRA.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 2518 MB |
| Czas treningu | 54 s |

## Artefakty

- Wyniki: `results/bioasq/pubmedbert_lora_r4/results.json`
- Predykcje: `results/bioasq/pubmedbert_lora_r4/{eval,test}_predictions.json`
- Adapter LoRA: `results/bioasq/pubmedbert_lora_r4/best_model/` (~0.6 MB)
