# Wynik treningu: PubMedBERT — BioASQ LoRA r=8 (Z pretreningiem SQuAD)

Data: 2026-04-13
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 (Full FT vs LoRA)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 109,189,636 |
| Parametry (trainable) | 296,450 (0.27% — LoRA) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.1 |
| Target modules | query, value |
| Learning rate | 1e-4 (override vs 3e-5 dla full FT) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 46.58% | 63.25% |
| **Test** | **53.66%** | **73.38%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 52.5 s |
| train_loss (średni) | 1.068 |
| train_samples_per_second | 69.52 |
| epoch | 3.0 |

## Porównanie z Full Fine-Tuning (E1)

| Metryka | Full FT | LoRA r=8 | Δ |
|---------|---------|----------|---|
| Test EM | 59.76% | 53.66% | -6.10pp |
| Test F1 | 73.52% | 73.38% | -0.14pp |
| Trainable params | 108,893,186 | 296,450 | -99.73% |
| Peak VRAM | ~3900 MB | 2522 MB | -35% |
| Czas treningu | 68.5 s | 52.5 s | -23% |

### Komentarz

LoRA r=8 osiąga praktycznie identyczny Test F1 (73.38 vs 73.52, Δ=-0.14pp)
przy 370× mniejszej liczbie trenowalnych parametrów. Spadek EM o 6.10pp
sugeruje, że LoRA daje nieco mniej precyzyjne granice spanów, ale jakość
semantyczna odpowiedzi (mierzona F1) jest zachowana.

Wynik jest zgodny z hipotezą H1 (LoRA ≥95% F1 full FT) i literaturą
(Hu et al. 2022, Table 6). Oszczędność VRAM (-35%) i czasu (-23%) to
dodatkowa korzyść istotna w kontekście pytania badawczego PB4.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 2522 MB |
| Peak VRAM (reserved) | 2772 MB |
| Czas treningu | 52.9 s |
| Platform | Windows 11 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |

## Artefakty

- Wyniki: `results/bioasq/pubmedbert_lora_r8/results.json`
- Predykcje: `results/bioasq/pubmedbert_lora_r8/{eval,test}_predictions.json`
- Adapter LoRA: `results/bioasq/pubmedbert_lora_r8/best_model/` (~1.2 MB)
