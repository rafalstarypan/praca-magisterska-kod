# Wynik treningu: PubMedBERT — BioASQ LoRA r=16 (ablacja ranku)

Data: 2026-04-14
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 — ablacja ranku LoRA (r=16)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 109,484,548 |
| Parametry (trainable) | 591,362 (0.54% — LoRA) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | **16** (ablacja — domyślny r=8) |
| Alpha (α) | 16 (α/r = 1.0) |
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
| train_runtime | 51 s |
| train_loss (średni) | 1.068 |

## Ablacja ranku — kompletne porównanie

| Strategia | Trainable | % total | Test EM | Test F1 | VRAM | α/r |
|-----------|-----------|---------|---------|---------|------|-----|
| Full FT | 108,893,186 | 100% | **59.76** | 73.52 | ~3900 MB | — |
| LoRA r=4 | 148,994 | 0.14% | 53.66 | 73.77 | 2518 MB | 4.0 |
| LoRA r=8 | 296,450 | 0.27% | 53.66 | 73.38 | 2522 MB | 2.0 |
| LoRA r=16 | 591,362 | 0.54% | 53.66 | 73.77 | 2528 MB | 1.0 |

### Komentarz — wnioski z ablacji

**Kluczowa obserwacja**: Wszystkie warianty LoRA (r=4, r=8, r=16) dają
praktycznie identyczny wynik na BioASQ (F1 ≈ 73.4–73.8, EM = 53.66),
mimo 4-krotnej różnicy w liczbie parametrów (149K vs 591K).

**Interpretacja**:
1. Adaptacja PubMedBERT do BioASQ factoid QA wymaga **bardzo niskiej
   wymiarowości** — nawet r=4 (0.14% parametrów) jest wystarczające.
   Potwierdza to hipotezę niskorangowej adaptacji (Aghajanyan et al. 2021).
2. Brak poprawy przy r=16 vs r=4 oznacza, że **dodatkowa pojemność nie
   pomaga** — model już "nauczył się" adaptacji w 4-wymiarowej podprzestrzeni.
3. Różnica α/r (1.0 vs 4.0) nie wpływa istotnie na wynik, co sugeruje
   robustność LoRA na hiperparametr skalowania dla tego zadania.
4. Full FT wygrywa jedynie w EM (+6.1pp), co może wynikać z większej
   swobody w dokładnym pozycjonowaniu granic spanów. F1 (mierzący overlap
   tokenów) jest identyczny lub lepszy dla LoRA.

**Wniosek dla pracy**: Ranga r nie jest krytycznym hiperparametrem dla
biomedycznego extractive QA z pretrenowanym modelem domenowym. Rekomendacja:
r=8 jako bezpieczny default, ale r=4 jest wystarczający i bardziej efektywny.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 2528 MB |
| Czas treningu | 51 s |

## Artefakty

- Wyniki: `results/bioasq/pubmedbert_lora_r16/results.json`
- Predykcje: `results/bioasq/pubmedbert_lora_r16/{eval,test}_predictions.json`
- Adapter LoRA: `results/bioasq/pubmedbert_lora_r16/best_model/` (~2.4 MB)
