# Kod źródłowy pracy magisterskiej

## Tytuł pracy

**Wykorzystanie modelu BERT w budowie ekstrakcyjnego systemu pytanie–odpowiedź dla danych z domeny biomedycznej**

*Using the BERT model to build an extractive question answering system for biomedical domain data*

**Autor:** Rafał Starypan (288753)  
**Promotor:** dr inż. Natalia Piórkowska  
**Uczelnia:** Politechnika Wrocławska, Wydział Informatyki i Telekomunikacji  
**Rok:** 2026

---

## Cel repozytorium

Repozytorium zawiera kod źródłowy narzędzi opracowanych na potrzeby empirycznej części pracy magisterskiej. Celem pracy jest wieloaspektowa ocena wariantów modeli transformatorowych typu BERT w zadaniu ekstrakcyjnego pytanie–odpowiedź (QA) dla danych biomedycznych — obejmująca trafność odpowiedzi, efektywność obliczeniową oraz interpretowalność.

---

## Eksperymenty

| Symbol | Opis |
|--------|------|
| **E1** | Macierz porównawcza 6 modeli × 2 zbiory danych (BioASQ, COVID-QA) |
| **E2** | Pełne dostrajanie vs metoda LoRA (niskorangowa adaptacja parametryczna) |
| **E3** | Profilowanie wydajności wnioskowania — latencja, przepustowość, zużycie VRAM |
| **E4** | Analiza interpretowalności — mechanizm uwagi, LIME, metryki wierności wyjaśnień |

### Badane modele

| Identyfikator HuggingFace | Nazwa |
|---------------------------|-------|
| `google-bert/bert-base-uncased` | BERT |
| `dmis-lab/biobert-v1.1` | BioBERT |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMedBERT |
| `FacebookAI/roberta-base` | RoBERTa |
| `distilbert/distilbert-base-uncased` | DistilBERT |
| `emilyalsentzer/Bio_ClinicalBERT` | ClinicalBERT |

---

## Struktura repozytorium

```
.
├── app/                    # Aplikacja demonstracyjna (Gradio)
│   ├── demo.py
│   └── README.md           # Instrukcje uruchomienia aplikacji
├── configs/                # Konfiguracje hiperparametrów (YAML)
│   ├── base.yaml           # Pełne dostrajanie
│   └── lora.yaml           # Metoda LoRA
├── data/                   # Zbiory danych (BioASQ, COVID-QA, SQuAD)
├── docs/                   # Dokumentacja pomocnicza
├── notebooks/              # Notebooki Jupyter (eksploracja, wizualizacje)
├── results/                # Wyniki eksperymentów
├── scripts/                # Skrypty uruchomieniowe
│   ├── run_covidqa_folds.bat   # E1 – COVID-QA, walidacja krzyżowa
│   ├── run_e2_lora.bat         # E2 – LoRA
│   ├── run_e3.bat              # E3 – profilowanie
│   ├── aggregate_e1.py         # Agregacja wyników E1
│   └── aggregate_e3.py         # Agregacja wyników E3
├── src/                    # Główny kod źródłowy
│   ├── train.py            # Trening modeli (E1, E2)
│   ├── evaluate.py         # Ewaluacja (Exact Match, F1)
│   ├── benchmark.py        # Profilowanie wydajności (E3)
│   ├── interpret.py        # Analiza interpretowalności (E4)
│   ├── run_e4.py           # Punkt wejścia eksperymentu E4
│   └── data.py             # Wczytywanie i przetwarzanie danych
└── requirements.txt        # Zależności Python
```

---

## Wymagania

- Python 3.12
- CUDA 12.4 (do eksperymentów na GPU)
- GPU z min. 8 GB VRAM (testowano na NVIDIA GeForce RTX 4070 Laptop)

### Instalacja zależności

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Uruchomienie eksperymentów

### E1 — Macierz porównawcza modeli

```powershell
# Wstępne dostrajanie na SQuAD (dla każdego modelu)
python src/train.py --config configs/base.yaml --dataset squad --model bert-base-uncased

# Dostrajanie na BioASQ
python src/train.py --config configs/base.yaml --dataset bioasq --model bert-base-uncased

# Dostrajanie na COVID-QA (5-krotna walidacja krzyżowa)
.\scripts\run_covidqa_folds.bat
```

### E2 — LoRA vs pełne dostrajanie

```powershell
.\scripts\run_e2_lora.bat
```

### E3 — Profilowanie wydajności

```powershell
.\scripts\run_e3.bat
```

### E4 — Analiza interpretowalności

```powershell
python src/run_e4.py
```

---

## Aplikacja demonstracyjna

Interaktywna aplikacja Gradio umożliwia porównanie działania wytrenowanych modeli w czasie rzeczywistym. Do uruchomienia potrzebne są wyeksportowane pliki z wytrenowanymi modelami. Szczegółowe instrukcje uruchomienia znajdują się w [app/README.md](app/README.md).

---

## Stos technologiczny

| Biblioteka | Wersja |
|------------|--------|
| PyTorch | 2.6.0+cu124 |
| Transformers (HuggingFace) | 5.3.0 |
| PEFT | 0.18.1 |
| Datasets | 4.7.0 |
| LIME | 0.2.0 |
| Gradio | 6.9.0 |
| wandb | — |

---
