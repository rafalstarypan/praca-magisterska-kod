# Aplikacja demonstracyjna — ekstrakcyjne QA

Aplikacja Gradio umożliwiająca interaktywne testowanie sześciu modeli BERT
w zadaniu ekstrakcyjnego pytanie–odpowiedź na tekstach biomedycznych.

## Wymagania

- Python 3.12
- Wytrenowane modele w katalogu `results/bioasq/` (generowane przez eksperymenty E1)
- Zależności projektu zainstalowane w środowisku wirtualnym

## Uruchomienie

### 1. Aktywacja środowiska wirtualnego

```powershell
# Z katalogu głównego projektu (praca-magisterska-kod)
.\venv\Scripts\Activate.ps1
```

### 2. Uruchomienie aplikacji

```powershell
python app/demo.py
```

Po uruchomieniu aplikacja jest dostępna pod adresem: `http://localhost:7860`

## Użytkowanie

1. **Wybierz model** z listy rozwijanej (domyślnie: PubMedBERT).
2. **Wklej tekst biomedyczny** w pole „Kontekst".
3. **Wpisz pytanie** dotyczące wklejonego tekstu.
4. Kliknij przycisk **„Odpowiedz"**.

Aplikacja wyświetla:
- przewidzianą odpowiedź (fragment kontekstu),
- kontekst z zaznaczonym fragmentem odpowiedzi,
- wartość pewności modelu (iloczyn prawdopodobieństw tokenu startowego i końcowego).

Gotowe przykłady z zestawów BioASQ i COVID-QA dostępne są w sekcji
„Przykładowe pytania" na dole strony.

## Dostępne modele

| Nazwa w interfejsie | Model bazowy |
|---|---|
| PubMedBERT (najlepszy) | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext |
| RoBERTa | FacebookAI/roberta-base |
| BioBERT | dmis-lab/biobert-v1.1 |
| BERT (bazowy) | google-bert/bert-base-uncased |
| DistilBERT (szybki) | distilbert/distilbert-base-uncased |
| ClinicalBERT | emilyalsentzer/Bio_ClinicalBERT |

Każdy model jest ładowany z katalogu `results/bioasq/<nazwa>/best_model/`
i buforowany w pamięci po pierwszym użyciu.
