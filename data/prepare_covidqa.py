"""
Przygotowanie datasetu COVID-QA (deepset/covid_qa_deepset).

Podział:
- 85% → 5-fold CV (train/val w każdym foldzie)
- 15% → held-out test (stały, seed=42)

Korekta: answer_start off-by-one (znany bug w anotacji deepset/covid_qa_deepset).
Wynik: data/processed/covidqa/ z plikami JSON w formacie SQuAD.
"""

import json
import statistics
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import KFold, train_test_split

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "processed" / "covidqa"
REPORT_PATH = Path(__file__).parent.parent / "internal_docs" / "raport_covidqa.md"
N_FOLDS = 5
TEST_SIZE = 0.15


# ---------------------------------------------------------------------------
# Korekta spanów
# ---------------------------------------------------------------------------
def fix_answer_spans(examples: list[dict]) -> dict:
    """Napraw answer_start off-by-one i zbierz statystyki korekty."""
    stats = {
        "total_spans": 0,
        "already_correct": 0,
        "fixed_off_by_one": 0,
        "fixed_search": 0,
        "unfixable": 0,
        "unfixable_details": [],
    }

    for ex in examples:
        new_starts = []
        new_texts = []
        for text, start in zip(ex["answers"]["text"], ex["answers"]["answer_start"]):
            stats["total_spans"] += 1
            context = ex["context"]
            extracted = context[start:start + len(text)]

            if extracted == text:
                stats["already_correct"] += 1
                new_starts.append(start)
                new_texts.append(text)
            elif context[start + 1:start + 1 + len(text)] == text:
                # Off-by-one (answer_start jest o 1 za mały)
                stats["fixed_off_by_one"] += 1
                new_starts.append(start + 1)
                new_texts.append(text)
            else:
                # Szukaj w kontekście
                actual_pos = context.find(text)
                if actual_pos >= 0:
                    stats["fixed_search"] += 1
                    new_starts.append(actual_pos)
                    new_texts.append(text)
                else:
                    stats["unfixable"] += 1
                    stats["unfixable_details"].append({
                        "id": ex["id"],
                        "answer": text[:100],
                        "given_start": start,
                        "extracted": extracted[:100],
                    })
                    # Zachowaj oryginał — HF Trainer i tak użyje tokenizacji
                    new_starts.append(start)
                    new_texts.append(text)

        ex["answers"]["answer_start"] = new_starts
        ex["answers"]["text"] = new_texts

    return stats


# ---------------------------------------------------------------------------
# Konwersja do SQuAD
# ---------------------------------------------------------------------------
def to_squad_format(examples: list[dict]) -> dict:
    """Konwertuje listę przykładów HF do formatu SQuAD."""
    data = []
    for ex in examples:
        article = {
            "title": ex.get("document_url", ""),
            "paragraphs": [
                {
                    "context": ex["context"],
                    "qas": [
                        {
                            "id": str(ex["id"]),
                            "question": ex["question"],
                            "answers": [
                                {
                                    "text": ex["answers"]["text"][i],
                                    "answer_start": ex["answers"]["answer_start"][i],
                                }
                                for i in range(len(ex["answers"]["text"]))
                            ],
                            "is_impossible": len(ex["answers"]["text"]) == 0,
                        }
                    ],
                }
            ],
        }
        data.append(article)
    return {"version": "v2.0", "data": data}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {path} ({len(data['data'])} examples)")


# ---------------------------------------------------------------------------
# Statystyki
# ---------------------------------------------------------------------------
def compute_stats(examples: list[dict]) -> dict:
    """Oblicz statystyki dla zbioru przykładów."""
    answer_lengths = []
    context_lengths = []
    answers_per_q = []
    questions_with_multiple_answers = 0

    for ex in examples:
        context_lengths.append(len(ex["context"]))
        n_answers = len(ex["answers"]["text"])
        answers_per_q.append(n_answers)
        if n_answers > 1:
            questions_with_multiple_answers += 1
        for ans_text in ex["answers"]["text"]:
            answer_lengths.append(len(ans_text))

    # Walidacja answer_start (po korekcie)
    valid_spans = 0
    invalid_spans = 0
    for ex in examples:
        for text, start in zip(ex["answers"]["text"], ex["answers"]["answer_start"]):
            if ex["context"][start:start + len(text)] == text:
                valid_spans += 1
            else:
                invalid_spans += 1

    return {
        "n_examples": len(examples),
        "answer_lengths": answer_lengths,
        "context_lengths": context_lengths,
        "answers_per_q": answers_per_q,
        "questions_with_multiple_answers": questions_with_multiple_answers,
        "valid_spans": valid_spans,
        "invalid_spans": invalid_spans,
    }


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------
def generate_report(
    raw_stats: dict,
    fix_stats: dict,
    all_examples: list[dict],
    cv_examples: list[dict],
    test_examples: list[dict],
    fold_sizes: list[tuple[int, int]],
) -> str:
    """Generuj szczegółowy raport markdown."""
    total_stats = compute_stats(all_examples)
    cv_stats = compute_stats(cv_examples)
    test_stats = compute_stats(test_examples)

    lines = []
    lines.append("# Raport: Przygotowanie datasetu COVID-QA\n")
    lines.append("Data generacji: automatyczna (skrypt `data/prepare_covidqa.py`)\n")

    # --- 1. Źródło danych ---
    lines.append("## 1. Źródło danych\n")
    lines.append("| Właściwość | Wartość |")
    lines.append("|------------|--------|")
    lines.append("| Dataset | `deepset/covid_qa_deepset` (HuggingFace Datasets) |")
    lines.append("| Licencja | Apache 2.0 |")
    lines.append("| Źródło artykułów | CORD-19 (COVID-19 Open Research Dataset) |")
    lines.append("| Anotatorzy | 15 ekspertów biomedycznych |")
    lines.append("| Format natywny | SQuAD-style (context + question + answers z answer_start) |")
    lines.append(f"| Łączna liczba przykładów | {raw_stats['n_examples']} |")
    lines.append("")

    # --- 2. Surowy dataset ---
    lines.append("## 2. Charakterystyka surowego datasetu\n")

    def stat_block(stats, name):
        block = []
        block.append(f"**{name}** ({stats['n_examples']} przykładów):\n")
        if stats["answer_lengths"]:
            block.append(f"- Długość odpowiedzi (znaki): min={min(stats['answer_lengths'])}, "
                        f"max={max(stats['answer_lengths'])}, "
                        f"średnia={statistics.mean(stats['answer_lengths']):.1f}, "
                        f"mediana={statistics.median(stats['answer_lengths']):.1f}")
        if stats["context_lengths"]:
            block.append(f"- Długość kontekstu (znaki): min={min(stats['context_lengths'])}, "
                        f"max={max(stats['context_lengths'])}, "
                        f"średnia={statistics.mean(stats['context_lengths']):.1f}, "
                        f"mediana={statistics.median(stats['context_lengths']):.1f}")
        if stats["answers_per_q"]:
            block.append(f"- Odpowiedzi na pytanie: min={min(stats['answers_per_q'])}, "
                        f"max={max(stats['answers_per_q'])}, "
                        f"średnia={statistics.mean(stats['answers_per_q']):.1f}")
            block.append(f"- Pytania z wieloma odpowiedziami: {stats['questions_with_multiple_answers']} "
                        f"({stats['questions_with_multiple_answers']/max(1,stats['n_examples'])*100:.1f}%)")
        block.append("")
        return block

    lines.extend(stat_block(raw_stats, "Cały dataset (przed korektą)"))

    # --- 3. Korekta spanów ---
    lines.append("## 3. Korekta `answer_start` (off-by-one bug)\n")
    lines.append("### 3.1. Problem\n")
    lines.append("Walidacja integralności surowego datasetu wykazała, że znaczna część pozycji")
    lines.append("`answer_start` jest przesunięta o 1 znak w lewo. Sprawdzenie:")
    lines.append("`context[answer_start : answer_start + len(answer_text)] != answer_text`")
    lines.append(f"wykazało **{fix_stats['fixed_off_by_one'] + fix_stats['fixed_search'] + fix_stats['unfixable']}** "
                f"niepoprawnych spanów z {fix_stats['total_spans']} łącznie "
                f"({(fix_stats['fixed_off_by_one'] + fix_stats['fixed_search'] + fix_stats['unfixable'])/max(1,fix_stats['total_spans'])*100:.1f}%).\n")
    lines.append("Jest to znany problem w anotacji datasetu `deepset/covid_qa_deepset`.\n")

    lines.append("### 3.2. Zastosowana korekta\n")
    lines.append("Kaskadowa strategia naprawy:\n")
    lines.append("1. **Off-by-one**: sprawdzenie `context[start+1 : start+1+len(text)] == text`")
    lines.append("2. **Wyszukiwanie**: `context.find(text)` — pierwsze wystąpienie w kontekście")
    lines.append("3. **Nienaprawialne**: zachowanie oryginału (tokenizator HF może sobie poradzić)\n")

    lines.append("### 3.3. Wyniki korekty\n")
    lines.append("| Kategoria | Liczba | % |")
    lines.append("|-----------|--------|---|")
    lines.append(f"| Poprawne od początku | {fix_stats['already_correct']} | {fix_stats['already_correct']/max(1,fix_stats['total_spans'])*100:.1f}% |")
    lines.append(f"| Naprawione (off-by-one) | {fix_stats['fixed_off_by_one']} | {fix_stats['fixed_off_by_one']/max(1,fix_stats['total_spans'])*100:.1f}% |")
    lines.append(f"| Naprawione (wyszukiwanie) | {fix_stats['fixed_search']} | {fix_stats['fixed_search']/max(1,fix_stats['total_spans'])*100:.1f}% |")
    lines.append(f"| Nienaprawialne | {fix_stats['unfixable']} | {fix_stats['unfixable']/max(1,fix_stats['total_spans'])*100:.1f}% |")
    lines.append(f"| **Łącznie** | **{fix_stats['total_spans']}** | **100%** |")
    lines.append("")

    lines.append("### 3.4. Walidacja po korekcie\n")
    lines.append(f"- Poprawne spany: **{total_stats['valid_spans']}** z {total_stats['valid_spans'] + total_stats['invalid_spans']}")
    lines.append(f"- Niepoprawne spany: **{total_stats['invalid_spans']}**")
    if total_stats['invalid_spans'] == 0:
        lines.append("\nWszystkie spany są poprawne po korekcie — integralność danych potwierdzona.")
    lines.append("")

    # --- 4. Podział ---
    lines.append("## 4. Podział danych\n")
    lines.append("### 4.1. Strategia podziału\n")
    lines.append(f"- **Held-out test**: {TEST_SIZE*100:.0f}% danych ({test_stats['n_examples']} przykładów), "
                f"losowanie z seed={SEED}")
    lines.append(f"- **Pula CV**: {(1-TEST_SIZE)*100:.0f}% danych ({cv_stats['n_examples']} przykładów)")
    lines.append(f"- **Cross-validation**: {N_FOLDS}-fold CV na puli CV (shuffle=True, seed={SEED})")
    lines.append("- Uzasadnienie: dataset jest zbyt mały na pojedynczy podział train/dev/test,")
    lines.append("  5-fold CV zapewnia stabilniejsze estymaty metryk\n")

    lines.append("### 4.2. Rozmiary foldów\n")
    lines.append("| Fold | Train | Val |")
    lines.append("|------|-------|-----|")
    for i, (train_size, val_size) in enumerate(fold_sizes):
        lines.append(f"| {i} | {train_size} | {val_size} |")
    lines.append(f"| held-out test | — | {test_stats['n_examples']} |")
    lines.append("")

    lines.append("### 4.3. Statystyki per split (po korekcie)\n")
    lines.extend(stat_block(total_stats, "Cały dataset"))
    lines.extend(stat_block(cv_stats, "Pula CV"))
    lines.extend(stat_block(test_stats, "Held-out test"))

    # --- 5. Modyfikacje ---
    lines.append("## 5. Podsumowanie modyfikacji danych\n")
    lines.append("| Modyfikacja | Opis |")
    lines.append("|-------------|------|")
    lines.append(f"| Korekta `answer_start` | Naprawiono {fix_stats['fixed_off_by_one'] + fix_stats['fixed_search']} spanów (off-by-one bug w anotacji) |")
    lines.append("| Re-strukturyzacja JSON | Konwersja z formatu HF Datasets do SQuAD v2.0 JSON |")
    lines.append("| Podział danych | Losowy: 85% CV (5-fold) + 15% held-out test, seed=42 |")
    lines.append("")
    lines.append("Nie zastosowano żadnych modyfikacji do treści pytań, kontekstów ani odpowiedzi.")
    lines.append("Pole `document_url` z oryginału zachowane jako `title` w formacie SQuAD.")
    lines.append("")

    # --- 6. Format ---
    lines.append("## 6. Format wyjściowy\n")
    lines.append("Pliki zapisane w formacie SQuAD v2.0 (JSON):\n")
    lines.append("```")
    lines.append("data/processed/covidqa/")
    lines.append("├── test.json                  (held-out test, 303 przykłady)")
    for i, (ts, vs) in enumerate(fold_sizes):
        lines.append(f"├── fold_{i}/train.json         ({ts} przykładów)")
        lines.append(f"├── fold_{i}/val.json           ({vs} przykładów)")
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading COVID-QA from HuggingFace...")
    dataset = load_dataset("deepset/covid_qa_deepset", split="train")
    all_examples = list(dataset)
    print(f"Total examples: {len(all_examples)}")

    # Statystyki surowego datasetu (przed korektą)
    raw_stats = compute_stats(all_examples)
    print(f"Raw: {raw_stats['valid_spans']} valid, {raw_stats['invalid_spans']} invalid spans")

    # Korekta answer_start
    print("\nFixing answer spans...")
    fix_stats = fix_answer_spans(all_examples)
    print(f"  Already correct: {fix_stats['already_correct']}")
    print(f"  Fixed (off-by-one): {fix_stats['fixed_off_by_one']}")
    print(f"  Fixed (search): {fix_stats['fixed_search']}")
    print(f"  Unfixable: {fix_stats['unfixable']}")

    # Podział: 85% CV, 15% held-out test
    cv_examples, test_examples = train_test_split(
        all_examples, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"\nCV pool: {len(cv_examples)}, Held-out test: {len(test_examples)}")

    # Zapis held-out test
    save_json(to_squad_format(test_examples), OUTPUT_DIR / "test.json")

    # 5-fold CV
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_sizes = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cv_examples)):
        fold_train = [cv_examples[i] for i in train_idx]
        fold_val = [cv_examples[i] for i in val_idx]

        fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
        save_json(to_squad_format(fold_train), fold_dir / "train.json")
        save_json(to_squad_format(fold_val), fold_dir / "val.json")
        fold_sizes.append((len(fold_train), len(fold_val)))

    # Raport
    report = generate_report(raw_stats, fix_stats, all_examples, cv_examples, test_examples, fold_sizes)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")

    # Podsumowanie
    print(f"\nDataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
