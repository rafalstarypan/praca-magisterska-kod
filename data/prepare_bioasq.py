"""
Konwersja BioASQ Task B do formatu SQuAD.

Pliki źródłowe (data/raw/bioasq/):
- BioASQ-training14b/BioASQ-training14b/training14b.json  (kumulatywne, edycje 1-14b)
- Task12BGoldenEnriched/Task12BGoldenEnriched/12B*_golden.json  (test BioASQ12)
- Task13BGoldenEnriched/Task13BGoldenEnriched/13B*_golden.json  (test BioASQ13)

Podział chronologiczny:
- train: Training 14b minus pytania z 12b i 13b golden (edycje 1-11b)
- dev:   12b golden enriched
- test:  13b golden enriched

Filtracja: tylko pytania typu "factoid" z odpowiedziami w snippetach.
Strategie dopasowania (kaskadowe):
  1. exact: case-insensitive substring match
  2. normalized: po usunięciu interpunkcji i normalizacji whitespace
  3. abbreviation: dopasowanie skrótu z nawiasu lub rozwiniętej formy
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw" / "bioasq"
OUTPUT_DIR = Path(__file__).parent / "processed" / "bioasq"
REPORT_PATH = Path(__file__).parent.parent / "internal_docs" / "raport_bioasq.md"


# ---------------------------------------------------------------------------
# Statystyki
# ---------------------------------------------------------------------------
@dataclass
class ConversionStats:
    split: str
    total_questions: int = 0
    by_type: dict = field(default_factory=lambda: Counter())
    factoid_total: int = 0
    factoid_no_snippet: int = 0
    factoid_no_answer: int = 0
    matched_exact: int = 0
    matched_normalized: int = 0
    matched_abbreviation: int = 0
    not_matched: int = 0
    not_matched_details: list = field(default_factory=list)
    answer_lengths: list = field(default_factory=list)
    context_lengths: list = field(default_factory=list)
    snippets_per_question: list = field(default_factory=list)
    answers_per_question: list = field(default_factory=list)

    @property
    def total_matched(self) -> int:
        return self.matched_exact + self.matched_normalized + self.matched_abbreviation

    def summary_dict(self) -> dict:
        return {
            "split": self.split,
            "total_questions": self.total_questions,
            "question_types": dict(self.by_type),
            "factoid_total": self.factoid_total,
            "factoid_no_snippet": self.factoid_no_snippet,
            "factoid_no_answer": self.factoid_no_answer,
            "factoid_with_data": self.factoid_total - self.factoid_no_snippet - self.factoid_no_answer,
            "matched_exact": self.matched_exact,
            "matched_normalized": self.matched_normalized,
            "matched_abbreviation": self.matched_abbreviation,
            "total_matched": self.total_matched,
            "not_matched": self.not_matched,
            "match_rate": f"{self.total_matched / max(1, self.factoid_total - self.factoid_no_snippet - self.factoid_no_answer) * 100:.1f}%",
        }


# ---------------------------------------------------------------------------
# Strategie dopasowania
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Usuń interpunkcję, normalizuj whitespace, lowercase."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_abbreviation(answer: str) -> list[str]:
    """Wyodrębnij skrót z nawiasu i/lub pełną formę.

    Np. 'Reverse transcription - polymerase chain reaction (RT-PCR)'
    → ['RT-PCR', 'Reverse transcription - polymerase chain reaction']
    """
    candidates = []
    # Skrót w nawiasie
    match = re.search(r'\(([A-Z][A-Za-z0-9\-/]{1,15})\)', answer)
    if match:
        abbr = match.group(1)
        candidates.append(abbr)
        # Forma bez nawiasu
        full_form = answer[:match.start()].strip().rstrip('-').strip()
        if len(full_form) > 3:
            candidates.append(full_form)
    return candidates


def find_span_exact(snippet: str, answer: str) -> tuple[int, str] | None:
    """Strategia 1: Case-insensitive exact substring match."""
    idx = snippet.lower().find(answer.lower())
    if idx >= 0:
        return idx, snippet[idx:idx + len(answer)]
    return None


def find_span_normalized(snippet: str, answer: str) -> tuple[int, str] | None:
    """Strategia 2: Match po normalizacji (usunięcie interpunkcji).

    Po znalezieniu dopasowania w znormalizowanym tekście, mapujemy pozycję
    z powrotem na oryginalny tekst.
    """
    norm_snippet = normalize_text(snippet)
    norm_answer = normalize_text(answer)

    if len(norm_answer) < 2:
        return None

    idx = norm_snippet.find(norm_answer)
    if idx < 0:
        return None

    # Mapowanie pozycji: budujemy tablicę pozycji oryginalnych
    temp_norm = []
    for i, ch in enumerate(snippet):
        normalized_ch = re.sub(r'[^\w\s]', ' ', ch.lower())
        for nc in normalized_ch:
            temp_norm.append((nc, i))

    # Rekonstrukcja z normalizacją whitespace
    clean_positions = []
    prev_space = True
    for nc, orig_i in temp_norm:
        if nc == ' ':
            if not prev_space:
                clean_positions.append((' ', orig_i))
                prev_space = True
        else:
            clean_positions.append((nc, orig_i))
            prev_space = False

    # Strip leading/trailing
    while clean_positions and clean_positions[0][0] == ' ':
        clean_positions.pop(0)
    while clean_positions and clean_positions[-1][0] == ' ':
        clean_positions.pop()

    if idx + len(norm_answer) > len(clean_positions):
        return None

    orig_start = clean_positions[idx][1]
    orig_end = clean_positions[idx + len(norm_answer) - 1][1] + 1
    matched_text = snippet[orig_start:orig_end]

    return orig_start, matched_text


def find_span_abbreviation(snippet: str, answer: str) -> tuple[int, str] | None:
    """Strategia 3: Dopasuj skrót lub formę rozwiniętą."""
    candidates = extract_abbreviation(answer)
    for candidate in candidates:
        result = find_span_exact(snippet, candidate)
        if result:
            return result
        result = find_span_normalized(snippet, candidate)
        if result:
            return result
    return None


def find_best_span(snippet: str, answer: str) -> tuple[int, str, str] | None:
    """Próbuj kolejnych strategii, zwróć (start, text, strategy) lub None."""
    result = find_span_exact(snippet, answer)
    if result:
        return result[0], result[1], "exact"

    result = find_span_normalized(snippet, answer)
    if result:
        return result[0], result[1], "normalized"

    result = find_span_abbreviation(snippet, answer)
    if result:
        return result[0], result[1], "abbreviation"

    return None


# ---------------------------------------------------------------------------
# Konwersja
# ---------------------------------------------------------------------------
def convert_bioasq_to_squad(questions: list[dict], stats: ConversionStats) -> dict:
    """Konwertuje pytania BioASQ factoid do formatu SQuAD."""
    data = []

    for q in questions:
        stats.total_questions += 1
        q_type = q.get("type", "unknown")
        stats.by_type[q_type] += 1

        if q_type != "factoid":
            continue

        stats.factoid_total += 1

        snippets = q.get("snippets", [])
        stats.snippets_per_question.append(len(snippets))
        if not snippets:
            stats.factoid_no_snippet += 1
            continue

        exact_answers = q.get("exact_answer", [])
        flat_answers = []
        for ans in exact_answers:
            if isinstance(ans, list):
                flat_answers.extend(ans)
            else:
                flat_answers.append(ans)

        stats.answers_per_question.append(len(flat_answers))
        if not flat_answers:
            stats.factoid_no_answer += 1
            continue

        # Próbuj dopasować odpowiedź w snippetach
        best_match = None
        for snippet_obj in snippets:
            snippet_text = snippet_obj.get("text", "")
            if not snippet_text:
                continue
            for answer_text in flat_answers:
                result = find_best_span(snippet_text, answer_text)
                if result:
                    best_match = (snippet_text, result[0], result[1], result[2])
                    break
            if best_match:
                break

        if best_match:
            snippet_text, answer_start, matched_text, strategy = best_match

            if strategy == "exact":
                stats.matched_exact += 1
            elif strategy == "normalized":
                stats.matched_normalized += 1
            elif strategy == "abbreviation":
                stats.matched_abbreviation += 1

            stats.answer_lengths.append(len(matched_text))
            stats.context_lengths.append(len(snippet_text))

            article = {
                "title": q.get("body", "")[:100],
                "paragraphs": [
                    {
                        "context": snippet_text,
                        "qas": [
                            {
                                "id": q["id"],
                                "question": q["body"],
                                "answers": [
                                    {
                                        "text": matched_text,
                                        "answer_start": answer_start,
                                    }
                                ],
                                "is_impossible": False,
                            }
                        ],
                    }
                ],
            }
            data.append(article)
        else:
            stats.not_matched += 1
            stats.not_matched_details.append({
                "id": q["id"],
                "question": q["body"][:120],
                "answers": flat_answers[:3],
                "snippet_preview": snippets[0]["text"][:200] if snippets else "",
                "n_snippets": len(snippets),
            })

    return {"version": "v2.0", "data": data}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_golden_enriched(task_dir: Path) -> list[dict]:
    """Załaduj pytania z golden enriched (wszystkie batche)."""
    questions = []
    for json_file in sorted(task_dir.rglob("*_golden.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        qs = data.get("questions", data) if isinstance(data, dict) else data
        questions.extend(qs)
        print(f"  Loaded {json_file.name}: {len(qs)} questions")
    return questions


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {path} ({len(data['data'])} examples)")


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------
def generate_report(all_stats: list[ConversionStats], golden_exclusion_info: dict) -> str:
    """Generuj szczegółowy raport markdown."""
    lines = []
    lines.append("# Raport: Przygotowanie datasetu BioASQ Task B\n")
    lines.append(f"Data generacji: automatyczna (skrypt `data/prepare_bioasq.py`)\n")

    lines.append("## 1. Źródła danych\n")
    lines.append("| Plik | Rola | Pytania |")
    lines.append("|------|------|---------|")
    lines.append(f"| `training14b.json` | Pula treningowa (kumulatywna, edycje 1-14b) | {golden_exclusion_info['total_training']} |")
    lines.append(f"| `Task12BGoldenEnriched` (4 batche) | Dev set | {golden_exclusion_info['golden_12b']} |")
    lines.append(f"| `Task13BGoldenEnriched` (4 batche) | Test set | {golden_exclusion_info['golden_13b']} |")
    lines.append("")

    lines.append("## 2. Podział danych (chronologiczny)\n")
    lines.append("Podział oparty na chronologii edycji BioASQ challenge:\n")
    lines.append(f"- **Training 14b** zawiera pytania z edycji 1b-14b ({golden_exclusion_info['total_training']} pytań)")
    lines.append(f"- Z puli treningowej **usunięto** pytania występujące w golden enriched 12b i 13b")
    lines.append(f"  - Usunięto {golden_exclusion_info['excluded']} pytań (po ID)")
    lines.append(f"  - Pozostało {golden_exclusion_info['after_exclusion']} pytań jako pula treningowa (efektywnie edycje 1-11b + pytania z 14b)")
    lines.append(f"- **Dev**: 12b golden enriched ({golden_exclusion_info['golden_12b']} pytań) — test set BioASQ12 (2024)")
    lines.append(f"- **Test**: 13b golden enriched ({golden_exclusion_info['golden_13b']} pytań) — test set BioASQ13 (2025)")
    lines.append("")

    lines.append("## 3. Filtracja i konwersja\n")
    lines.append("### 3.1. Filtracja typów pytań\n")
    lines.append("BioASQ Task B zawiera 4 typy pytań. W tej pracy wykorzystujemy **wyłącznie factoid**,")
    lines.append("ponieważ są to pytania z krótką, ekstrakcyjną odpowiedzią (kompatybilne z formatem SQuAD).\n")

    for s in all_stats:
        lines.append(f"**{s.split.upper()}** — rozkład typów pytań:\n")
        lines.append("| Typ | Liczba | % |")
        lines.append("|-----|--------|---|")
        for qtype, count in s.by_type.most_common():
            pct = count / max(1, s.total_questions) * 100
            lines.append(f"| {qtype} | {count} | {pct:.1f}% |")
        lines.append("")

    lines.append("### 3.2. Strategie dopasowania odpowiedzi w snippetach\n")
    lines.append("Pytania factoid w BioASQ mają pole `exact_answer` (gold answer) oraz `snippets`")
    lines.append("(fragmenty artykułów PubMed). Aby skonwertować do formatu SQuAD, musimy znaleźć")
    lines.append("`exact_answer` jako span w jednym ze snippetów.\n")
    lines.append("Zastosowano **kaskadowe strategie dopasowania** (od najściślejszej do najluźniejszej):\n")
    lines.append("1. **exact** — case-insensitive substring match (`answer.lower() in snippet.lower()`)")
    lines.append("2. **normalized** — po usunięciu interpunkcji i normalizacji whitespace z obu stron,")
    lines.append("   z mapowaniem pozycji z powrotem na oryginalny tekst")
    lines.append("3. **abbreviation** — wyodrębnienie skrótu z nawiasu (np. `(RT-PCR)`) lub formy")
    lines.append("   rozwiniętej i dopasowanie ich osobno strategiami 1-2\n")

    lines.append("### 3.3. Wyniki dopasowania per split\n")
    lines.append("| Split | Factoid | Bez snippetów | Bez odpowiedzi | exact | normalized | abbreviation | **Razem dopasowane** | Odrzucone | Match rate |")
    lines.append("|-------|---------|---------------|----------------|-------|------------|-------------|---------------------|-----------|------------|")
    for s in all_stats:
        factorable = s.factoid_total - s.factoid_no_snippet - s.factoid_no_answer
        rate = s.total_matched / max(1, factorable) * 100
        lines.append(
            f"| {s.split} | {s.factoid_total} | {s.factoid_no_snippet} | {s.factoid_no_answer} "
            f"| {s.matched_exact} | {s.matched_normalized} | {s.matched_abbreviation} "
            f"| **{s.total_matched}** | {s.not_matched} | {rate:.1f}% |"
        )
    lines.append("")

    # Porównanie ze starą metodą (tylko exact)
    lines.append("### 3.4. Poprawa dzięki rozszerzonym strategiom\n")
    lines.append("Porównanie z podejściem bazowym (tylko exact match):\n")
    lines.append("| Split | Tylko exact | + normalized | + abbreviation | Poprawa |")
    lines.append("|-------|------------|-------------|---------------|---------|")
    for s in all_stats:
        improvement = s.matched_normalized + s.matched_abbreviation
        pct_improve = improvement / max(1, s.matched_exact) * 100
        lines.append(
            f"| {s.split} | {s.matched_exact} | +{s.matched_normalized} | +{s.matched_abbreviation} "
            f"| +{improvement} (+{pct_improve:.1f}%) |"
        )
    lines.append("")

    lines.append("### 3.5. Statystyki końcowego datasetu\n")
    for s in all_stats:
        import statistics
        lines.append(f"**{s.split.upper()}** ({s.total_matched} przykładów):\n")
        if s.answer_lengths:
            lines.append(f"- Długość odpowiedzi (znaki): min={min(s.answer_lengths)}, "
                        f"max={max(s.answer_lengths)}, "
                        f"średnia={statistics.mean(s.answer_lengths):.1f}, "
                        f"mediana={statistics.median(s.answer_lengths):.1f}")
        if s.context_lengths:
            lines.append(f"- Długość kontekstu (znaki): min={min(s.context_lengths)}, "
                        f"max={max(s.context_lengths)}, "
                        f"średnia={statistics.mean(s.context_lengths):.1f}, "
                        f"mediana={statistics.median(s.context_lengths):.1f}")
        if s.snippets_per_question:
            lines.append(f"- Snippety na pytanie: min={min(s.snippets_per_question)}, "
                        f"max={max(s.snippets_per_question)}, "
                        f"średnia={statistics.mean(s.snippets_per_question):.1f}")
        lines.append("")

    lines.append("## 4. Analiza odrzuconych pytań\n")
    for s in all_stats:
        if not s.not_matched_details:
            continue
        lines.append(f"### {s.split.upper()} — {s.not_matched} odrzuconych pytań factoid\n")
        lines.append("Pytania odrzucone to takie, gdzie żadna z 3 strategii nie znalazła `exact_answer`")
        lines.append("w żadnym snippecie. Typowe przyczyny:\n")
        lines.append("- Odpowiedź jest parafrazą (nie substring snippetu)")
        lines.append("- Odpowiedź jest zdaniem wyjaśniającym (nie krótkim spanem)")
        lines.append("- Literówki w gold answer")
        lines.append("- Odpowiedź odnosi się do informacji spoza dostarczonych snippetów\n")
        lines.append(f"Przykłady (pierwsze {min(10, len(s.not_matched_details))}):\n")
        for i, detail in enumerate(s.not_matched_details[:10]):
            lines.append(f"**{i+1}.** `{detail['id']}`")
            lines.append(f"- Pytanie: {detail['question']}")
            lines.append(f"- Gold answer: {detail['answers']}")
            lines.append(f"- Snippet (fragment): {detail['snippet_preview'][:150]}...")
            lines.append(f"- Snippetów: {detail['n_snippets']}")
            lines.append("")
    lines.append("")

    lines.append("## 5. Format wyjściowy\n")
    lines.append("Pliki zapisane w formacie SQuAD v2.0 (JSON):\n")
    lines.append("```")
    lines.append("data/processed/bioasq/")
    lines.append("├── train.json")
    lines.append("├── dev.json")
    lines.append("└── test.json")
    lines.append("```\n")
    lines.append("Każdy plik ma strukturę:")
    lines.append("```json")
    lines.append('{')
    lines.append('  "version": "v2.0",')
    lines.append('  "data": [')
    lines.append('    {')
    lines.append('      "title": "...",')
    lines.append('      "paragraphs": [{"context": "...", "qas": [{"id": "...", "question": "...",')
    lines.append('        "answers": [{"text": "...", "answer_start": N}], "is_impossible": false}]}]')
    lines.append('    }')
    lines.append('  ]')
    lines.append('}')
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    training_path = RAW_DIR / "BioASQ-training14b" / "BioASQ-training14b" / "training14b.json"
    golden_12b_dir = RAW_DIR / "Task12BGoldenEnriched"
    golden_13b_dir = RAW_DIR / "Task13BGoldenEnriched"

    for path, name in [
        (training_path, "training14b.json"),
        (golden_12b_dir, "Task12BGoldenEnriched"),
        (golden_13b_dir, "Task13BGoldenEnriched"),
    ]:
        if not path.exists():
            print(f"ERROR: Nie znaleziono {name} w {path}")
            return

    # 1. Załaduj golden enriched (dev + test)
    print("--- DEV (12b golden enriched) ---")
    dev_questions = load_golden_enriched(golden_12b_dir)

    print("\n--- TEST (13b golden enriched) ---")
    test_questions = load_golden_enriched(golden_13b_dir)

    # Zbierz ID pytań z golden sets
    golden_ids = set()
    for q in dev_questions + test_questions:
        golden_ids.add(q["id"])
    print(f"\nGolden question IDs to exclude from training: {len(golden_ids)}")

    # 2. Załaduj training14b i usuń pytania z golden sets
    print("\n--- TRAIN (training14b minus golden 12b+13b) ---")
    with open(training_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    all_training = training_data.get("questions", training_data)
    print(f"  Training14b total: {len(all_training)} questions")

    train_questions = [q for q in all_training if q["id"] not in golden_ids]
    excluded = len(all_training) - len(train_questions)
    print(f"  After excluding golden IDs: {len(train_questions)} questions (removed {excluded})")

    golden_exclusion_info = {
        "total_training": len(all_training),
        "golden_12b": len(dev_questions),
        "golden_13b": len(test_questions),
        "golden_ids": len(golden_ids),
        "excluded": excluded,
        "after_exclusion": len(train_questions),
    }

    # 3. Konwersja i zapis
    all_stats = []
    for split_name, questions in [("train", train_questions), ("dev", dev_questions), ("test", test_questions)]:
        stats = ConversionStats(split=split_name)
        print(f"\nConverting {split_name}...")
        squad_data = convert_bioasq_to_squad(questions, stats)
        save_json(squad_data, OUTPUT_DIR / f"{split_name}.json")

        summary = stats.summary_dict()
        for k, v in summary.items():
            print(f"  {k}: {v}")

        all_stats.append(stats)

    # 4. Generuj raport
    report = generate_report(all_stats, golden_exclusion_info)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
