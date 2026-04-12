"""Download SQuAD v1.1 from HuggingFace and save in project format.

SQuAD v1.1 is used as a pretraining step before domain-specific fine-tuning
(BioASQ, COVID-QA). It teaches models the mechanics of extractive QA on
~87k examples before specializing on small biomedical datasets (~1k examples).

Usage:
    python data/prepare_squad.py
"""

import json
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "processed" / "squad"


def hf_to_squad_json(hf_dataset):
    """Convert HuggingFace flat dataset to nested SQuAD JSON format."""
    articles = OrderedDict()

    for ex in hf_dataset:
        title = ex["title"]
        context = ex["context"]

        if title not in articles:
            articles[title] = OrderedDict()
        if context not in articles[title]:
            articles[title][context] = []

        articles[title][context].append({
            "id": ex["id"],
            "question": ex["question"],
            "answers": [
                {"text": t, "answer_start": s}
                for t, s in zip(ex["answers"]["text"], ex["answers"]["answer_start"])
            ],
            "is_impossible": False,
        })

    return {
        "version": "v1.1",
        "data": [
            {
                "title": title,
                "paragraphs": [
                    {"context": ctx, "qas": qas}
                    for ctx, qas in contexts.items()
                ],
            }
            for title, contexts in articles.items()
        ],
    }


def main():
    print("Loading SQuAD v1.1 from HuggingFace...")
    dataset = load_dataset("rajpurkar/squad")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, hf_key in [("train", "train"), ("dev", "validation")]:
        hf_split = dataset[hf_key]
        print(f"  {split_name}: {len(hf_split)} examples")

        squad_json = hf_to_squad_json(hf_split)

        n_articles = len(squad_json["data"])
        n_paragraphs = sum(len(a["paragraphs"]) for a in squad_json["data"])
        n_qas = sum(
            len(p["qas"])
            for a in squad_json["data"]
            for p in a["paragraphs"]
        )
        print(f"  → {n_articles} articles, {n_paragraphs} paragraphs, {n_qas} QA pairs")

        output_path = OUTPUT_DIR / f"{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(squad_json, f, ensure_ascii=False)

        size_mb = output_path.stat().st_size / 1e6
        print(f"  → Saved to {output_path} ({size_mb:.1f} MB)")

    print("Done!")


if __name__ == "__main__":
    main()
