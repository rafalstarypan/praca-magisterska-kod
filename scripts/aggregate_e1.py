"""
Aggregate E1 results into summary tables (Markdown + CSV).

Usage:
    python scripts/aggregate_e1.py

Outputs:
    results/e1_summary.md   — Markdown tables for thesis
    results/e1_summary.csv  — CSV for further analysis
"""

import json
import os
import statistics
import csv
from pathlib import Path

RESULTS_DIR = Path("results")

MODEL_ORDER = [
    "bert",
    "biobert",
    "pubmedbert",
    "roberta",
    "distilbert",
    "clinicalbert",
]

MODEL_DISPLAY = {
    "bert": "BERT",
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT",
    "clinicalbert": "ClinicalBERT",
}

MODEL_HF = {
    "bert": "google-bert/bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


def load_bioasq(model: str, variant: str) -> dict | None:
    """Load BioASQ results for model/variant."""
    dirname = model if variant == "with_squad" else f"{model}_no_squad"
    path = RESULTS_DIR / "bioasq" / dirname / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_covidqa_folds(model: str, variant: str) -> list[dict]:
    """Load all COVID-QA fold results for model/variant."""
    dirname = model if variant == "with_squad" else f"{model}_no_squad"
    folds = []
    for i in range(5):
        path = RESULTS_DIR / "covidqa" / dirname / f"fold_{i}" / "results.json"
        if path.exists():
            folds.append(json.loads(path.read_text(encoding="utf-8")))
    return folds


def load_squad(model: str) -> dict | None:
    """Load SQuAD pretraining results."""
    path = RESULTS_DIR / "squad" / model / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(val: float, prec: int = 2) -> str:
    return f"{val:.{prec}f}"


def fmt_pm(mean: float, std: float, prec: int = 2) -> str:
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


def main():
    lines = []
    csv_rows = []

    def add(text: str = ""):
        lines.append(text)

    # ── Table 1: SQuAD pretraining ──
    add("# E1 — Wyniki eksperymentów (podsumowanie)")
    add()
    add("## 1. Pretrening na SQuAD v1.1")
    add()
    add("| Model | Params | Eval EM | Eval F1 |")
    add("|-------|--------|---------|---------|")

    for m in MODEL_ORDER:
        r = load_squad(m)
        if r is None:
            continue
        params = f"{r['total_params']:,}"
        em = fmt(r["eval_metrics"]["exact_match"])
        f1 = fmt(r["eval_metrics"]["f1"])
        add(f"| {MODEL_DISPLAY[m]} | {params} | {em} | {f1} |")
        csv_rows.append({
            "dataset": "squad",
            "model": MODEL_DISPLAY[m],
            "variant": "pretraining",
            "params": r["total_params"],
            "eval_em": r["eval_metrics"]["exact_match"],
            "eval_f1": r["eval_metrics"]["f1"],
            "test_em": None,
            "test_f1": None,
            "test_em_std": None,
            "test_f1_std": None,
        })

    # ── Table 2: BioASQ ──
    add()
    add("## 2. BioASQ (factoid)")
    add()
    add("| Model | Wariant | Eval EM | Eval F1 | Test EM | Test F1 |")
    add("|-------|---------|---------|---------|---------|---------|")

    for m in MODEL_ORDER:
        for v in ["no_squad", "with_squad"]:
            r = load_bioasq(m, v)
            if r is None:
                continue
            label = "z SQuAD" if v == "with_squad" else "bez SQuAD"
            ee = fmt(r["eval_metrics"]["exact_match"])
            ef = fmt(r["eval_metrics"]["f1"])
            te = fmt(r["test_metrics"]["exact_match"])
            tf = fmt(r["test_metrics"]["f1"])
            add(f"| {MODEL_DISPLAY[m]} | {label} | {ee} | {ef} | {te} | {tf} |")
            csv_rows.append({
                "dataset": "bioasq",
                "model": MODEL_DISPLAY[m],
                "variant": label,
                "params": r["total_params"],
                "eval_em": r["eval_metrics"]["exact_match"],
                "eval_f1": r["eval_metrics"]["f1"],
                "test_em": r["test_metrics"]["exact_match"],
                "test_f1": r["test_metrics"]["f1"],
                "test_em_std": None,
                "test_f1_std": None,
            })

    # ── Table 3: BioASQ improvement from SQuAD ──
    add()
    add("### Poprawa dzięki pretreningowi SQuAD (BioASQ)")
    add()
    add("| Model | Test F1 bez SQuAD | Test F1 z SQuAD | Δ F1 |")
    add("|-------|-------------------|-----------------|------|")

    for m in MODEL_ORDER:
        r_no = load_bioasq(m, "no_squad")
        r_sq = load_bioasq(m, "with_squad")
        if r_no and r_sq:
            f1_no = r_no["test_metrics"]["f1"]
            f1_sq = r_sq["test_metrics"]["f1"]
            delta = f1_sq - f1_no
            add(f"| {MODEL_DISPLAY[m]} | {fmt(f1_no)} | {fmt(f1_sq)} | +{fmt(delta)} |")

    # ── Table 4: COVID-QA ──
    add()
    add("## 3. COVID-QA (5-fold CV)")
    add()
    add("| Model | Wariant | Eval EM | Eval F1 | Test EM | Test F1 |")
    add("|-------|---------|---------|---------|---------|---------|")

    for m in MODEL_ORDER:
        for v in ["no_squad", "with_squad"]:
            folds = load_covidqa_folds(m, v)
            if len(folds) != 5:
                continue
            label = "z SQuAD" if v == "with_squad" else "bez SQuAD"

            eval_ems = [f["eval_metrics"]["exact_match"] for f in folds]
            eval_f1s = [f["eval_metrics"]["f1"] for f in folds]
            test_ems = [f["test_metrics"]["exact_match"] for f in folds]
            test_f1s = [f["test_metrics"]["f1"] for f in folds]

            ee = fmt_pm(statistics.mean(eval_ems), statistics.stdev(eval_ems))
            ef = fmt_pm(statistics.mean(eval_f1s), statistics.stdev(eval_f1s))
            te = fmt_pm(statistics.mean(test_ems), statistics.stdev(test_ems))
            tf = fmt_pm(statistics.mean(test_f1s), statistics.stdev(test_f1s))

            add(f"| {MODEL_DISPLAY[m]} | {label} | {ee} | {ef} | {te} | {tf} |")
            csv_rows.append({
                "dataset": "covidqa",
                "model": MODEL_DISPLAY[m],
                "variant": label,
                "params": folds[0]["total_params"],
                "eval_em": statistics.mean(eval_ems),
                "eval_f1": statistics.mean(eval_f1s),
                "test_em": statistics.mean(test_ems),
                "test_f1": statistics.mean(test_f1s),
                "test_em_std": statistics.stdev(test_ems),
                "test_f1_std": statistics.stdev(test_f1s),
            })

    # ── Table 5: COVID-QA improvement from SQuAD ──
    add()
    add("### Poprawa dzięki pretreningowi SQuAD (COVID-QA)")
    add()
    add("| Model | Test F1 bez SQuAD | Test F1 z SQuAD | Δ F1 |")
    add("|-------|-------------------|-----------------|------|")

    for m in MODEL_ORDER:
        folds_no = load_covidqa_folds(m, "no_squad")
        folds_sq = load_covidqa_folds(m, "with_squad")
        if len(folds_no) == 5 and len(folds_sq) == 5:
            f1_no = statistics.mean([f["test_metrics"]["f1"] for f in folds_no])
            f1_sq = statistics.mean([f["test_metrics"]["f1"] for f in folds_sq])
            delta = f1_sq - f1_no
            no_std = statistics.stdev([f["test_metrics"]["f1"] for f in folds_no])
            sq_std = statistics.stdev([f["test_metrics"]["f1"] for f in folds_sq])
            add(f"| {MODEL_DISPLAY[m]} | {fmt_pm(f1_no, no_std)} | {fmt_pm(f1_sq, sq_std)} | +{fmt(delta)} |")

    # ── Table 6: Final ranking (Test F1 with SQuAD) ──
    add()
    add("## 4. Ranking końcowy (Test F1, z pretreningiem SQuAD)")
    add()
    add("| # | Model | BioASQ F1 | COVID-QA F1 | Średnia F1 |")
    add("|---|-------|-----------|-------------|------------|")

    ranking = []
    for m in MODEL_ORDER:
        r_bio = load_bioasq(m, "with_squad")
        folds_cov = load_covidqa_folds(m, "with_squad")
        if r_bio and len(folds_cov) == 5:
            bio_f1 = r_bio["test_metrics"]["f1"]
            cov_f1 = statistics.mean([f["test_metrics"]["f1"] for f in folds_cov])
            avg = (bio_f1 + cov_f1) / 2
            ranking.append((m, bio_f1, cov_f1, avg))

    ranking.sort(key=lambda x: x[3], reverse=True)
    for i, (m, bio_f1, cov_f1, avg) in enumerate(ranking, 1):
        add(f"| {i} | {MODEL_DISPLAY[m]} | {fmt(bio_f1)} | {fmt(cov_f1)} | {fmt(avg)} |")

    # ── Write outputs ──
    md_path = RESULTS_DIR / "e1_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {md_path}")

    csv_path = RESULTS_DIR / "e1_summary.csv"
    fieldnames = [
        "dataset", "model", "variant", "params",
        "eval_em", "eval_f1", "test_em", "test_f1",
        "test_em_std", "test_f1_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
