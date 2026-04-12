"""Evaluation utilities for extractive QA (SQuAD-style EM and F1)."""

import re
import string
from collections import Counter, defaultdict

import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def compute_exact(gold, pred):
    """Exact Match after normalization."""
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold, pred):
    """Token-level F1 after normalization."""
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()

    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(predictions, examples):
    """Compute EM and F1 over all examples (max over gold answers).

    Args:
        predictions: dict mapping example_id -> predicted answer text
        examples: list of example dicts with 'id' and 'answers' keys

    Returns:
        dict with 'exact_match', 'f1' (percentages) and 'total'
    """
    exact_scores = []
    f1_scores = []

    for ex in examples:
        pred = predictions.get(ex["id"], "")
        gold_answers = ex["answers"]["text"]

        if not gold_answers:
            exact_scores.append(int(pred == ""))
            f1_scores.append(1.0 if pred == "" else 0.0)
            continue

        exact_scores.append(max(compute_exact(g, pred) for g in gold_answers))
        f1_scores.append(max(compute_f1(g, pred) for g in gold_answers))

    return {
        "exact_match": 100.0 * sum(exact_scores) / len(exact_scores),
        "f1": 100.0 * sum(f1_scores) / len(f1_scores),
        "total": len(examples),
    }


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    n_best_size=20,
    max_answer_length=30,
):
    """Convert raw model logits to text predictions.

    Args:
        examples: list of original example dicts (from load_squad_examples)
        features: HF Dataset with 'example_id' and 'offset_mapping' columns
        raw_predictions: tuple (start_logits, end_logits), each shape (num_features, seq_len)
        n_best_size: how many top start/end positions to consider per feature
        max_answer_length: max answer span length in tokens

    Returns:
        dict mapping example_id -> predicted answer text
    """
    all_start_logits, all_end_logits = raw_predictions

    # Map example id -> index in examples list
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}

    # Map example index -> list of feature indices
    features_per_example = defaultdict(list)
    all_example_ids = features["example_id"]
    for feat_idx, eid in enumerate(all_example_ids):
        features_per_example[example_id_to_index[eid]].append(feat_idx)

    all_offset_mappings = features["offset_mapping"]
    predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = all_offset_mappings[feature_index]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1].tolist()
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[
                            offset_mapping[start_index][0] : offset_mapping[end_index][1]
                        ],
                    })

        if valid_answers:
            predictions[example["id"]] = max(
                valid_answers, key=lambda x: x["score"]
            )["text"]
        else:
            predictions[example["id"]] = ""

    return predictions
