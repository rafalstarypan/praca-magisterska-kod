"""Data loading and tokenization for extractive QA."""

import json
from pathlib import Path


def load_squad_examples(path):
    """Load SQuAD v2.0 JSON and flatten to list of examples.

    Returns list of dicts with keys: id, question, context, answers.
    answers is a dict with keys: text (list[str]), answer_start (list[int]).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append({
                    "id": str(qa["id"]),
                    "question": qa["question"],
                    "context": context,
                    "answers": {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    },
                })
    return examples


def tokenize_examples(examples, tokenizer, max_seq_length, doc_stride):
    """Tokenize QA examples with sliding window for long contexts.

    Designed for use with datasets.Dataset.map(batched=True).
    Returns all columns needed for both training and evaluation:
      - input_ids, attention_mask, [token_type_ids]: model inputs
      - start_positions, end_positions: answer span labels
      - example_id: maps each feature back to its source example
      - offset_mapping: token-to-char positions (None for non-context tokens)
    """
    # Pre-truncate questions that are too long for the sequence budget.
    # With truncation="only_second" only the context is truncated, so if
    # the question alone exceeds max_length minus special tokens the
    # tokenizer raises an error.  This affects BPE tokenizers (RoBERTa)
    # more than WordPiece (BERT) because they produce more sub-tokens.
    max_question_tokens = max_seq_length // 2
    questions = []
    for q in examples["question"]:
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        if len(q_ids) > max_question_tokens:
            q_ids = q_ids[:max_question_tokens]
            q = tokenizer.decode(q_ids, skip_special_tokens=True)
        questions.append(q)

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = []
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sample_index = sample_mapping[i]
        sequence_ids = tokenized.sequence_ids(i)

        tokenized["example_id"].append(examples["id"][sample_index])

        # --- answer span labels ---
        answers = examples["answers"][sample_index]
        answer_starts = answers["answer_start"]
        answer_texts = answers["text"]
        cls_index = 0

        if len(answer_starts) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])

            # Find first and last context token
            token_start = 0
            while sequence_ids[token_start] != 1:
                token_start += 1
            token_end = len(offsets) - 1
            while sequence_ids[token_end] != 1:
                token_end -= 1

            # Check if answer span fits in this window
            if offsets[token_start][0] > start_char or offsets[token_end][1] < end_char:
                # Answer not in this window — point to CLS
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                # Find start token: last token whose char-start <= answer start
                idx = token_start
                while idx <= token_end and offsets[idx][0] <= start_char:
                    idx += 1
                tokenized["start_positions"].append(idx - 1)

                # Find end token: first token whose char-end >= answer end
                idx = token_end
                while idx >= token_start and offsets[idx][1] >= end_char:
                    idx -= 1
                tokenized["end_positions"].append(idx + 1)

        # Null out non-context offsets (question, special tokens, padding)
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offsets)
        ]

    return tokenized
