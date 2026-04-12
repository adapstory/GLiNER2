"""Accuracy evaluation harness for GLiNER2 safety classification.

Measures precision, recall, F1, and accuracy of safety predictions
against labeled datasets (e.g., XSTest).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class EvalResult:
    """Confusion matrix counts with computed metrics."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"EvalResult(P={self.precision:.3f}, R={self.recall:.3f}, "
            f"F1={self.f1:.3f}, Acc={self.accuracy:.3f} | "
            f"TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn})"
        )


def load_labeled_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Load a labeled CSV with columns ``user_msg`` and ``expected_safety``.

    Returns:
        Tuple of (texts, labels).
    """
    texts: list[str] = []
    labels: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["user_msg"])
            labels.append(row["expected_safety"])
    return texts, labels


def evaluate_safety(
    texts: list[str],
    expected_labels: list[str],
    model_id: str = "hivetrace/gliner-guard-uniencoder",
    schema_builder: Callable | None = None,
    batch_size: int = 8,
) -> EvalResult:
    """Run safety classification and compare against expected labels.

    Positive class is ``unsafe``.

    Args:
        texts: Input texts to classify.
        expected_labels: Ground-truth labels (``safe`` / ``unsafe``).
        model_id: HuggingFace model identifier.
        schema_builder: Optional callable ``(model) -> schema``.
            If ``None``, a default schema with ``["safe", "unsafe"]`` is built.
        batch_size: Batch size for inference.

    Returns:
        EvalResult with confusion matrix counts.
    """
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_id)

    if schema_builder is not None:
        schema = schema_builder(model)
    else:
        schema = model.create_schema()
        schema = schema.classification(task="safety", labels=["safe", "unsafe"])

    results = model.batch_extract(texts=texts, schemas=schema, batch_size=batch_size)

    tp = fp = fn = tn = 0
    for expected, result in zip(expected_labels, results):
        predicted = result.get("safety", "safe")
        is_unsafe_expected = expected == "unsafe"
        is_unsafe_predicted = predicted == "unsafe"

        if is_unsafe_expected and is_unsafe_predicted:
            tp += 1
        elif not is_unsafe_expected and is_unsafe_predicted:
            fp += 1
        elif is_unsafe_expected and not is_unsafe_predicted:
            fn += 1
        else:
            tn += 1

    return EvalResult(tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# Schema variant builders for label-sensitivity analysis
# ---------------------------------------------------------------------------

_SAFETY_LABELS = ["safe", "unsafe"]

_PII_LABELS = [
    "person",
    "email",
    "phone_number",
    "address",
    "credit_card",
    "ssn",
    "date_of_birth",
    "ip_address",
]

_ADVERSARIAL_LABELS = [
    "jailbreak_attempt",
    "prompt_injection",
    "role_play_attack",
    "encoding_bypass",
    "multi_turn_manipulation",
    "social_engineering",
]

_TOPIC_LABELS = [
    "violence",
    "self_harm",
    "sexual_content",
    "hate_speech",
    "harassment",
    "illegal_activity",
    "weapons",
    "drugs",
    "financial_fraud",
    "child_exploitation",
    "terrorism",
    "misinformation",
    "privacy_violation",
    "copyright_infringement",
    "political_manipulation",
    "medical_misinformation",
    "gambling",
    "alcohol",
    "tobacco",
    "profanity",
    "discrimination",
    "stalking",
    "doxxing",
    "revenge_porn",
    "deepfake",
    "spam",
    "phishing",
    "malware",
    "ransomware",
    "identity_theft",
    "money_laundering",
    "tax_evasion",
    "human_trafficking",
    "animal_cruelty",
    "environmental_crime",
    "insider_trading",
    "bribery",
    "corruption",
    "extortion",
    "blackmail",
]

_SCHEMA_VARIANTS: dict[str, list[str]] = {
    "safety_only": _SAFETY_LABELS,
    "safety_pii": _SAFETY_LABELS + _PII_LABELS,
    "safety_pii_adversarial": _SAFETY_LABELS + _PII_LABELS + _ADVERSARIAL_LABELS,
    "full": _SAFETY_LABELS + _PII_LABELS + _ADVERSARIAL_LABELS + _TOPIC_LABELS,
}


def _make_schema_builder(labels: list[str]) -> Callable:
    """Create a schema builder that uses the given label set."""

    def builder(model):  # type: ignore[no-untyped-def]
        schema = model.create_schema()
        safety_labels = [l for l in labels if l in ("safe", "unsafe")]
        entity_labels = [l for l in labels if l not in ("safe", "unsafe")]
        if safety_labels:
            schema = schema.classification(task="safety", labels=safety_labels)
        if entity_labels:
            schema = schema.entities(entity_types=entity_labels, threshold=0.5)
        return schema

    return builder


def evaluate_safety_with_varying_labels(
    texts: list[str],
    expected_labels: list[str],
    model_id: str = "hivetrace/gliner-guard-uniencoder",
    batch_size: int = 8,
) -> dict[str, EvalResult]:
    """Run safety evaluation with 4 schema variants of increasing label count.

    Variants:
        - ``safety_only``: 2 labels (safe, unsafe)
        - ``safety_pii``: 10 labels
        - ``safety_pii_adversarial``: 16 labels
        - ``full``: 56 labels

    Returns:
        Dict mapping variant name to EvalResult.
    """
    results: dict[str, EvalResult] = {}
    for variant_name, labels in _SCHEMA_VARIANTS.items():
        print(f"[eval] Running variant '{variant_name}' ({len(labels)} labels)...")
        result = evaluate_safety(
            texts=texts,
            expected_labels=expected_labels,
            model_id=model_id,
            schema_builder=_make_schema_builder(labels),
            batch_size=batch_size,
        )
        results[variant_name] = result
        print(f"[eval]   {result}")
    return results


def main() -> None:
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate GLiNER2 safety classification accuracy."
    )
    parser.add_argument(
        "--model",
        default="hivetrace/gliner-guard-uniencoder",
        help="HuggingFace model ID (default: hivetrace/gliner-guard-uniencoder)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to labeled CSV (columns: user_msg, expected_safety)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--vary-labels",
        action="store_true",
        help="Run all 4 schema variants (safety_only → full)",
    )
    args = parser.parse_args()

    texts, labels = load_labeled_csv(args.dataset)
    print(f"Loaded {len(texts)} samples from {args.dataset}")

    if args.vary_labels:
        results = evaluate_safety_with_varying_labels(
            texts=texts,
            expected_labels=labels,
            model_id=args.model,
            batch_size=args.batch_size,
        )
        print("\n=== Summary ===")
        for name, result in results.items():
            print(f"  {name:30s} {result}")
    else:
        result = evaluate_safety(
            texts=texts,
            expected_labels=labels,
            model_id=args.model,
            batch_size=args.batch_size,
        )
        print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
