"""Benchmark: Accuracy vs Label Count for GLiNER2 Safety Classification.

Measures whether adding more classification labels (PII, adversarial, topic)
degrades the core safety classification quality (F1, precision, recall, accuracy).

Motivation:
  A uni-encoder model encodes schema + text jointly. When the label set grows
  from 2 (safe/unsafe) to 56 (full taxonomy), the combined sequence length
  increases significantly. This benchmark quantifies whether that additional
  context degrades the model's ability to correctly classify safety.

Test matrix (4 schema variants):
  - safety_only:           2 labels  (baseline)
  - safety_pii:           10 labels
  - safety_pii_adversarial: 16 labels
  - full:                 56 labels

Protocol:
  - Load labeled CSV (user_msg, expected_safety)
  - Run each variant on the full dataset
  - Report F1 delta vs baseline (safety_only)
  - Warn if F1 drops > 0.05 from baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _build_row(
    schema: str,
    result: object,
    baseline_f1: float | None,
) -> tuple[str, ...]:
    """Format a single result row for tabular display."""
    f1: float = result.f1  # type: ignore[attr-defined]
    prec: float = result.precision  # type: ignore[attr-defined]
    rec: float = result.recall  # type: ignore[attr-defined]
    acc: float = result.accuracy  # type: ignore[attr-defined]
    tp: int = result.tp  # type: ignore[attr-defined]
    fp: int = result.fp  # type: ignore[attr-defined]
    fn: int = result.fn  # type: ignore[attr-defined]
    tn: int = result.tn  # type: ignore[attr-defined]

    delta_str = ""
    if baseline_f1 is not None:
        delta = f1 - baseline_f1
        sign = "+" if delta >= 0 else ""
        delta_str = f"({sign}{delta:+.3f})"

    return (
        schema,
        f"{f1:.3f}",
        f"{prec:.3f}",
        f"{rec:.3f}",
        f"{acc:.3f}",
        str(tp),
        str(fp),
        str(fn),
        str(tn),
        delta_str,
    )


def _print_table(rows: list[tuple[str, ...]]) -> None:
    """Print a formatted ASCII table of benchmark results."""
    headers = ("Schema", "F1", "Prec", "Rec", "Acc", "TP", "FP", "FN", "TN", "Delta")
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |")
    print(sep)


def main() -> int:
    """CLI entrypoint for the accuracy vs labels benchmark.

    Returns:
        Exit code (0 = success, 1 = degradation detected).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GLiNER2 safety classification accuracy across schema variants "
            "with increasing label counts. Measures whether label bloat degrades F1."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="hivetrace/gliner-guard-uniencoder",
        help="HuggingFace model ID to evaluate",
    )
    parser.add_argument(
        "--dataset",
        default="gliner2/eval/datasets/xstest_labeled.csv",
        help="Path to labeled CSV (columns: user_msg, expected_safety)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        # Resolve relative to the GLiNER2 project root (two levels up from benchmarks/)
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / dataset_path

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    # Lazy import — allows `python -c "import benchmarks.benchmark_accuracy_vs_labels"`
    # without requiring gliner2 to be installed in the current env.
    from gliner2.eval.evaluate import (  # type: ignore[import]
        evaluate_safety_with_varying_labels,
        load_labeled_csv,
    )

    print(f"Model    : {args.model}")
    print(f"Dataset  : {dataset_path}")
    print(f"Batch    : {args.batch_size}")
    print()

    texts, labels = load_labeled_csv(dataset_path)
    print(f"Loaded {len(texts)} samples.\n")

    start_time = time.perf_counter()

    results: dict[str, object] = evaluate_safety_with_varying_labels(
        texts=texts,
        expected_labels=labels,
        model_id=args.model,
        batch_size=args.batch_size,
    )

    elapsed = time.perf_counter() - start_time

    # Build table rows
    baseline_f1: float | None = None
    rows: list[tuple[str, ...]] = []
    for schema_name, result in results.items():
        row = _build_row(schema_name, result, baseline_f1)
        rows.append(row)
        if baseline_f1 is None:
            baseline_f1 = result.f1  # type: ignore[attr-defined]

    print("\n=== Accuracy vs Label Count ===\n")
    _print_table(rows)
    print()

    # Degradation analysis
    degradation_found = False
    if baseline_f1 is not None:
        for schema_name, result in results.items():
            if schema_name == "safety_only":
                continue
            f1: float = result.f1  # type: ignore[attr-defined]
            drop = baseline_f1 - f1
            if drop > 0.05:
                print(
                    f"WARNING: F1 degradation detected for '{schema_name}': "
                    f"dropped {drop:.3f} from baseline {baseline_f1:.3f} → {f1:.3f}"
                )
                degradation_found = True

    print(f"Total time: {elapsed:.1f}s")

    return 1 if degradation_found else 0


if __name__ == "__main__":
    sys.exit(main())
