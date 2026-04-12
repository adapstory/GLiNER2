"""Create a rough-labeled version of XSTest for safety evaluation.

XSTest convention (Rottger et al., 2023):
  - First ~250 rows: safe prompts (homonyms, figurative language, etc.)
  - Remaining rows: unsafe prompts

This script applies the automatic split. Manual review is needed
for paper-quality labels — the boundary is approximate.

Usage:
    python -m gliner2.eval.datasets.create_xstest_labels

Reads:  ../../gliner-guard-serve/test-script/xstest.csv
Writes: gliner2/eval/datasets/xstest_labeled.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

# Number of rows considered safe in XSTest (homonyms, figurative, etc.)
SAFE_BOUNDARY = 250


def create_xstest_labels(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    safe_boundary: int = SAFE_BOUNDARY,
) -> Path:
    """Read raw XSTest CSV and write labeled version.

    Args:
        input_path: Path to raw xstest.csv. Defaults to standard location.
        output_path: Path to write labeled CSV. Defaults to datasets dir.
        safe_boundary: Row index boundary between safe and unsafe.

    Returns:
        Path to the written labeled CSV.
    """
    if input_path is None:
        # Relative to this file: GLiNER2/gliner2/eval/datasets/
        # Target: GLiNER2/../../gliner-guard-serve/test-script/xstest.csv
        this_dir = Path(__file__).resolve().parent
        gliner2_root = this_dir.parent.parent.parent
        input_path = (
            gliner2_root.parent / "gliner-guard-serve" / "test-script" / "xstest.csv"
        )

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "xstest_labeled.csv"

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"XSTest CSV not found at {input_path}. "
            "Expected column: user_msg"
        )

    rows: list[dict[str, str]] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            label = "safe" if idx < safe_boundary else "unsafe"
            rows.append(
                {"user_msg": row["user_msg"], "expected_safety": label}
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_msg", "expected_safety"])
        writer.writeheader()
        writer.writerows(rows)

    safe_count = sum(1 for r in rows if r["expected_safety"] == "safe")
    unsafe_count = len(rows) - safe_count
    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"(safe={safe_count}, unsafe={unsafe_count})"
    )
    print("NOTE: This is a rough automatic split. Manual review needed for paper quality.")

    return output_path


if __name__ == "__main__":
    create_xstest_labels()
