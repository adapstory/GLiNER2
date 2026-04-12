"""Dynamic Schema Registry for plugin-first label management.

Plugins register their entity types and classification tasks at startup.
The registry produces a cached schema that can be built into a GLiNER2 Schema.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gliner2.inference.engine import GLiNER2, Schema

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Dynamic registry for entity types and classification tasks.

    Plugins call ``register_entities`` and ``register_classification`` at
    startup. The registry deduplicates labels, tracks a budget, and can
    produce a ``Schema`` object ready for ``GLiNER2`` inference.

    Args:
        max_labels: Warn when total label count exceeds this budget.
    """

    def __init__(self, max_labels: int = 100) -> None:
        self._max_labels = max_labels
        self._entity_types: list[str] = []
        self._entity_threshold: float = 0.5
        self._classification_tasks: dict[str, dict[str, Any]] = {}
        self._cache_key: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entity_types(self) -> list[str]:
        """Return a copy of registered entity types."""
        return list(self._entity_types)

    @property
    def classification_tasks(self) -> dict[str, dict[str, Any]]:
        """Return a copy of registered classification tasks."""
        return {k: dict(v) for k, v in self._classification_tasks.items()}

    @property
    def total_label_count(self) -> int:
        """Total number of unique labels across entities and classifications."""
        cls_count = sum(
            len(task["labels"]) for task in self._classification_tasks.values()
        )
        return len(self._entity_types) + cls_count

    @property
    def cache_key(self) -> str:
        """SHA-256 hex digest of current registry state. Cached until mutation."""
        if self._cache_key is None:
            canonical = json.dumps(
                {
                    "entities": sorted(self._entity_types),
                    "classifications": {
                        k: {
                            "labels": sorted(v["labels"]),
                            "multi_label": v["multi_label"],
                            "cls_threshold": v["cls_threshold"],
                        }
                        for k, v in sorted(self._classification_tasks.items())
                    },
                },
                sort_keys=True,
            )
            self._cache_key = hashlib.sha256(canonical.encode()).hexdigest()
        return self._cache_key

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def register_entities(
        self, entity_types: list[str], threshold: float = 0.5
    ) -> SchemaRegistry:
        """Append entity types without duplicates.

        Args:
            entity_types: Entity type names to register.
            threshold: Confidence threshold for extraction.

        Returns:
            Self for fluent chaining.
        """
        self._invalidate_cache()
        self._entity_threshold = threshold
        for et in entity_types:
            if et not in self._entity_types:
                self._entity_types.append(et)
        self._check_budget()
        return self

    def register_classification(
        self,
        task: str,
        labels: list[str],
        multi_label: bool = False,
        cls_threshold: float = 0.5,
    ) -> SchemaRegistry:
        """Register or merge a classification task.

        If the task already exists, new labels are merged (deduplicated).

        Args:
            task: Classification task name (e.g. "safety", "intent").
            labels: Label names for this task.
            multi_label: Whether multiple labels can be active simultaneously.
            cls_threshold: Confidence threshold for classification.

        Returns:
            Self for fluent chaining.
        """
        self._invalidate_cache()
        if task in self._classification_tasks:
            existing = self._classification_tasks[task]
            merged = list(existing["labels"])
            for label in labels:
                if label not in merged:
                    merged.append(label)
            existing["labels"] = merged
        else:
            self._classification_tasks[task] = {
                "labels": list(labels),
                "multi_label": multi_label,
                "cls_threshold": cls_threshold,
            }
        self._check_budget()
        return self

    # ------------------------------------------------------------------
    # Schema building
    # ------------------------------------------------------------------

    def build_schema(self, model: GLiNER2) -> Schema:
        """Build a ``Schema`` object from the current registry state.

        Uses ``model.create_schema()`` then chains ``.entities()`` and
        ``.classification()`` calls for each registered item.

        Args:
            model: A loaded GLiNER2 model instance.

        Returns:
            A fully configured Schema ready for ``.build()`` or inference.
        """
        schema = model.create_schema()

        if self._entity_types:
            schema = schema.entities(
                entity_types=list(self._entity_types),
                threshold=self._entity_threshold,
            )

        for task_name, task_cfg in self._classification_tasks.items():
            schema = schema.classification(
                task=task_name,
                labels=list(task_cfg["labels"]),
                multi_label=task_cfg["multi_label"],
                cls_threshold=task_cfg["cls_threshold"],
            )

        return schema

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the registry state."""
        lines = [
            f"SchemaRegistry: {self.total_label_count} total labels "
            f"(budget: {self._max_labels})",
            f"  Entities ({len(self._entity_types)}): "
            f"{', '.join(self._entity_types) or '(none)'}",
        ]
        for task_name, task_cfg in self._classification_tasks.items():
            label_str = ", ".join(task_cfg["labels"])
            lines.append(
                f"  Classification '{task_name}' ({len(task_cfg['labels'])} labels): "
                f"{label_str}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        """Reset cached key so it's recomputed on next access."""
        self._cache_key = None

    def _check_budget(self) -> None:
        """Warn if total label count exceeds the configured budget."""
        count = self.total_label_count
        if count > self._max_labels:
            logger.warning(
                "Label count %d exceeds budget of %d. "
                "Consider increasing max_labels or reducing registered labels.",
                count,
                self._max_labels,
            )
