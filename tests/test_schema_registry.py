"""Tests for SchemaRegistry — dynamic label management for plugin-first platform."""

from __future__ import annotations

import logging

import pytest

from gliner2.inference.schema_registry import SchemaRegistry


class TestRegisterClassificationTask:
    """test_register_classification_task — register task, verify it's stored."""

    def test_register_classification_task(self) -> None:
        registry = SchemaRegistry()
        registry.register_classification("safety", labels=["safe", "unsafe"])

        assert "safety" in registry.classification_tasks
        task = registry.classification_tasks["safety"]
        assert set(task["labels"]) == {"safe", "unsafe"}


class TestRegisterEntities:
    """test_register_entities — register entity types, verify list."""

    def test_register_entities(self) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person", "email"])

        assert "person" in registry.entity_types
        assert "email" in registry.entity_types


class TestRegisterDuplicateClassificationMerges:
    """test_register_duplicate_classification_merges — same task twice merges labels."""

    def test_register_duplicate_classification_merges(self) -> None:
        registry = SchemaRegistry()
        registry.register_classification("safety", labels=["safe", "unsafe"])
        registry.register_classification("safety", labels=["unsafe", "harmful"])

        task = registry.classification_tasks["safety"]
        assert set(task["labels"]) == {"safe", "unsafe", "harmful"}


class TestTotalLabelCount:
    """test_total_label_count — 3 entities + 2 safety + 3 intent = 8."""

    def test_total_label_count(self) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person", "email", "org"])
        registry.register_classification("safety", labels=["safe", "unsafe"])
        registry.register_classification("intent", labels=["buy", "sell", "hold"])

        assert registry.total_label_count == 8


class TestBuildSchemaReturnsSchemaObject:
    """test_build_schema_returns_schema_object — requires model download."""

    @pytest.mark.slow
    def test_build_schema_returns_schema_object(
        self, gliner2_model: object
    ) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person", "email"])
        registry.register_classification(
            "safety", labels=["safe", "unsafe"], multi_label=False, cls_threshold=0.6
        )

        schema = registry.build_schema(gliner2_model)  # type: ignore[arg-type]
        schema_dict = schema.build()

        # entities key present with registered types
        assert "person" in schema_dict["entities"]
        assert "email" in schema_dict["entities"]

        # classifications key present with registered task
        assert len(schema_dict["classifications"]) == 1
        cls_task = schema_dict["classifications"][0]
        assert cls_task["task"] == "safety"
        assert set(cls_task["labels"]) == {"safe", "unsafe"}


class TestSchemaCacheKeyStable:
    """test_schema_cache_key_stable — same state produces same cache_key."""

    def test_schema_cache_key_stable(self) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person", "email"])
        registry.register_classification("safety", labels=["safe", "unsafe"])

        key1 = registry.cache_key
        key2 = registry.cache_key

        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest


class TestSchemaCacheKeyChangesOnMutation:
    """test_schema_cache_key_changes_on_mutation — adding labels changes key."""

    def test_schema_cache_key_changes_on_mutation(self) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person"])
        key_before = registry.cache_key

        registry.register_entities(["email"])
        key_after = registry.cache_key

        assert key_before != key_after


class TestWarnsWhenExceedingLabelBudget:
    """test_warns_when_exceeding_label_budget — warns when 11th label added."""

    def test_warns_when_exceeding_label_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = SchemaRegistry(max_labels=10)
        # Register 10 entities — no warning
        registry.register_entities([f"entity_{i}" for i in range(10)])
        assert registry.total_label_count == 10

        # Register 11th — should warn
        with caplog.at_level(logging.WARNING):
            registry.register_entities(["overflow_entity"])

        assert registry.total_label_count == 11
        assert any("exceeds" in record.message.lower() or "budget" in record.message.lower()
                    for record in caplog.records)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gliner2_model():  # type: ignore[no-untyped-def]
    """Load GLiNER2 model — slow, downloads weights on first run."""
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained("fastino/gliner2-base-v1")


class TestRegisterEntitiesFluent:
    """Verify fluent (chained) API returns self."""

    def test_register_entities_returns_self(self) -> None:
        registry = SchemaRegistry()
        result = registry.register_entities(["person"])
        assert result is registry

    def test_register_classification_returns_self(self) -> None:
        registry = SchemaRegistry()
        result = registry.register_classification("safety", labels=["safe"])
        assert result is registry


class TestRegisterEntitiesDeduplication:
    """Registering the same entity twice doesn't duplicate."""

    def test_no_duplicate_entities(self) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person", "email"])
        registry.register_entities(["email", "org"])

        assert registry.entity_types.count("email") == 1
        assert set(registry.entity_types) == {"person", "email", "org"}


class TestSummary:
    """summary() returns a human-readable string."""

    def test_summary_contains_counts(self) -> None:
        registry = SchemaRegistry()
        registry.register_entities(["person"])
        registry.register_classification("safety", labels=["safe", "unsafe"])

        s = registry.summary()
        assert "1" in s  # 1 entity
        assert "safety" in s or "classification" in s.lower()
