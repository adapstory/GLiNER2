"""
Tests for sequence length guard: warning on overflow and auto-truncation.

Validates that the processor warns when combined schema + text tokens exceed
the encoder's max_position_embeddings, and auto-truncates text (not schema)
when max_len is not explicitly set.
"""

import logging

import pytest

from gliner2 import GLiNER2


@pytest.fixture(scope="module")
def model():
    """Load model once for the module."""
    return GLiNER2.from_pretrained("fastino/gliner2-base-v1")


def _make_entity_schema(labels):
    """Build a valid entity schema dict from a list of label names."""
    return {"entities": {label: "" for label in labels}}


def _make_batch(model, text, schema, max_len=None):
    """Helper: run the processor collate pipeline and return a PreprocessedBatch."""
    return model.processor._collate_batch(
        [(text, schema)],
        max_len=max_len,
    )


def test_warns_when_sequence_exceeds_max_position_embeddings(model, caplog):
    """60 entity labels + 400-word text with explicit max_len should trigger a WARNING log.

    We pass max_len=400 to bypass auto-truncation so the combined
    schema + text sequence exceeds max_position_embeddings.
    """
    labels = [f"entity_type_{i}" for i in range(60)]
    words = [f"word{i}" for i in range(400)]
    long_text = " ".join(words)
    schema = _make_entity_schema(labels)

    with caplog.at_level(logging.WARNING, logger="gliner2.processor"):
        _make_batch(model, long_text, schema, max_len=400)

    warning_messages = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and "max_position_embeddings" in r.message
    ]
    assert len(warning_messages) >= 1, (
        f"Expected a warning about max_position_embeddings, got: {caplog.text}"
    )


def test_short_text_no_warning(model, caplog):
    """Short text + few labels should NOT warn."""
    labels = ["person", "location"]
    short_text = "Alice lives in Paris."
    schema = _make_entity_schema(labels)

    with caplog.at_level(logging.WARNING, logger="gliner2.processor"):
        _make_batch(model, short_text, schema)

    warning_messages = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and "max_position_embeddings" in r.message
    ]
    assert len(warning_messages) == 0, (
        f"Expected no warning about max_position_embeddings, got: {caplog.text}"
    )


def test_truncates_text_not_schema_when_auto_max_len(model):
    """With auto-truncation, final input_ids.shape[1] must be <= max_position_embeddings."""
    labels = [f"entity_type_{i}" for i in range(56)]
    words = [f"word{i}" for i in range(600)]
    long_text = " ".join(words)
    schema = _make_entity_schema(labels)

    batch = _make_batch(model, long_text, schema)

    max_pos = model.encoder.config.max_position_embeddings
    seq_len = batch.input_ids.shape[1]

    assert seq_len <= max_pos, (
        f"input_ids seq_len {seq_len} exceeds max_position_embeddings {max_pos}. "
        "Auto-truncation should have trimmed the text."
    )
