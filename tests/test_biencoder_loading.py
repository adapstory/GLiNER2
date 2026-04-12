"""Tests for bi-encoder support in Extractor model.

Verifies that ExtractorConfig supports use_bi_encoder flag and that
Extractor conditionally creates bi_classifier, schema_proj, text_proj layers.
"""

import pytest
import torch

from gliner2.model import ExtractorConfig, Extractor


class TestExtractorConfigBiEncoderFlag:
    """ExtractorConfig must expose a use_bi_encoder boolean field."""

    def test_extractor_config_has_bi_encoder_flag(self):
        config = ExtractorConfig(model_name="bert-base-uncased", use_bi_encoder=True)
        assert config.use_bi_encoder is True

    def test_extractor_config_default_is_uniencoder(self):
        config = ExtractorConfig(model_name="bert-base-uncased")
        assert config.use_bi_encoder is False


class TestBiEncoderExtractorLayers:
    """Extractor must conditionally create bi-encoder layers."""

    @pytest.fixture()
    def biencoder_model(self):
        config = ExtractorConfig(model_name="bert-base-uncased", use_bi_encoder=True)
        model = Extractor(config)
        return model

    @pytest.fixture()
    def uniencoder_model(self):
        config = ExtractorConfig(model_name="bert-base-uncased", use_bi_encoder=False)
        model = Extractor(config)
        return model

    def test_biencoder_extractor_has_extra_layers(self, biencoder_model):
        assert hasattr(biencoder_model, "bi_classifier")
        assert hasattr(biencoder_model, "schema_proj")
        assert hasattr(biencoder_model, "text_proj")

    def test_uniencoder_extractor_lacks_biencoder_layers(self, uniencoder_model):
        assert not hasattr(uniencoder_model, "bi_classifier")
        assert not hasattr(uniencoder_model, "schema_proj")
        assert not hasattr(uniencoder_model, "text_proj")

    def test_biencoder_state_dict_keys(self, biencoder_model):
        state_dict = biencoder_model.state_dict()
        keys = list(state_dict.keys())
        assert any(k.startswith("bi_classifier") for k in keys), (
            f"No bi_classifier keys found in state_dict. Keys: {keys}"
        )
        assert any(k.startswith("schema_proj") for k in keys), (
            f"No schema_proj keys found in state_dict. Keys: {keys}"
        )
        assert any(k.startswith("text_proj") for k in keys), (
            f"No text_proj keys found in state_dict. Keys: {keys}"
        )
