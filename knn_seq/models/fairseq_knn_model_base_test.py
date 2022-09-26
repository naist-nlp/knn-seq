from typing import List

import pytest
import torch

from . import fairseq_knn_model_base

from data.fixtures import init_models

class TestFairseqKNNModelBase:
    @pytest.fixture
    def init_knn_model_base(self, init_models) -> None:
        self.ensemble, self.saved_args = init_models
        self.knn_base = fairseq_knn_model_base.FairseqKNNModelBase(self.ensemble)

    def test_get_embed_dim(self, init_knn_model_base) -> None:

        embed_dims = self.knn_base.get_embed_dim()
        test_case: List[int] = [
            model.decoder.embed_tokens.embedding_dim for model in self.ensemble
        ]
        assert embed_dims == test_case

    def test_extract_sentence_features_from_encoder_outs(
        self, init_knn_model_base
    ) -> None:

        # Optionality test
        encoder_outs = None
        assert (
            self.knn_base.extract_sentence_features_from_encoder_outs(encoder_outs)
            is None
        )

        # Encoder out shape test
        src_tokens = torch.LongTensor([[4, 5, 6]])
        src_lengths = torch.Tensor([1])

        encoder_outs = [
            model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            for model in self.ensemble
        ]
        sentence_features = self.knn_base.extract_sentence_features_from_encoder_outs(
            encoder_outs
        )
        feature_sizes = [list(feature.size()) for feature in sentence_features]

        test_case: List[List[int]] = [
            [src_tokens.size()[0], model.encoder.embed_tokens.embedding_dim]
            for model in self.ensemble
        ]
        assert feature_sizes == test_case
