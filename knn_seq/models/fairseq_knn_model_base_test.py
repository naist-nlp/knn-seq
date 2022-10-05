from typing import List

import pytest
import torch

from data.fixtures import testdata_models
from knn_seq.models.fairseq_knn_model_base import FairseqKNNModelBase


class TestFairseqKNNModelBase:
    @pytest.fixture
    def init_knn_model_base(self, testdata_models) -> None:
        self.ensemble, self.saved_args = testdata_models
        self.knn_base = FairseqKNNModelBase(self.ensemble)

    def test_get_embed_dim(self, init_knn_model_base) -> None:
        embed_dims = self.knn_base.get_embed_dim()
        expected_embed_dims = [
            model.decoder.embed_tokens.embedding_dim for model in self.ensemble
        ]
        assert embed_dims == expected_embed_dims

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
        src_lengths = torch.Tensor([3])

        encoder_outs = [
            model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
            for model in self.ensemble
        ]
        sentence_features = self.knn_base.extract_sentence_features_from_encoder_outs(
            encoder_outs
        )
        feature_sizes = [list(feature.size()) for feature in sentence_features]

        expected_feature_sizes: List[List[int]] = [
            [src_tokens.size()[0], model.encoder.embed_tokens.embedding_dim]
            for model in self.ensemble
        ]
        assert feature_sizes == expected_feature_sizes

    def test_extract_sentence_features(self, init_knn_model_base):
        net_inputs = {
            "src_tokens": torch.LongTensor([[4, 5, 6]]),
            "src_lengths": torch.Tensor([3]),
        }
        encoder_features = self.knn_base.extract_sentence_features(net_inputs)
        feature_sizes = [feature.size() for feature in encoder_features]

        src_num, *_ = net_inputs["src_tokens"].size()
        embed_dims = [
            model.encoder.embed_tokens.embedding_dim for model in self.ensemble
        ]
        expected_feature_sizes = [
            torch.zeros(src_num, embed_dim).size() for embed_dim in embed_dims
        ]
        assert feature_sizes == expected_feature_sizes
