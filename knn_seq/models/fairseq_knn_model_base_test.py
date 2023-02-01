from typing import List

import pytest
import torch

from data.fixtures import testdata_models  # pylint: disable=unused-import
from knn_seq.models.fairseq_knn_model_base import FairseqKNNModelBase


@pytest.fixture(scope="module")
def knn_model_base(testdata_models):
    ensemble, _ = testdata_models
    return FairseqKNNModelBase(ensemble)


class TestFairseqKNNModelBase:
    def test_get_embed_dim(self, testdata_models, knn_model_base) -> None:
        ensemble, _ = testdata_models
        embed_dims = knn_model_base.get_embed_dim()
        expected_embed_dims = [
            model.decoder.embed_tokens.embedding_dim for model in ensemble
        ]
        assert embed_dims == expected_embed_dims

    def test_extract_sentence_features_from_encoder_outs(
        self, testdata_models, knn_model_base
    ) -> None:
        ensemble, _ = testdata_models
        net_inputs = {
            "src_tokens": torch.LongTensor([[4, 5, 6], [7, 8, 0]]),
            "src_lengths": torch.Tensor([3, 2]),
        }

        # Optionality test
        encoder_outs = None
        assert (
            knn_model_base.extract_sentence_features_from_encoder_outs(encoder_outs)
            is None
        )

        # Encoder out shape test
        encoder_outs = [
            model.encoder(
                src_tokens=net_inputs["src_tokens"],
                src_lengths=net_inputs["src_lengths"],
            )
            for model in ensemble
        ]
        sentence_features = knn_model_base.extract_sentence_features_from_encoder_outs(
            encoder_outs
        )
        feature_sizes = [list(feature.size()) for feature in sentence_features]

        expected_feature_sizes: List[List[int]] = [
            [
                net_inputs["src_tokens"].size()[0],
                model.encoder.embed_tokens.embedding_dim,
            ]
            for model in ensemble
        ]
        assert feature_sizes == expected_feature_sizes

    def test_extract_sentence_features(self, testdata_models, knn_model_base) -> None:
        ensemble, _ = testdata_models
        net_inputs = {
            "src_tokens": torch.LongTensor([[4, 5, 6], [7, 8, 0]]),
            "src_lengths": torch.Tensor([3, 2]),
        }

        encoder_features = knn_model_base.extract_sentence_features(net_inputs)
        feature_sizes = [feature.size() for feature in encoder_features]

        src_num, *_ = net_inputs["src_tokens"].size()
        embed_dims = [model.encoder.embed_tokens.embedding_dim for model in ensemble]
        expected_feature_sizes = [
            torch.zeros(src_num, embed_dim).size() for embed_dim in embed_dims
        ]
        assert feature_sizes == expected_feature_sizes
