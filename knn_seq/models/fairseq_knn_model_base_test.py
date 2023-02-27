from typing import List

import pytest
import torch

from data.fixtures import (  # pylint: disable=unused-import
    testdata_langpair_dataset,
    testdata_models,
    testdata_src_dict,
    testdata_tgt_dict,
)
from knn_seq import utils
from knn_seq.models.fairseq_knn_model_base import FairseqKNNModelBase
from knn_seq.models.fairseq_knn_transformer import KNNTransformer


@pytest.fixture(scope="module")
def knn_model_base(testdata_models):
    ensemble, _ = testdata_models
    return FairseqKNNModelBase(ensemble)


@pytest.fixture(scope="module")
def generate_test_data(testdata_langpair_dataset):
    dataset = testdata_langpair_dataset
    return dataset.collater([dataset[i] for i in range(2)])


class TestFairseqKNNModelBase:
    def test__init__(self, knn_model_base) -> None:
        dictionary = knn_model_base.single_model.decoder.dictionary
        assert knn_model_base.tgt_dict == dictionary
        assert knn_model_base.pad == knn_model_base.tgt_dict.pad()

        for m in knn_model_base.wrapped_models:
            assert isinstance(m, KNNTransformer)

        assert knn_model_base.knn_weight == 0.0
        assert knn_model_base.knn_threshold == None
        assert knn_model_base.knn_ensemble == False

        assert isinstance(knn_model_base.knn_timer, utils.StopwatchMeter)

    def test_set_index(self, knn_model_base) -> None:
        with pytest.raises(NotImplementedError):
            knn_model_base.set_index()

    def test_init_model(self, knn_model_base) -> None:
        for p in knn_model_base.parameters():
            if getattr(p, "requires_grad", None) is not None:
                assert p.requires_grad == False
        assert knn_model_base.training == False

    def test_get_embed_dim(self, testdata_models, knn_model_base) -> None:
        ensemble, _ = testdata_models
        embed_dims = knn_model_base.get_embed_dim()
        expected_embed_dims = [
            model.decoder.embed_tokens.embedding_dim for model in ensemble
        ]
        assert embed_dims == expected_embed_dims

    def test_extract_sentence_features_from_encoder_outs(
        self, testdata_models, knn_model_base, generate_test_data
    ) -> None:
        ensemble, _ = testdata_models

        # Optionality test
        encoder_outs = None
        assert (
            knn_model_base.extract_sentence_features_from_encoder_outs(encoder_outs)
            is None
        )

        # Encoder out shape test
        net_inputs = {
            "src_tokens": generate_test_data["net_input"]["src_tokens"],
            "src_lengths": generate_test_data["net_input"]["src_lengths"],
        }

        encoder_outs = [model.encoder(**net_inputs) for model in ensemble]
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

    def test_extract_sentence_features(
        self,
        testdata_models,
        knn_model_base,
        generate_test_data,
    ) -> None:
        ensemble, _ = testdata_models
        net_inputs = {
            "src_tokens": generate_test_data["net_input"]["src_tokens"],
            "src_lengths": generate_test_data["net_input"]["src_lengths"],
        }

        encoder_features = knn_model_base.extract_sentence_features(net_inputs)
        feature_sizes = [feature.size() for feature in encoder_features]

        src_num, *_ = net_inputs["src_tokens"].size()
        embed_dims = [model.encoder.embed_tokens.embedding_dim for model in ensemble]
        expected_feature_sizes = [
            torch.zeros(src_num, embed_dim).size() for embed_dim in embed_dims
        ]
        assert feature_sizes == expected_feature_sizes
