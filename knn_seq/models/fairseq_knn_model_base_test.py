import math
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from data.fixtures import (  # pylint: disable=unused-import
    testdata_langpair_dataset,
    testdata_models,
    testdata_src_dict,
    testdata_tgt_dict,
)
from knn_seq import utils
from knn_seq.models.fairseq_knn_model_base import FairseqKNNModelBase
from knn_seq.models.fairseq_knn_transformer import KNNTransformer


class FairseqKNNMockModel(FairseqKNNModelBase):
    def set_index(self):
        pass

    def search(self, querys: torch.Tensor, index_id: int = 0):
        pass


@pytest.fixture(scope="module")
def generate_test_data(testdata_langpair_dataset):
    dataset = testdata_langpair_dataset
    return dataset.collater([dataset[i] for i in range(2)])


class BeamableModel(nn.Module):
    def set_beam_size(self, beam_size):
        self.beam_size = beam_size


def search(self, queries, index_id=0):
    batch_size = 2
    k = 2
    scores = torch.rand(batch_size, k)
    probs = torch.rand(batch_size, k)
    indices = torch.zeros(batch_size, k, dtype=torch.long)
    return FairseqKNNModelBase.KNNOutput(scores, probs, indices)


class TestFairseqKNNModelBase:
    def test__init__(self, testdata_models) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        dictionary = knn_model_base.single_model.decoder.dictionary
        assert knn_model_base.tgt_dict == dictionary
        assert knn_model_base.pad == dictionary.pad()

        for m in knn_model_base.wrapped_models:
            assert isinstance(m, KNNTransformer)

        assert knn_model_base.knn_weight == 0.0
        assert knn_model_base.knn_threshold == None
        assert knn_model_base.knn_ensemble == False

        assert isinstance(knn_model_base.knn_timer, utils.StopwatchMeter)

    def test_init_model(self, testdata_models) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        for p in knn_model_base.parameters():
            if getattr(p, "requires_grad", None) is not None:
                assert p.requires_grad == False
        assert knn_model_base.training == False

    def test_get_embed_dim(self, testdata_models) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        embed_dims = knn_model_base.get_embed_dim()
        expected_embed_dims = [
            model.decoder.embed_tokens.embedding_dim for model in ensemble
        ]
        assert embed_dims == expected_embed_dims

    def test_set_src_sents(self, testdata_models) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        src_sents = ["test1", "test2"]
        knn_model_base.set_src_sents(src_sents)
        assert knn_model_base.src_sents == src_sents

    def test_set_decoder_beam_size(self, testdata_models) -> None:
        ensemble, _ = testdata_models

        beamable_model = BeamableModel()
        ensemble_with_beamable_model = ensemble + [beamable_model]

        beam_size = 4
        knn_model_base = FairseqKNNMockModel(ensemble_with_beamable_model)
        knn_model_base.set_decoder_beam_size(beam_size)
        assert beamable_model.beam_size == beam_size
        assert knn_model_base.beam_size == beam_size

    @pytest.mark.parametrize("output_encoder_features", [True, False])
    def test_forward(
        self, testdata_models, generate_test_data, output_encoder_features
    ) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        if output_encoder_features:
            net_inputs = {
                "src_tokens": generate_test_data["net_input"]["src_tokens"],
                "src_lengths": generate_test_data["net_input"]["src_lengths"],
            }
            features = knn_model_base(
                **net_inputs, output_encoder_features=output_encoder_features
            )
            feature_sizes = [feature.size() for feature in features]

            expected_features = knn_model_base.extract_sentence_features(net_inputs)
            expected_sizes = [feature.size() for feature in expected_features]
            assert feature_sizes == expected_sizes
        else:
            net_inputs = {
                "src_tokens": generate_test_data["net_input"]["src_tokens"],
                "src_lengths": generate_test_data["net_input"]["src_lengths"],
                "prev_output_tokens": generate_test_data["net_input"][
                    "prev_output_tokens"
                ],
            }
            features = knn_model_base(**net_inputs)
            feature_sizes = [feature.size() for feature in features]

            decoder_outputs = [
                model(**net_inputs, features_only=True)
                for model in knn_model_base.wrapped_models
            ]
            expected_features = [
                decoder_out[1]["features"][0] for decoder_out in decoder_outputs
            ]
            expected_sizes = [feature.size() for feature in expected_features]
            assert feature_sizes == expected_sizes

    def test_extract_sentence_features_from_encoder_outs(
        self, testdata_models, generate_test_data
    ) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

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
        generate_test_data,
    ) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

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

    @pytest.mark.parametrize("knn_threshold", [torch.zeros(2), None])
    def test_add_knn_probs(self, knn_threshold, testdata_models, monkeypatch) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        batch_size = 2
        vocab_size = 20
        embed_dim = 16
        test_lprobs = torch.rand(batch_size, vocab_size)
        test_queries = torch.rand(batch_size, embed_dim)

        monkeypatch.setattr(FairseqKNNMockModel, "search", search)
        expected_knn_output = knn_model_base.search(test_queries)
        knn_probs = expected_knn_output.probs

        knn_vocab_probs = knn_probs.new_zeros(*test_lprobs.size())
        knn_vocab_probs = knn_vocab_probs.scatter_add_(
            dim=-1, index=expected_knn_output.indices, src=knn_probs
        )
        knn_vocab_probs *= knn_model_base.knn_weight
        knn_vocab_probs[:, knn_model_base.pad] = 0.0

        probs = torch.exp(test_lprobs)

        knn_model_base.knn_threshold = knn_threshold
        lprobs, knn_output = knn_model_base.add_knn_probs(test_lprobs, test_queries)

        assert knn_model_base.knn_timer.start_time is not None
        assert knn_model_base.knn_timer.stop_time is not None

        if knn_model_base.knn_threshold is not None:
            max_scores = torch.max(expected_knn_output.scores, dim=1).values
            update_batch_indices = max_scores.gt(knn_model_base.knn_threshold)
            probs[update_batch_indices] = (1.0 - knn_model_base.knn_weight) * probs[
                update_batch_indices
            ] + knn_vocab_probs[update_batch_indices]
        else:
            probs = (1.0 - knn_model_base.knn_weight) * probs + knn_vocab_probs
        expected_lprobs = torch.log(probs)

        assert lprobs.shape == expected_lprobs.shape
        assert knn_output.scores.shape == expected_knn_output.scores.shape
        assert knn_output.probs.shape == expected_knn_output.probs.shape
        assert knn_output.indices.shape == expected_knn_output.indices.shape

    @pytest.mark.parametrize("is_knn_ensemble", [True, False])
    @pytest.mark.parametrize("has_incremental", [True, False])
    @pytest.mark.parametrize("return_retrieved", [True, False])
    @pytest.mark.parametrize("has_multiple_models", [True, False])
    def test_forward_decoder(
        self,
        generate_test_data,
        testdata_models,
        has_multiple_models,
        return_retrieved,
        has_incremental,
        is_knn_ensemble,
        monkeypatch,
    ) -> None:
        ensemble, _ = testdata_models

        if has_multiple_models:
            ensemble = ensemble * 2
        knn_model_base = FairseqKNNModelBase(ensemble)

        net_inputs = {
            "src_tokens": generate_test_data["net_input"]["src_tokens"],
            "src_lengths": generate_test_data["net_input"]["src_lengths"],
        }

        encoder_outs = [model.encoder(**net_inputs) for model in ensemble]
        tokens = generate_test_data["net_input"]["prev_output_tokens"]

        log_probs = []
        avg_attn = None
        encoder_out = None
        knn_queries = None
        temperature = 1.0
        attn = None

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(knn_model_base.models_size)
            ],
        )
        monkeypatch.setattr(FairseqKNNModelBase, "search", search)

        knn_model_base.has_incremental = has_incremental
        for i, model in enumerate(knn_model_base.wrapped_models):
            if knn_model_base.has_encoder():
                encoder_out = encoder_outs[i]
            if knn_model_base.has_incremental_states():
                decoder_out = model.forward_decoder(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                decoder_out = model.forward_decoder(tokens, encoder_out=encoder_out)

            knn_model_base.knn_ensemble = is_knn_ensemble
            attn = decoder_out[1]["attn"][0]
            if knn_model_base.knn_ensemble or i == 0:
                knn_queries = decoder_out[1]["features"][0][:, -1, :]
            if attn is not None:
                attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                decoder_out[1],
            )
            lprobs: Tensor = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            lprobs = lprobs[:, -1, :]

            if knn_queries is not None and knn_model_base.knn_ensemble:
                lprobs, knn_output = knn_model_base.add_knn_probs(
                    lprobs, knn_queries, index_id=i
                )

            if knn_model_base.models_size == 1:
                break

            log_probs.append(lprobs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        if knn_model_base.models_size > 1:
            avg_lprobs = torch.logsumexp(
                torch.stack(log_probs, dim=0), dim=0
            ) - math.log(knn_model_base.models_size)

            if avg_attn is not None:
                avg_attn.div_(knn_model_base.models_size)

            lprobs = avg_lprobs
            attn = avg_attn

        if knn_queries is not None and not knn_model_base.knn_ensemble:
            lprobs, knn_output = knn_model_base.add_knn_probs(lprobs, knn_queries)

        results = knn_model_base.forward_decoder(
            tokens, encoder_outs, incremental_states, return_retrieved=return_retrieved
        )
        if return_retrieved:
            assert len(results) == 4
            assert results[0].shape == lprobs.shape
            assert results[1].shape == attn.shape
            assert results[2].shape == knn_output.scores.shape
            assert results[3].shape == knn_output.indices.shape
        else:
            assert len(results) == 2
            assert results[0].shape == lprobs.shape
            assert results[1].shape == attn.shape
