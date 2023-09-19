import copy
from typing import Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from fairseq.sequence_generator import EnsembleModel
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

    def search(self, queries: torch.Tensor, index_id: int = 0):
        batch_size = queries.size(0)

        scores = torch.tensor(
            [
                [
                    torch.sigmoid(queries[i][index_id]),
                    1 - torch.sigmoid(queries[i][index_id]),
                ]
                for i in range(batch_size)
            ]
        )
        probs = torch.tensor(
            [
                [
                    torch.sigmoid(queries[i][index_id]),
                    1 - torch.sigmoid(queries[i][index_id]),
                ]
                for i in range(batch_size)
            ]
        )
        indices = torch.LongTensor(
            [[i + index_id + 4, i + index_id + 5] for i in range(batch_size)]
        )
        return FairseqKNNModelBase.KNNOutput(scores, probs, indices)


@pytest.fixture(scope="module")
def generate_test_data(testdata_langpair_dataset):
    dataset = testdata_langpair_dataset
    return dataset.collater([dataset[i] for i in range(2)])


class BeamableModel(nn.Module):
    def set_beam_size(self, beam_size):
        self.beam_size = beam_size


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

    @pytest.mark.parametrize("knn_threshold", [0.5, None])
    @pytest.mark.parametrize("knn_weight", [0.0, 0.5, 1.0])
    def test_add_knn_probs(self, knn_weight, knn_threshold, testdata_models) -> None:
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        batch_size = 1
        vocab_size = 128
        embed_dim = 16

        test_lprobs = torch.rand(batch_size, vocab_size)
        test_queries = torch.rand(batch_size, embed_dim)

        knn_model_base.knn_weight = knn_weight
        expected_knn_output = knn_model_base.search(test_queries)
        knn_probs = expected_knn_output.probs

        knn_vocab_probs = torch.zeros(*test_lprobs.size())
        knn_vocab_probs[:, expected_knn_output.indices[0, :]] += knn_probs
        knn_vocab_probs *= knn_model_base.knn_weight
        knn_vocab_probs[:, knn_model_base.pad] = 0.0

        knn_model_base.knn_threshold = knn_threshold
        probs = torch.exp(test_lprobs)
        if knn_model_base.knn_threshold is not None:
            max_scores = torch.max(expected_knn_output.scores, dim=1).values
            update_batch_indices = max_scores.gt(knn_model_base.knn_threshold)
            probs[update_batch_indices] = (1.0 - knn_model_base.knn_weight) * probs[
                update_batch_indices
            ] + knn_vocab_probs[update_batch_indices]
        else:
            probs = (1.0 - knn_model_base.knn_weight) * probs + knn_vocab_probs
        expected_lprobs = torch.log(probs)

        lprobs, knn_output = knn_model_base.add_knn_probs(test_lprobs, test_queries)

        assert knn_model_base.knn_timer.start_time is not None
        assert knn_model_base.knn_timer.stop_time is not None

        assert torch.allclose(lprobs, expected_lprobs)
        assert torch.allclose(knn_output.scores, expected_knn_output.scores)
        assert torch.allclose(knn_output.probs, expected_knn_output.probs)
        assert torch.allclose(knn_output.indices, expected_knn_output.indices)

    @pytest.mark.parametrize("is_knn_ensemble", [True, False])
    @pytest.mark.parametrize("has_incremental", [True, False])
    @pytest.mark.parametrize("has_multiple_models", [True, False])
    def test_forward_decoder_with_knn(
        self,
        generate_test_data,
        testdata_models,
        has_multiple_models,
        has_incremental,
        is_knn_ensemble,
    ) -> None:
        ensemble, _ = testdata_models
        if has_multiple_models:
            alt_model = copy.deepcopy(ensemble[0])
            alt_params = alt_model.state_dict()
            for param, value in alt_params.items():
                alt_params[param] = torch.rand_like(value)

            alt_model.load_state_dict(alt_params)
            ensemble.append(alt_model)

        knn_model_base = FairseqKNNMockModel(ensemble)
        net_inputs = {
            "src_tokens": generate_test_data["net_input"]["src_tokens"],
            "src_lengths": generate_test_data["net_input"]["src_lengths"],
        }
        tokens = generate_test_data["net_input"]["prev_output_tokens"]

        encoder_outs = [model.encoder(**net_inputs) for model in ensemble]
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(knn_model_base.models_size)
            ],
        )

        knn_model_base.knn_ensemble = is_knn_ensemble
        knn_model_base.has_incremental = has_incremental
        lprobs, attn = knn_model_base.forward_decoder(
            tokens, encoder_outs, incremental_states
        )

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

            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if knn_model_base.knn_ensemble or i == 0:
                    knn_queries = decoder_out[1]["features"][0][:, -1, :]

            if knn_model_base.knn_ensemble:
                lprobs, knn_output = knn_model_base.add_knn_probs(
                    lprobs, knn_queries, index_id=i
                )
                expected_lprobs = lprobs
                expected_attn = attn
                expected_knn_output = knn_output

        if not knn_model_base.knn_ensemble:
            lprobs, knn_output = knn_model_base.add_knn_probs(lprobs, knn_queries)
            expected_lprobs = lprobs
            expected_attn = attn
            expected_knn_output = knn_output

        lprobs, attn, knn_output = knn_model_base.forward_decoder_with_knn(
            tokens, encoder_outs, incremental_states
        )

        assert torch.allclose(lprobs, expected_lprobs)
        assert torch.allclose(attn, expected_attn)
        assert torch.allclose(knn_output.scores, expected_knn_output.scores)
        assert torch.allclose(knn_output.probs, expected_knn_output.probs)
        assert torch.equal(knn_output.indices, expected_knn_output.indices)

    def test_forward_decoder(self, generate_test_data, testdata_models):
        ensemble, _ = testdata_models
        knn_model_base = FairseqKNNMockModel(ensemble)

        net_inputs = {
            "src_tokens": generate_test_data["net_input"]["src_tokens"],
            "src_lengths": generate_test_data["net_input"]["src_lengths"],
        }

        tokens = generate_test_data["net_input"]["prev_output_tokens"]
        encoder_outs = [model.encoder(**net_inputs) for model in ensemble]
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(knn_model_base.models_size)
            ],
        )

        expected_lprobs, expected_attn, _ = knn_model_base.forward_decoder_with_knn(
            tokens, encoder_outs, incremental_states
        )
        lprobs, attn = knn_model_base.forward_decoder(
            tokens, encoder_outs, incremental_states
        )
        assert lprobs.shape == expected_lprobs.shape
        assert attn.shape == expected_attn.shape
