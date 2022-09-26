import pytest
import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.models.transformer import TransformerDecoderBase, TransformerEncoderBase

from knn_seq.models.fairseq_knn_transformer import KNNTransformer


class TestKNNTransformer:
    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    # init_models comes from /models/conftest.py
    def test_init(self, key, init_models):
        models, _ = init_models
        new_model = KNNTransformer(models[0], key=key)
        assert isinstance(new_model.encoder, TransformerEncoderBase)
        assert isinstance(new_model.decoder, TransformerDecoderBase)
        assert new_model.decoder_last_layer == new_model.decoder.layers[-1]
        if key == "ffn_in":
            assert new_model.use_ffn_input == True
        elif key == "ffn_out":
            assert new_model.use_ffn_input == False

    @pytest.mark.parametrize("alignment_heads", [None, 1, 2])
    @pytest.mark.parametrize("alignment_layer", [None, 0, 1])
    @pytest.mark.parametrize("features_only", [True, False])
    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    def test_forward(
        self,
        key,
        features_only,
        alignment_layer,
        alignment_heads,
        init_models, #from /models/conftest.py
        testdata_langpair_dataset,
    ):
        models, _ = init_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        num_samples = 5

        dataset = BaseWrapperDataset(testdata_langpair_dataset)
        collator = dataset.dataset.collater([dataset[i] for i in range(num_samples)])

        src_tokens = collator["net_input"]["src_tokens"]
        src_lengths = collator["net_input"]["src_lengths"]
        prev_output_tokens = collator["net_input"]["prev_output_tokens"]

        decoder_out, model_specific_out = knnmodel.forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=prev_output_tokens,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        expected_dim = len(testdata_langpair_dataset.tgt_dict)
        if features_only:
            expected_dim = knnmodel.decoder.embed_tokens.embedding_dim

        assert decoder_out.shape == torch.Size(
            [
                num_samples,  # Batch size
                collator["target"].shape[1],  # tgt length
                expected_dim,
            ]
        )

        encoder_out = model.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )

        expected_decoder_out, expected_model_specific_out = model.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        assert torch.equal(decoder_out, expected_decoder_out)

        for attn, expected_attn in zip(
            model_specific_out["attn"],
            expected_model_specific_out["attn"],
        ):
            assert torch.equal(attn, expected_attn)

        for state, expected_state in zip(
            model_specific_out["inner_states"],
            expected_model_specific_out["inner_states"],
        ):
            assert torch.equal(state, expected_state)

        # only KNNTransformer will return features
        _, expected_model_specific_out = knnmodel.forward_decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        for feature, expected_feature in zip(
            model_specific_out["features"], expected_model_specific_out["features"]
        ):
            assert torch.equal(feature, expected_feature)

        assert model_specific_out.keys() == expected_model_specific_out.keys()
