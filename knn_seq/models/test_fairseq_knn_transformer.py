import pytest
import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.models.transformer import TransformerDecoderBase, TransformerEncoderBase

from data.fixtures import (
    testdata_langpair_dataset,
    testdata_models,
    testdata_src_dict,
    testdata_tgt_dict,
)
from knn_seq.models.fairseq_knn_transformer import KNNTransformer


class TestKNNTransformer:
    @pytest.fixture
    def testdata_collater(self, testdata_langpair_dataset):
        self.num_samples = 5

        dataset = BaseWrapperDataset(testdata_langpair_dataset)
        return dataset.dataset.collater([dataset[i] for i in range(self.num_samples)])

    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    def test_init(self, key, testdata_models):
        models, _ = testdata_models
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
        testdata_models,
        testdata_langpair_dataset,
        testdata_collater,
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        src_tokens = testdata_collater["net_input"]["src_tokens"]
        src_lengths = testdata_collater["net_input"]["src_lengths"]
        prev_output_tokens = testdata_collater["net_input"]["prev_output_tokens"]

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
                self.num_samples,  # Batch size
                testdata_collater["target"].shape[1],  # tgt length
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

    @pytest.mark.parametrize("alignment_heads", [None, 1, 2])
    @pytest.mark.parametrize("alignment_layer", [None, 0, 1])
    @pytest.mark.parametrize("features_only", [True, False])
    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    def test_forward_decoder(
        self,
        key,
        features_only,
        alignment_layer,
        alignment_heads,
        testdata_models,
        testdata_langpair_dataset,
        testdata_collater,
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        src_tokens = testdata_collater["net_input"]["src_tokens"]
        src_lengths = testdata_collater["net_input"]["src_lengths"]
        prev_output_tokens = testdata_collater["net_input"]["prev_output_tokens"]

        encoder_out = knnmodel.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )

        decoder_out, model_specific_out = knnmodel.forward_decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        expected_dim = len(testdata_langpair_dataset.tgt_dict)
        if features_only:
            expected_dim = knnmodel.decoder.embed_tokens.embedding_dim

        assert decoder_out.shape == torch.Size(
            [
                self.num_samples,  # Batch size
                testdata_collater["target"].shape[1],  # tgt length
                expected_dim,
            ]
        )

        (
            expected_decoder_out,
            expected_model_specific_out,
        ) = model.decoder.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            expected_decoder_out = model.decoder.output_layer(expected_decoder_out)

        assert torch.equal(decoder_out, expected_decoder_out)

        assert torch.equal(
            model_specific_out["attn"][0], expected_model_specific_out["attn"][0]
        )

        for state, expected_state in zip(
            model_specific_out["inner_states"],
            expected_model_specific_out["inner_states"],
        ):
            assert torch.equal(state, expected_state)

        # Figuring out features from KNNTransformer
        _, expected_model_specific_out = knnmodel.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        for feature, expected_feature in zip(
            model_specific_out["features"], expected_model_specific_out["features"]
        ):
            if feature is None:
                assert feature == expected_feature
            else:
                assert torch.equal(feature, expected_feature)

    @pytest.mark.parametrize("full_context_alignment", [True, False])
    @pytest.mark.parametrize("alignment_heads", [None, 1, 2])
    @pytest.mark.parametrize("alignment_layer", [None, 0, 1])
    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    def test_extract_features_scriptable(
        self,
        key,
        alignment_layer,
        alignment_heads,
        full_context_alignment,
        testdata_models,
        testdata_collater,
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        src_tokens = testdata_collater["net_input"]["src_tokens"]
        src_lengths = testdata_collater["net_input"]["src_lengths"]
        prev_output_tokens = testdata_collater["net_input"]["prev_output_tokens"]

        encoder_out = knnmodel.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )

        decoder_out, model_specific_out = knnmodel.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        assert decoder_out.shape == torch.Size(
            [
                self.num_samples,  # Batch size
                testdata_collater["target"].shape[1],  # tgt length
                knnmodel.decoder.embed_tokens.embedding_dim,
            ]
        )

        # Figuring out features from KNNTransformer
        is_alignment_layer0 = alignment_layer == 0
        is_alignment_layer1 = alignment_layer == 1 or alignment_layer == None

        embedded_positions = model.decoder.embed_positions(prev_output_tokens)
        embedded_tokens = model.decoder.embed_scale * model.decoder.embed_tokens(
            prev_output_tokens
        )

        x = embedded_positions + embedded_tokens
        x = x.transpose(0, 1)
        assert torch.equal(model_specific_out["inner_states"][0], x)
        attn_mask = (
            model.decoder.buffered_future_mask(x)
            if not full_context_alignment
            else None
        )
        self_attn_padding_mask = prev_output_tokens.eq(model.decoder.padding_idx)
        x, layer_attn, _ = model.decoder.layers[0](
            x,
            encoder_out["encoder_out"][0],
            encoder_out["encoder_padding_mask"][0],
            None,
            self_attn_mask=attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=is_alignment_layer0,
            need_head_weights=is_alignment_layer0,
        )
        assert torch.equal(model_specific_out["inner_states"][1], x)
        if is_alignment_layer0:
            attn = layer_attn.float()
            if alignment_heads is not None:
                attn = attn[:alignment_heads]
            torch.equal(model_specific_out["attn"][0], attn.mean(dim=0))

        attn_mask = (
            model.decoder.buffered_future_mask(x)
            if not full_context_alignment
            else None
        )
        x, _, _, expected_features = knnmodel.forward_decoder_last_layer(
            x,
            encoder_out["encoder_out"][0],
            encoder_out["encoder_padding_mask"][0],
            None,
            self_attn_mask=attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=is_alignment_layer1,
            need_head_weights=is_alignment_layer1,
        )
        assert torch.equal(model_specific_out["inner_states"][2], x)
        if is_alignment_layer1:
            attn = layer_attn.float()
            if alignment_heads is not None:
                attn = attn[:alignment_heads]
            torch.equal(model_specific_out["attn"][0], attn.mean(dim=0))

        if key == "ffn_out":
            assert expected_features == None
        else:
            assert torch.equal(model_specific_out["features"][0], expected_features)

        expected_decoder_out = x.transpose(0, 1)
        assert torch.equal(decoder_out, expected_decoder_out)
