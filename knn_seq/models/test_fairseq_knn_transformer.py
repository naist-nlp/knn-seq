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
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        num_samples = 5
        dataset = BaseWrapperDataset(testdata_langpair_dataset)
        collater = dataset.dataset.collater([dataset[i] for i in range(num_samples)])

        src_tokens = collater["net_input"]["src_tokens"]
        src_lengths = collater["net_input"]["src_lengths"]
        prev_output_tokens = collater["net_input"]["prev_output_tokens"]

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
                collater["target"].shape[1],  # tgt length
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
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        num_samples = 5
        dataset = BaseWrapperDataset(testdata_langpair_dataset)
        collater = dataset.dataset.collater([dataset[i] for i in range(num_samples)])

        src_tokens = collater["net_input"]["src_tokens"]
        src_lengths = collater["net_input"]["src_lengths"]
        prev_output_tokens = collater["net_input"]["prev_output_tokens"]

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
                num_samples,  # Batch size
                collater["target"].shape[1],  # tgt length
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
        testdata_langpair_dataset,
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        num_samples = 5
        dataset = BaseWrapperDataset(testdata_langpair_dataset)
        collater = dataset.dataset.collater([dataset[i] for i in range(num_samples)])

        src_tokens = collater["net_input"]["src_tokens"]
        src_lengths = collater["net_input"]["src_lengths"]
        prev_output_tokens = collater["net_input"]["prev_output_tokens"]

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
                num_samples,  # Batch size
                collater["target"].shape[1],  # tgt length
                knnmodel.decoder.embed_tokens.embedding_dim,
            ]
        )

        # Figuring out features from KNNTransformer
        actual_alignment_layer = (
            model.decoder.num_layers - 1 if alignment_layer == None else alignment_layer
        )

        is_alignment_layer0 = actual_alignment_layer == 0
        is_alignment_layer1 = actual_alignment_layer == 1

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
            assert torch.equal(model_specific_out["attn"][0], attn.mean(dim=0))

        attn_mask = (
            model.decoder.buffered_future_mask(x)
            if not full_context_alignment
            else None
        )
        x, layer_attn, _, expected_features = knnmodel.forward_decoder_last_layer(
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
            assert torch.equal(model_specific_out["attn"][0], attn.mean(dim=0))

        if key == "ffn_out":
            assert expected_features == None
        else:
            assert torch.equal(model_specific_out["features"][0], expected_features)

        expected_decoder_out = x.transpose(0, 1)
        assert torch.equal(decoder_out, expected_decoder_out)

    @pytest.mark.parametrize("full_context_alignment", [True, False])
    @pytest.mark.parametrize("alignment_heads", [None, 1, 2])
    @pytest.mark.parametrize("alignment_layer", [None, 0, 1])
    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    def test_forward_decoder_last_layer(
        self,
        key,
        alignment_layer,
        alignment_heads,
        full_context_alignment,
        testdata_models,
        testdata_langpair_dataset,
    ):
        models, _ = testdata_models
        model = models[0]
        model.eval()
        knnmodel = KNNTransformer(model, key=key)

        num_samples = 5
        dataset = BaseWrapperDataset(testdata_langpair_dataset)
        collater = dataset.dataset.collater([dataset[i] for i in range(num_samples)])

        src_tokens = collater["net_input"]["src_tokens"]
        src_lengths = collater["net_input"]["src_lengths"]
        prev_output_tokens = collater["net_input"]["prev_output_tokens"]

        encoder_out = model.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )

        _, model_specific_out = model.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            features_only=True,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        index_last_layer = model.decoder.num_layers - 1
        actual_alignment_layer = (
            index_last_layer if alignment_layer == None else alignment_layer
        )
        is_alignment_layer = actual_alignment_layer == index_last_layer

        input_to_last_layer = model_specific_out["inner_states"][index_last_layer]
        self_attn_mask = (
            model.decoder.buffered_future_mask(input_to_last_layer)
            if not full_context_alignment
            else None
        )
        self_attn_padding_mask = prev_output_tokens.eq(model.decoder.padding_idx)

        result, attn, attn_state, features = knnmodel.forward_decoder_last_layer(
            input_to_last_layer,
            encoder_out["encoder_out"][0],
            encoder_out["encoder_padding_mask"][0],
            None,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=is_alignment_layer,
            need_head_weights=is_alignment_layer,
        )

        expected_result, expected_attn, expected_attn_state = model.decoder.layers[-1](
            input_to_last_layer,
            encoder_out["encoder_out"][0],
            encoder_out["encoder_padding_mask"][0],
            None,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=is_alignment_layer,
            need_head_weights=is_alignment_layer,
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(attn, expected_attn)
        if attn_state is not None:
            assert torch.equal(attn_state, expected_attn_state)
        else:
            assert expected_attn_state is None

        # Calculate Features!
        if key == "ffn_out":
            if model.decoder.layer_norm is not None:
                expected_features = model.decoder.layers[-1].layer_norm(result)
            else:
                expected_features = result

            # T x B x C -> B x T x C
            expected_features = expected_features.transpose(0, 1)
        else:
            x = self_attention(
                model.decoder.layers[-1],
                input_to_last_layer,
                encoder_out["encoder_out"][0],
                self_attn_mask,
                self_attn_padding_mask,
            )
            x = encoder_decoder_attention(
                model.decoder.layers[-1],
                x,
                is_alignment_layer,
                encoder_out["encoder_out"][0],
                encoder_out["encoder_padding_mask"][0],
            )

            if model.decoder.layers[-1].normalize_before:
                x = model.decoder.layers[-1].final_layer_norm(x)

            # T x B x C -> B x T x C
            expected_features = x.transpose(0, 1)

        if features != None:
            assert torch.equal(features, expected_features)


def self_attention(last_layer, x, encoder_out, self_attn_mask, self_attn_padding_mask):
    residual = x
    if last_layer.normalize_before:
        x = last_layer.self_attn_layer_norm(x)

    if last_layer.cross_self_attention:
        if self_attn_mask is not None:
            self_attn_mask = torch.cat(
                (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
            )
        if self_attn_padding_mask is not None:
            if encoder_padding_mask is None:
                encoder_padding_mask = self_attn_padding_mask.new_zeros(
                    encoder_out.size(1), encoder_out.size(0)
                )
            self_attn_padding_mask = torch.cat(
                (encoder_padding_mask, self_attn_padding_mask), dim=1
            )
        y = torch.cat((encoder_out, x), dim=0)
    else:
        y = x

    x, _ = last_layer.self_attn(
        query=x,
        key=y,
        value=y,
        key_padding_mask=self_attn_padding_mask,
        need_weights=False,
        attn_mask=self_attn_mask,
    )
    if last_layer.c_attn is not None:
        tgt_len, bsz = x.size(0), x.size(1)
        x = x.view(
            tgt_len,
            bsz,
            last_layer.nh,
            last_layer.head_dim,
        )
        x = torch.einsum("tbhd,h->tbhd", x, last_layer.c_attn)
        x = x.reshape(tgt_len, bsz, last_layer.embed_dim)
    if last_layer.attn_ln is not None:
        x = last_layer.attn_ln(x)
    x = last_layer.dropout_module(x)
    x = last_layer.residual_connection(x, residual)
    if not last_layer.normalize_before:
        x = last_layer.self_attn_layer_norm(x)

    return x


def encoder_decoder_attention(
    last_layer, x, need_attn, encoder_out, encoder_padding_mask
):
    if last_layer.encoder_attn is None:
        return x, None

    residual = x
    if last_layer.normalize_before:
        x = last_layer.encoder_attn_layer_norm(x)

    need_weights = need_attn or (not last_layer.training and last_layer.need_attn)

    x, _ = last_layer.encoder_attn(
        query=x,
        key=encoder_out,
        value=encoder_out,
        key_padding_mask=encoder_padding_mask,
        static_kv=True,
        need_weights=need_weights,
        need_head_weights=need_attn,
    )
    x = last_layer.dropout_module(x)
    x = last_layer.residual_connection(x, residual)
    if not last_layer.normalize_before:
        x = last_layer.encoder_attn_layer_norm(x)

    return x
