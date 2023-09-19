import logging
from typing import Any, Dict, List, Optional

import torch
from fairseq.dataclass import ChoiceEnum
from fairseq.models.fairseq_model import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerDecoderBase,
    TransformerEncoderBase,
    TransformerModelBase,
)
from torch import Tensor

KEY_CHOICES = ChoiceEnum(["ffn_in", "ffn_out"])


logger = logging.getLogger(__name__)


class KNNTransformer(FairseqEncoderDecoderModel):
    def __init__(self, model: TransformerModelBase, key: KEY_CHOICES):
        super().__init__(model.encoder, model.decoder)
        self.encoder: TransformerEncoderBase
        self.decoder: TransformerDecoderBase
        self.decoder_last_layer = model.decoder.layers[-1]
        self.use_ffn_input = key == "ffn_in"
        self._debug = True

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.forward_decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.decoder.output_layer(x)
        return x, extra

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.decoder.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.decoder.embed_positions is not None:
            positions = self.decoder.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.decoder.embed_scale * self.decoder.embed_tokens(prev_output_tokens)

        if self.decoder.quant_noise is not None:
            x = self.decoder.quant_noise(x)

        if self.decoder.project_in_dim is not None:
            x = self.decoder.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.decoder.layernorm_embedding is not None:
            x = self.decoder.layernorm_embedding(x)

        x = self.decoder.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if (
            self.decoder.cross_self_attention
            or prev_output_tokens.eq(self.decoder.padding_idx).any()
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.decoder.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        features: Optional[Tensor] = None
        for idx, layer in enumerate(self.decoder.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.decoder.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if idx == self.decoder.num_layers - 1:
                x, layer_attn, _, features = self.forward_decoder_last_layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            else:
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.decoder.layer_norm is not None:
            x = self.decoder.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.decoder.project_out_dim is not None:
            x = self.decoder.project_out_dim(x)

        return x, {
            "attn": [attn],
            "inner_states": inner_states,
            "features": [features if features is not None else x],
        }

    def forward_decoder_last_layer(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.decoder_last_layer.normalize_before:
            x = self.decoder_last_layer.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.decoder_last_layer.self_attn._set_input_buffer(
                incremental_state, saved_state
            )
        _self_attn_input_buffer = self.decoder_last_layer.self_attn._get_input_buffer(
            incremental_state
        )
        if self.decoder_last_layer.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.decoder_last_layer.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.decoder_last_layer.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(
                tgt_len,
                bsz,
                self.decoder_last_layer.nh,
                self.decoder_last_layer.head_dim,
            )
            x = torch.einsum("tbhd,h->tbhd", x, self.decoder_last_layer.c_attn)
            x = x.reshape(tgt_len, bsz, self.decoder_last_layer.embed_dim)
        if self.decoder_last_layer.attn_ln is not None:
            x = self.decoder_last_layer.attn_ln(x)
        x = self.decoder_last_layer.dropout_module(x)
        x = self.decoder_last_layer.residual_connection(x, residual)
        if not self.decoder_last_layer.normalize_before:
            x = self.decoder_last_layer.self_attn_layer_norm(x)

        if self.decoder_last_layer.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.decoder_last_layer.normalize_before:
                x = self.decoder_last_layer.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.decoder_last_layer.encoder_attn._set_input_buffer(
                    incremental_state, saved_state
                )

            x, attn = self.decoder_last_layer.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn
                or (
                    not self.decoder_last_layer.training
                    and self.decoder_last_layer.need_attn
                ),
                need_head_weights=need_head_weights,
            )
            x = self.decoder_last_layer.dropout_module(x)
            x = self.decoder_last_layer.residual_connection(x, residual)
            if not self.decoder_last_layer.normalize_before:
                x = self.decoder_last_layer.encoder_attn_layer_norm(x)

        residual = x
        if self.decoder_last_layer.normalize_before:
            x = self.decoder_last_layer.final_layer_norm(x)

        # Uses FFN input for nearest neighbor search
        features = None
        if self.use_ffn_input:
            # T x B x C -> B x T x C
            features = x.clone().transpose(0, 1)
            if self._debug:
                logger.info("Feature vector: The last FFN input.")
                self._debug = False

        x = self.decoder_last_layer.activation_fn(self.decoder_last_layer.fc1(x))
        x = self.decoder_last_layer.activation_dropout_module(x)
        if self.decoder_last_layer.ffn_layernorm is not None:
            x = self.decoder_last_layer.ffn_layernorm(x)
        x = self.decoder_last_layer.fc2(x)
        x = self.decoder_last_layer.dropout_module(x)
        if self.decoder_last_layer.w_resid is not None:
            residual = torch.mul(self.decoder_last_layer.w_resid, residual)
        x = self.decoder_last_layer.residual_connection(x, residual)
        if not self.decoder_last_layer.normalize_before:
            x = self.decoder_last_layer.final_layer_norm(x)
        if self.decoder_last_layer.onnx_trace and incremental_state is not None:
            saved_state = self.decoder_last_layer.self_attn._get_input_buffer(
                incremental_state
            )
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state, features
        return x, attn, None, features
