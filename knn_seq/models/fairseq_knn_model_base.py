from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerModelBase
from fairseq.sequence_generator import EnsembleModel
from torch import LongTensor, Tensor

from knn_seq import utils
from knn_seq.models.fairseq_knn_transformer import KNNTransformer

logger = logging.getLogger(__name__)


class FairseqKNNModelBase(EnsembleModel, metaclass=abc.ABCMeta):
    """A base wrapper for kNN-MT."""

    def __init__(
        self,
        models: List[FairseqEncoderDecoderModel],
        key: KEY_CHOICES = "ffn_in",
        knn_ensemble: bool = False,
    ) -> None:
        super().__init__(models)
        self.init_model()
        self.tgt_dict: Dictionary = self.single_model.decoder.dictionary
        self.pad = self.tgt_dict.pad()

        self.wrapped_models = [
            KNNTransformer(m, key) if isinstance(m, TransformerModelBase) else m
            for m in models
        ]

        self.knn_weight: float = 0.0
        self.knn_threshold: Optional[float] = None
        self.knn_ensemble = knn_ensemble

        self.knn_timer = utils.StopwatchMeter()

    @abc.abstractmethod
    def set_index(self, *args, **kwargs) -> None:
        """Set the search index."""

    def init_model(self) -> None:
        for p in self.parameters():
            if getattr(p, "requires_grad", None) is not None:
                p.requires_grad = False
        self.eval()

    def get_embed_dim(self) -> List[int]:
        """Gets the embedding dimension size.

        Returns:
            List[int]: the embedding dimension size.
        """
        return [model.decoder.embed_tokens.embedding_dim for model in self.models]

    def set_src_sents(self, src_sents: List[str]) -> None:
        """Set source sentences for source-side retrieval."""
        self.src_sents = src_sents

    def set_decoder_beam_size(self, beam_size: int) -> None:
        """Set beam size for efficient beamable enc-dec attention."""
        super().set_decoder_beam_size(beam_size)
        self.beam_size = beam_size

    def clear_cache(self) -> None:
        """Clear the cache."""

    def forward(
        self,
        src_tokens: LongTensor,
        src_lengths: LongTensor,
        prev_output_tokens: Optional[LongTensor] = None,
        output_encoder_features: bool = False,
        **kawrgs,
    ) -> List[Tensor]:

        if output_encoder_features:
            encoder_sentence_features = self.extract_sentence_features(
                {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                }
            )
            # returns (B x C)
            return encoder_sentence_features

        decoder_outputs = [
            model(
                src_tokens,
                src_lengths,
                prev_output_tokens=prev_output_tokens,
                features_only=True,
            )
            for model in self.wrapped_models
        ]
        decoder_features = [
            decoder_out[1]["features"][0] for decoder_out in decoder_outputs
        ]

        return decoder_features

    def extract_sentence_features_from_encoder_outs(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]]
    ) -> Optional[List[Tensor]]:
        """Extracts sentnece features from encoder outputs.

        Args:
            encoder_outs (List[Dict[str, List[Tensor]]], optional): encoder outputs.

        Returns:
            List[Tensor]: sentence features of shape `(batch_size, feature_dim)`.
        """
        if encoder_outs is None:
            return encoder_outs

        # encoder_features: T x B x C
        encoder_features = [
            encoder_out["encoder_out"][0].float() for encoder_out in encoder_outs
        ]
        non_pad_mask = (
            (~(encoder_outs[0]["encoder_padding_mask"][0].transpose(1, 0)))
            .unsqueeze(-1)
            .float()
        )
        # returns B x C
        return [
            (
                (encoder_out * non_pad_mask).sum(dim=0)
                / torch.clamp(non_pad_mask.sum(dim=0), min=1e-9)
            ).to(encoder_out)
            for encoder_out in encoder_features
        ]

    def extract_sentence_features(self, net_inputs: Dict[str, Tensor]) -> List[Tensor]:
        """Extracts sentnece features.

        Args:
            net_inputs (Dict[str, Tensor]): network inputs.

        Returns:
            List[Tensor]: sentence features of shape `(batch_size, feature_dim)`.
        """
        encoder_outs = self.forward_encoder(net_inputs)
        return self.extract_sentence_features_from_encoder_outs(encoder_outs)

    @dataclass
    class KNNOutput:
        """kNN output dataclass.

        Attributes:
            scores (Tensor): Raw scores of shape `(batch_size, k)`.
            probs (Tensor): kNN probability of shape `(batch_size, k)`.
            indices (LongTensor): Vocaburary IDs of shape `(batch_size, k)`.
            uniq_indices (LongTensor, optional): Unique token indices of shape `(batch_size, k)`.
        """

        scores: Tensor
        probs: Tensor
        indices: LongTensor
        uniq_indices: Optional[LongTensor] = None

    @abc.abstractmethod
    def search(self, querys: Tensor, index_id: int = 0) -> KNNOutput:
        """Search k-nearest-neighbors.

        Args:
            querys (Tensor): A query tensor of shape `(batch_size, dim)`.
            index_id (int): Index ID for ensemble.

        Output:
            KNNOutput: A kNN output object.
        """

    def add_knn_probs(
        self, lprobs: Tensor, querys: Tensor, index_id: int = 0
    ) -> Tuple[Tensor, KNNOutput]:
        """Add kNN probabiltiy to MT probabiltiy.

        Args:
            lprobs (Tensor): MT output log-probabilty distribution of shape `(batch_size, vocab_size)`.
            querys (Tensor): Query vectors for kNN of shape `(batch_size, embed_dim)`.

        Returns:
            Tensor: kNN-MT interpolated log-probability distribution.
            KNNOutput: kNN output dataclass.
        """
        self.knn_timer.start()
        knn_output = self.search(querys, index_id=index_id)
        knn_probs = knn_output.probs

        knn_vocab_probs = knn_probs.new_zeros(*lprobs.size())
        knn_vocab_probs = knn_vocab_probs.scatter_add_(
            dim=-1, index=knn_output.indices, src=knn_probs
        )
        knn_vocab_probs *= self.knn_weight
        knn_vocab_probs[:, self.pad] = 0.0

        probs = torch.exp(lprobs)
        if self.knn_threshold is not None:
            max_scores = torch.max(knn_output.scores, dim=1).values
            update_batch_indices = max_scores.gt(self.knn_threshold)
            probs[update_batch_indices] = (1.0 - self.knn_weight) * probs[
                update_batch_indices
            ] + knn_vocab_probs[update_batch_indices]
        else:
            probs = (1.0 - self.knn_weight) * probs + knn_vocab_probs
        lprobs = torch.log(probs)

        self.knn_timer.stop()
        return lprobs, knn_output

    def forward_decoder_with_knn(
        self,
        tokens: Tensor,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, "FairseqKNNModelBase.KNNOutput"]:
        """Forwards a decoder with kNN search.

        This method is called from `self.forward_decoder()` in `SequenceGenerator`.

        Args:
            tokens (Tensor): Tokens tensor of shape `(bbsz, tgt_len)`.
            encoder_outs (List[Dict[str, List[Tensor]]]): Encoder outputs.
            incremental_states (List[Dict[str, Dict[str, Optional[Tensor]]]]): Fairseq incremental state objects.
            temperature (float): Temperature for multiple model ensemble.

        Returns:
            - Tensor: Log probability tensor of shape `(bbsz, vocab_size)`.
            - Tensor: Attention weight tensor of shape `(bbsz, src_len)`.
            - FairseqKNNModelBase.KNNOutput: kNN outputs.
        """
        # Ensemble MT outputs
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        knn_querys: Tensor
        for i, model in enumerate(self.wrapped_models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.forward_decoder(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                decoder_out = model.forward_decoder(tokens, encoder_out=encoder_out)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

                if self.knn_ensemble or i == 0:
                    knn_querys = decoder_out[1]["features"][0][:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            lprobs: Tensor = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            lprobs = lprobs[:, -1, :]

            # kNN-MT
            if self.knn_ensemble:
                lprobs, knn_output = self.add_knn_probs(lprobs, knn_querys, index_id=i)

            if self.models_size == 1:
                break

            log_probs.append(lprobs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        if self.models_size > 1:
            avg_lprobs = torch.logsumexp(
                torch.stack(log_probs, dim=0), dim=0
            ) - math.log(self.models_size)

            if avg_attn is not None:
                avg_attn.div_(self.models_size)

            lprobs = avg_lprobs
            attn = avg_attn

        # kNN-MT
        if not self.knn_ensemble:
            lprobs, knn_output = self.add_knn_probs(lprobs, knn_querys)

        return lprobs, attn, knn_output

    def forward_decoder(
        self,
        tokens: Tensor,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """Forwards a decoder.

        This method is called in the `SequenceGenerator` class.

        Args:
            tokens (Tensor): Tokens tensor of shape `(bbsz, tgt_len)`.
            encoder_outs (List[Dict[str, List[Tensor]]]): Encoder outputs.
            incremental_states (List[Dict[str, Dict[str, Optional[Tensor]]]]): Fairseq incremental state objects.
            temperature (float): Temperature for multiple model ensemble.

        Returns:
            - Tensor: Log probability tensor of shape `(bbsz, vocab_size)`.
            - Tensor: Attention weight tensor of shape `(bbsz, src_len)`.
        """
        lprobs, attn, knn_output = self.forward_decoder_with_knn(
            tokens, encoder_outs, incremental_states, temperature
        )
        return lprobs, attn
