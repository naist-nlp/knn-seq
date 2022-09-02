from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import LongTensor, Tensor

from knn_seq import utils
from knn_seq.data.token_storage import TokenStorage
from knn_seq.models.fairseq_knn_model import FairseqKNNModel
from knn_seq.models.hf_model import HFAutoModelBase
from knn_seq.search_index import FaissIndex
from knn_seq.search_index.search_index import SearchIndex
from knn_seq.search_index.torch_pq import TorchPQIndex

logger = logging.getLogger(__name__)


class FairseqSubsetKNNModel(FairseqKNNModel):
    """A wrapper for subset kNN-MT."""

    def set_index(
        self,
        val: TokenStorage,
        indexes: List[SearchIndex],
        src_knn_model: Optional[HFAutoModelBase],
        src_val: TokenStorage,
        src_index: FaissIndex,
        knn_topk: int = 16,
        knn_temperature: float = 1.0,
        knn_threshold: Optional[float] = None,
        knn_weight: float = 0.5,
        src_topk: int = 16,
        src_knn_temperature: float = 1.0,
        shard_size: int = 20800000,
    ):
        index = indexes[0]
        self.val = val
        # TODO(deguchi): ensemble
        self.index = index
        self.subset_index = TorchPQIndex(self.index, use_gpu=True, use_half=False)
        self.subset_index.shard_size = shard_size
        self.knn_topk = knn_topk
        self.knn_temperature = knn_temperature
        self.knn_threshold = knn_threshold
        self.knn_weight = knn_weight

        self.src_knn_model = src_knn_model
        self.src_val = src_val
        self.src_index = src_index
        self.src_topk = src_topk
        self.src_knn_temperature = src_knn_temperature

        self.src_knn_timer = utils.StopwatchMeter()
        self.reorder_timer = utils.StopwatchMeter()

    @torch.jit.export
    def forward_encoder(
        self, net_input: Dict[str, Tensor]
    ) -> Optional[List[Dict[str, List[Tensor]]]]:
        if not self.has_encoder():
            return None

        self.src_indices: List[Tensor] = []
        encoder_outs = [
            model.encoder.forward_torchscript(net_input) for model in self.models
        ]

        # Source sentence search
        self.src_knn_timer.start()
        device = net_input["src_tokens"].device
        if self.src_knn_model is not None:
            tokenizer = self.src_knn_model.tokenizer
            src_query = self.src_knn_model(
                tokenizer.collate(tokenizer.encode_lines(self.src_sents))
            )
        else:
            src_query = self.extract_sentence_features_from_encoder_outs(
                encoder_outs,
            )[0]

        _, src_indices = self.src_index.search(
            src_query, k=self.src_topk, idmap=self.src_val.sort_order
        )
        self.src_knn = src_indices.tolist()

        subset_indices = [
            [self.val.get_interval(src_nbest_i_k) for src_nbest_i_k in src_nbest_i]
            for src_nbest_i in src_indices
        ]
        flatten_subset_indices = [
            np.concatenate(subset_i) for subset_i in subset_indices
        ]
        self.subset_index.set_subsets(
            [torch.LongTensor(subset) for subset in flatten_subset_indices]
        )
        self.subset_vocab_ids = utils.pad(
            [
                torch.LongTensor(self.val.tokens[subset])
                for subset in flatten_subset_indices
            ],
            self.pad,
        ).to(device)
        self.batch_idxs = torch.arange(len(self.subset_vocab_ids)).to(device)

        self.src_knn_timer.stop(n=len(src_query))

        return encoder_outs

    def search(
        self, querys: Tensor, index_id: int = 0
    ) -> FairseqSubsetKNNModel.KNNOutput:
        """Search k-nearest-neighbors.

        Args:
            querys (Tensor): A query tensor of shape `(batch_size, dim)`.
            index_id (int): Index ID for ensemble.

        Output:
            KNNOutput: A kNN output object.
        """
        scores, indices = self.subset_index.search(querys, k=self.knn_topk)
        bsz, subset_size = self.subset_vocab_ids.size()
        k_indices = indices["k_indices"].view(bsz, self.beam_size, self.knn_topk)
        vocab_indices = (
            self.subset_vocab_ids.unsqueeze(1)
            .expand(bsz, self.beam_size, subset_size)
            .gather(dim=2, index=k_indices)
            .view(bsz * self.beam_size, self.knn_topk)
        )

        if self.knn_threshold is not None:
            scores[scores < self.knn_threshold] = float("-inf")
        probs = utils.softmax(scores / self.knn_temperature).type_as(querys)

        return self.KNNOutput(scores, probs, vocab_indices)

    @torch.jit.export
    def reorder_encoder_out(
        self,
        encoder_outs: Optional[List[Dict[str, List[Tensor]]]],
        new_order: LongTensor,
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = super().reorder_encoder_out(
            encoder_outs, new_order
        )

        self.reorder_timer.start()
        if len(self.src_indices) > 0:
            self.src_indices = [self.src_indices[0].index_select(0, new_order)]

        new_batch_order = self.batch_idxs.index_select(0, new_order[:: self.beam_size])
        self.subset_vocab_ids = self.subset_vocab_ids.index_select(0, new_batch_order)
        num_paddings = self.subset_vocab_ids.eq(self.pad).sum(dim=1).min()
        subset_size = self.subset_vocab_ids.size(1)
        new_subset_size = subset_size - num_paddings
        if num_paddings > 0:
            self.subset_vocab_ids = self.subset_vocab_ids[
                :, :new_subset_size
            ].contiguous()
        new_bsz = new_batch_order.size(0)
        self.batch_idxs = (
            torch.arange(new_bsz)
            .unsqueeze(1)
            .expand(new_bsz, self.beam_size)
            .to(self.subset_vocab_ids.device)
            .view(-1)
        )
        self.subset_index.reorder_encoder_out(new_order)
        self.reorder_timer.stop(len(new_order))

        return new_outs
