from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from knn_seq import utils
from knn_seq.data.token_storage import TokenStorage
from knn_seq.models.fairseq_knn_model_vanilla import FairseqKNNModel
from knn_seq.models.hf_model import HFModelBase
from knn_seq.search_index import FaissIndex
from knn_seq.search_index.torch_pq_index import TorchPQIndex

logger = logging.getLogger(__name__)


class FairseqSubsetKNNModel(FairseqKNNModel):
    """Subset kNN-MT class."""

    def set_index(
        self,
        val: TokenStorage,
        indexes: List[FaissIndex],
        src_knn_model: Optional[HFModelBase],
        src_val: TokenStorage,
        src_index: FaissIndex,
        knn_topk: int = 16,
        knn_temperature: float = 1.0,
        knn_threshold: Optional[float] = None,
        knn_weight: float = 0.5,
        src_topk: int = 16,
        precompute: bool = True,
        use_gpu: bool = False,
        use_fp16: bool = False,
    ):
        """Set kNN indexes.

        Args:
            val (TokenStorage): Value of the datastore.
            indexes (List[SearchIndex]): Key indexes of the datastore.
            src_knn_model (HFModelBase, optional): Sentence encoder model for the similar sentence search.
            src_val (TokenStorage): Value of the sentence datastore.
            src_index (FaissIndex): Key index of the sentence datastore.
            knn_topk (int): Retrieve the top-k nearest neighbors.
            knn_temperature (float): Temperature for the kNN probability distribution.
            knn_threshold (float, optional): Threshold which controls whether to use the retrieved examples.
            knn_weight (float): Weight for the kNN probabiltiy.
            src_topk (int): Retrieve the top-k nearest neighbor sentences.
            precompute (bool): Compute the distance between query and keys by using asymmetric distance computation (ADC).
            use_gpu (bool): Use GPU to compute the distance.
            use_fp16 (bool): Use fp16 on the distance computation.
        """

        # TODO(deguchi): ensemble
        index = indexes[0]
        self.val = val
        self.index = index
        self.subset_index = TorchPQIndex(
            self.index,
            use_gpu=use_gpu,
            use_fp16=use_fp16,
            precompute=precompute,
        )
        self.knn_topk = knn_topk
        self.knn_temperature = knn_temperature
        self.knn_threshold = knn_threshold
        self.knn_weight = knn_weight

        self.src_knn_model = src_knn_model
        self.src_val = src_val
        self.src_index = src_index
        self.src_topk = src_topk

        self.src_knn_timer = utils.StopwatchMeter()
        self.reorder_timer = utils.StopwatchMeter()

    def set_decoder_beam_size(self, beam_size: int) -> None:
        """Set beam size for efficient computation.

        Args:
            beam_size (int): Beam size.
        """
        super().set_decoder_beam_size(beam_size)
        self.beam_size = beam_size
        self.subset_index.set_beam_size(beam_size)

    def clear_cache(self) -> None:
        """Clear the caches.

        Subset kNN-MT has the following two caches on a CUDA memory during decoding.
        - `subset_tokens` (Tensor): Vocabulary IDs of the subset tokens of shape
           `(batch_size, max_subset_size)`.
        - `subset_index.subset_codes` (List[Tensor]): A list of uint8 PQ codes of subset
           tokens. Length of the list is equal to the batch size. The shape of each
           element is `(subset_size, M)`, where `M` is the number of subvectors in PQ.

        Since faiss and PyTorch manage CUDA memory separately, they must be explicitly
        released to reduce the memory footprint.
        """
        del self.subset_tokens
        del self.subset_index.subset_codes
        torch.cuda.empty_cache()

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
        if self.src_knn_model is None:
            src_query = self.extract_sentence_features_from_encoder_outs(encoder_outs)[
                0
            ]
        elif isinstance(self.src_knn_model, HFModelBase):
            tokenizer = self.src_knn_model.tokenizer
            src_query = self.src_knn_model(
                tokenizer.collate([tokenizer.encode(sent) for sent in self.src_sents])
            )
        else:
            raise NotImplementedError

        _, src_indices = self.src_index.search(
            src_query, k=self.src_topk, idmap=self.src_val.sort_order
        )
        self.src_knn = src_indices.tolist()

        subset_sent_idxs = self.val.orig_order[src_indices.numpy()]
        subset_indices = [
            np.concatenate([np.arange(b_ik, e_ik) for b_ik, e_ik in zip(b_i, e_i)])
            for b_i, e_i in zip(
                self.val.offsets[subset_sent_idxs],
                self.val.offsets[subset_sent_idxs + 1],
            )
        ]
        self.subset_index.set_subsets(
            [torch.LongTensor(subset) for subset in subset_indices]
        )
        subset_tokens = [
            torch.LongTensor(self.val.tokens[subset]) for subset in subset_indices
        ]
        self.subset_tokens = utils.pad(subset_tokens, self.pad).to(device)
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
        bsz, subset_size = self.subset_tokens.size()
        tokens = (
            self.subset_tokens.unsqueeze(1)
            .expand(bsz, self.beam_size, subset_size)
            .gather(dim=-1, index=indices.view(bsz, self.beam_size, self.knn_topk))
            .view(bsz * self.beam_size, self.knn_topk)
        )
        if self.knn_threshold is not None:
            scores[scores < self.knn_threshold] = float("-inf")
        probs = F.softmax(
            scores / self.knn_temperature, dim=-1, dtype=torch.float32
        ).type_as(querys)

        return self.KNNOutput(scores, probs, tokens)

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

        new_batch_order = new_order[:: self.beam_size] // self.beam_size

        if self.subset_tokens.size(0) == new_batch_order.size(0):
            self.reorder_timer.stop(len(new_order))
            return new_outs

        self.subset_tokens = self.subset_tokens.index_select(0, new_batch_order)
        self.subset_index.reorder_encoder_out(new_order)
        self.reorder_timer.stop(len(new_order))

        return new_outs
