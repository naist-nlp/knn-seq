import logging
import sys
from typing import List, Optional, Tuple

import torch
from torch import BoolTensor, ByteTensor, FloatTensor, LongTensor, Tensor

from knn_seq import utils
from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.torch_pq_index_base import TorchPQIndexBase

logger = logging.getLogger(__name__)


class TorchPQIndex(TorchPQIndexBase):
    """Product Quantizer for PyTorch.

    Args:
        faiss_index (FaissIndex): Wrapped faiss index.
        padding_idx (int): The padding index for `subset_indices`. (default: -1)
        use_gpu (bool): Compute distances on GPUs.
        use_fp16 (bool): Compute distances on float16.
        shard_size (int, optional): Shard size for distance computation.
          If OOM is occured, decrease this parameter.
    """

    def __init__(
        self,
        faiss_index: FaissIndex,
        padding_idx: int = -1,
        precompute: bool = True,
        use_gpu: bool = True,
        use_fp16: bool = False,
        shard_size: Optional[int] = None,
    ):
        super().__init__(faiss_index, padding_idx=padding_idx, precompute=precompute)
        self.use_gpu = use_gpu
        self.use_fp16 = use_fp16

        if self.use_gpu:
            self.to_cuda()
            if self.use_fp16:
                self.to_fp16()

        # subset search
        self.subset_codes = None
        self.shard_size = (
            shard_size if shard_size is not None and self.use_gpu else sys.maxsize
        )

    def to_cuda(self) -> None:
        self._codewords = self.codewords.cuda()
        self.A = self.A.cuda()
        self.b = self.b.cuda()

    def to_fp16(self) -> None:
        self._codewords = self.codewords.half()
        self.A = self.A.half()
        self.b = self.b.half()

    def set_subsets(self, subset_indices: List[LongTensor], **kwargs) -> None:
        """Sets the subsets.

        Args:
            subset_indices (List[LongTensor]): The indices for subset search.
        """
        subset_indices = utils.pad(subset_indices, self.padding_idx)
        self.subset_codes = self.codes[subset_indices]
        if self.use_gpu:
            self.subset_codes = self.subset_codes.cuda()

    @torch.jit.export
    def reorder_encoder_out(self, new_order: LongTensor) -> None:
        """Reorder encoder output according to *new_order*.
        Args:
            new_order (LongTensor): Desired order
        """
        new_batch_order = new_order[:: self.beam_size] // self.beam_size
        if self.subset_codes.size(0) == len(new_batch_order):
            return
        self.subset_codes = self.subset_codes.index_select(
            0, new_batch_order.to(self.subset_codes.device)
        )

    @torch.jit.export
    def compute_distance_precompute(self, querys: Tensor, codes: ByteTensor) -> Tensor:
        """Computes distances by the pre-computed lookup table on PQ.

        The distance table `(bsz, num_keys)` are computed as follows:
        1. The given D-dimension querys are mapped to the M*dsub-dimension sub-spaces
            where M is the number of sub-spaces and assigned the byte-codewords.
        2. The distance lookup between the querys and the codewords is computed.
        3. The distances between querys and keys are computed by gathering from the lookup table.

        Args:
            querys (Tensor): Query vectors of shape `(bbsz, D)`.
            codes (ByteTensor): Quantized codes of shape `(bsz, num_keys, M)`.

        Returns:
            FloatTensor: Distance table of shape `(bsz, num_keys)`.
        """
        if self.pre_transform:
            querys = self.pre_encode(querys)

        # querys (float): (bbsz, D) -> (bsz, M, 1, dsub)
        # codewords (float): (M, ksub, dsub) -> (1, M, ksub, dsub)
        bbsz = querys.size(0)
        bsz = bbsz // self.beam_size
        distance_lookup = self.compute_distance(
            querys.view(bbsz, self.M, 1, self.dsub), self.codewords[None, :]
        )

        # distance_lookup (float): (bbsz, M, ksub) -> (bsz, beam, ksub, M)
        # codes (long): (bsz, beam, subset_size, M)
        # distances (float): (bsz, beam, subset_size)
        distance_lookup = (
            distance_lookup.transpose(1, 2)
            .contiguous()
            .view(bsz, self.beam_size, self.ksub, self.M)
        )
        subset_size = codes.size(1)
        distances = distance_lookup.new_full(
            (bsz, self.beam_size, subset_size), float("-inf")
        )

        shard_start = 0
        max_bsz = self.shard_size // subset_size // self.beam_size
        while shard_start < bsz:
            shard_end = min(shard_start + max_bsz, bsz)
            shard_bsz = shard_end - shard_start
            distances[shard_start:shard_end] = (
                distance_lookup[shard_start:shard_end]
                .gather(
                    dim=2,
                    index=codes[shard_start:shard_end]
                    .long()
                    .unsqueeze(1)
                    .expand(shard_bsz, self.beam_size, subset_size, self.M),
                )
                .sum(dim=-1)
            )
            shard_start = shard_end
        distances = distances.view(bbsz, subset_size).float()

        # distances (float): (bbsz, num_keys)
        return distances

    @torch.jit.export
    def compute_distance_flat(self, querys: Tensor, codes: ByteTensor) -> Tensor:
        """Computes distances by direct distance computation.

        Args:
            querys (Tensor): Query vectors of shape `(bbsz, D)`.
            codes (ByteTensor): Quantized codes of shape `(bsz, num_keys, M)`.

        Returns:
            Tensor: Distance table of shape `(bbsz, num_keys)`.
        """
        bbsz = querys.size(0)
        bsz = bbsz // self.beam_size
        querys = querys.view(bsz, self.beam_size, -1)
        subset_size = codes.size(1)
        distances = querys.new_full((bsz, self.beam_size, subset_size), float("-inf"))
        shard_start = 0
        max_bsz = self.shard_size // subset_size // self.beam_size
        while shard_start < bsz:
            shard_end = min(shard_start + max_bsz, bsz)
            shard_codes = codes[shard_start:shard_end]
            shard_keys = self.decode(shard_codes)
            distances[shard_start:shard_end] = self.compute_distance_table(
                querys[shard_start:shard_end], shard_keys
            )
            shard_start = shard_end
        distances = distances.view(bbsz, subset_size).float()

        # distances (float): (bbsz, num_keys)
        return distances

    @torch.jit.export
    def query(
        self,
        querys: FloatTensor,
        k: int = 1,
        key_padding_mask: Optional[BoolTensor] = None,
    ) -> Tuple[FloatTensor, LongTensor]:
        """Querys the k-nearest vectors to the index.

        Args:
            querys (FloatTensor): Query vectors of shape `(n, D)`.
            k (int): Number of nearest neighbors.
            key_padding_mask (BoolTensor, optional): Key padding mask of shape `(bsz, subset_size)`.

        Returns:
            FloatTensor: Top-k distances.
            LongTensor: Top-k indices.
        """
        if self.subset_codes is None:
            raise NotImplementedError(
                "TorchPQ class only supports fixed subset search."
            )

        device = querys.device
        if not self.use_gpu:
            querys = querys.cpu().float()
        elif not self.use_fp16:
            querys = querys.float()

        if self.precompute:
            distances = self.compute_distance_precompute(querys, self.subset_codes)
        else:
            distances = self.compute_distance_flat(querys, self.subset_codes)

        distances = distances.to(device)
        if self.metric == "l2":
            distances = distances.neg()

        if key_padding_mask is not None:
            distances = distances.view(-1, self.beam_size, distances.size(-1))
            distances = distances.masked_fill_(
                key_padding_mask.unsqueeze(1), float("-inf")
            )
            distances = distances.view(-1, distances.size(-1))

        return torch.topk(distances, k=k, dim=1)
