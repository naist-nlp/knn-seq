import logging
from typing import List, Optional, Tuple

import torch
from torch import ByteTensor, FloatTensor, LongTensor, Tensor

from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.torch_pq_index_base import TorchPQIndexBase

logger = logging.getLogger(__name__)


class TorchPQIndex(TorchPQIndexBase):
    """Product Quantizer for PyTorch.

    Args:
        faiss_index (FaissIndex): Wrapped faiss index.
        use_gpu (bool): Compute distances on GPUs.
        use_fp16 (bool): Compute distances on float16.
    """

    def __init__(
        self,
        faiss_index: FaissIndex,
        precompute: bool = True,
        use_gpu: bool = True,
        use_fp16: bool = False,
    ):
        super().__init__(faiss_index, precompute=precompute)
        self.use_gpu = use_gpu
        self.use_fp16 = use_fp16

        if self.use_gpu:
            self.to_cuda()
            if self.use_fp16:
                self.to_fp16()

        # subset search
        self.subset_codes: List[Tensor] = []

    def to_cuda(self) -> None:
        self._codewords = self.codewords.cuda()
        if self.A is not None:
            self.A = self.A.cuda()
        if self.b is not None:
            self.b = self.b.cuda()

    def to_fp16(self) -> None:
        self._codewords = self.codewords.half()
        if self.A is not None:
            self.A = self.A.half()
        if self.b is not None:
            self.b = self.b.half()

    def set_subsets(self, subset_indices: List[LongTensor]) -> None:
        """Sets the subsets.

        Args:
            subset_indices (List[LongTensor]): The indices for subset search.
        """
        self.subset_codes = [self.codes[idxs] for idxs in subset_indices]
        if self.use_gpu:
            self.subset_codes = [codes.cuda() for codes in self.subset_codes]

    @torch.jit.export
    def reorder_encoder_out(self, new_order: LongTensor) -> None:
        """Reorder encoder output according to *new_order*.
        Args:
            new_order (LongTensor): Desired order
        """
        new_batch_order = (new_order[:: self.beam_size] // self.beam_size).tolist()
        if len(self.subset_codes) == len(new_batch_order):
            return
        self.subset_codes = [self.subset_codes[i] for i in new_batch_order]

    @torch.jit.export
    def compute_distance_precompute(
        self, querys: Tensor, codes: List[ByteTensor]
    ) -> Tensor:
        """Computes distances by the pre-computed lookup table on PQ.

        The distance table `(bsz, num_keys)` are computed as follows:
        1. The given D-dimension querys are mapped to the M*dsub-dimension sub-spaces
            where M is the number of sub-spaces and assigned the byte-codewords.
        2. The distance lookup between the querys and the codewords is computed.
        3. The distances between querys and keys are computed by gathering from the lookup table.

        Args:
            querys (Tensor): Query vectors of shape `(bbsz, D)`.
            codes (List[ByteTensor]): Quantized codes of shape `(num_keys, M)` x bsz.

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
        subset_size = max([len(x) for x in codes])
        distances = distance_lookup.new_full(
            (bsz, self.beam_size, subset_size),
            float("inf") if self.metric == "l2" else float("-inf"),
        )

        for i, c in enumerate(codes):
            distances[i, :, : len(c)] = (
                distance_lookup[i]
                .gather(
                    dim=-2,
                    index=c.long().unsqueeze(0).expand(self.beam_size, len(c), self.M),
                )
                .sum(dim=-1)
            )
        distances = distances.view(bbsz, subset_size).float()

        # distances (float): (bbsz, num_keys)
        return distances

    @torch.jit.export
    def compute_distance_flat(self, querys: Tensor, codes: List[ByteTensor]) -> Tensor:
        """Computes distances by direct distance computation.

        Args:
            querys (Tensor): Query vectors of shape `(bbsz, D)`.
            codes (List[ByteTensor]): Quantized codes of shape `(num_keys, M)` x bsz.

        Returns:
            Tensor: Distance table of shape `(bbsz, num_keys)`.
        """
        bbsz = querys.size(0)
        bsz = bbsz // self.beam_size
        querys = querys.view(bsz, self.beam_size, -1)
        subset_size = max([len(x) for x in codes])
        distances = querys.new_full(
            (bsz, self.beam_size, subset_size),
            float("inf") if self.metric == "l2" else float("-inf"),
        )
        for i, c in enumerate(codes):
            distances[i, :, : len(c)] = self.compute_distance_table(
                querys[i], self.decode(c)
            )
        distances = distances.view(bbsz, subset_size).float()

        # distances (float): (bbsz, num_keys)
        return distances

    @torch.jit.export
    def query(self, querys: FloatTensor, k: int = 1) -> Tuple[FloatTensor, LongTensor]:
        """Querys the k-nearest vectors to the index.

        Args:
            querys (FloatTensor): Query vectors of shape `(n, D)`.
            k (int): Number of nearest neighbors.

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

        return torch.topk(distances, k=k, dim=1)
