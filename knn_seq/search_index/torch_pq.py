import logging
from typing import Dict, List, Tuple

import torch
from torch import ByteTensor, FloatTensor, LongTensor, Tensor

from knn_seq import utils
from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.torch_pq_index_base import TorchPQIndexBase

logger = logging.getLogger(__name__)


class TorchPQIndex(TorchPQIndexBase):
    """Product Quantizer for PyTorch.

    Args:
        faiss_index (FaissIndex): wrapped faiss index.
        padding_idx (int): the padding index for `subset_indices`. (default: -1)
    """

    def __init__(
        self,
        faiss_index: FaissIndex,
        padding_idx: int = -1,
        precompute: bool = True,
        use_gpu: bool = True,
        use_half: bool = False,
    ):
        super().__init__(faiss_index, padding_idx=padding_idx, precompute=precompute)
        self.use_gpu = use_gpu
        self.use_half = use_half

        if self.use_gpu:
            self.to_cuda()
        if self.use_half:
            self.to_half()
        self._codes = self.codes.transpose_(0, 1).contiguous()

        # subset search
        self.subset_indices = None
        self.batched_subset_indices = None
        self.subset_codes = None
        self.subset_vectors = None
        self.shard_size = 20800000
        self.beam_size = 5

    def to_cuda(self) -> None:
        self._codewords = self.codewords.cuda()
        self.A = self.A.cuda()
        self.b = self.b.cuda()

    def to_half(self) -> None:
        self._codewords = self.codewords.half()
        self.A = self.A.half()
        self.b = self.b.half()

    @property
    def codes(self) -> ByteTensor:
        """Byte-encoded vectors of shape `(M, N)`."""
        return self._codes

    def distance_buf(self, t: Tensor) -> Tensor:
        """Gets distance buffer tensor of shape `(M, bsz, n)`.

        Args:
            t (Tensor): a tensor of shape `(M, bsz, n)`

        Returns:
            Tensor: buffered tensor of shape `(M, bsz, n)`.
        """
        M, bsz, subset_size = t.size()
        if (
            not hasattr(self, "_distance_buf")
            or self._distance_buf is None
            or self._distance_buf.size(1) < bsz
            or self._distance_buf.size(2) < subset_size
            or self._distance_buf.device != t.device
        ):
            self._distance_buf = t.new_zeros(M, bsz, subset_size).float()
        return self._distance_buf[:, :bsz, :subset_size]

    @property
    def need_sharding(self) -> bool:
        """Whether subset codes can be transfered to GPU without shard or not."""
        _, bsz, n = self.subset_codes.size()
        bbsz = bsz * self.beam_size
        return n * bbsz > self.shard_size

    def set_subsets(self, subset_indices: List[LongTensor], **kwargs) -> None:
        """Sets the subsets.

        Args:
            subset_indices (List[LongTensor]): the indices for subset search.
            clustering (bool): runs clustering for faster search.
        """
        self.subset_indices = subset_indices
        self.batch_idxs = torch.arange(len(subset_indices))
        self.batched_subset_indices = utils.pad(self.subset_indices, self.padding_idx)
        if self.use_gpu:
            self.batch_idxs = self.batch_idxs.cuda()
            self.batched_subset_indices = self.batched_subset_indices.cuda()
        self.subset_codes = self.codes[:, self.batched_subset_indices]
        if self.use_gpu and not self.need_sharding:
            self.subset_codes = self.subset_codes.cuda()

    @torch.jit.export
    def reorder_encoder_out(self, new_order: LongTensor) -> None:
        """Reorder encoder output according to *new_order*.

        Args:
            new_order (LongTensor): desired order
        """
        if not self.use_gpu:
            new_order = new_order.cpu()
            self.subset_indices = [self.subset_indices[i] for i in new_order]
            self.batched_subset_indices = utils.pad(
                self.subset_indices, self.padding_idx
            )
            self.subset_codes = self.codes[:, self.batched_subset_indices]
            return

        new_batch_order = self.batch_idxs.index_select(0, new_order[:: self.beam_size])
        self.batched_subset_indices = self.batched_subset_indices.index_select(
            0, new_batch_order
        )
        num_paddings = self.batched_subset_indices.eq(self.padding_idx).sum(dim=1).min()
        subset_size = self.batched_subset_indices.size(1)
        new_subset_size = subset_size - num_paddings
        if num_paddings > 0:
            self.batched_subset_indices = self.batched_subset_indices[
                :, :new_subset_size
            ].contiguous()
        new_bsz = new_batch_order.size(0)
        self.batch_idxs = (
            torch.arange(new_bsz)
            .unsqueeze(1)
            .expand(new_bsz, self.beam_size)
            .cuda()
            .view(-1)
        )
        if self.need_sharding:
            self.subset_codes = self.codes[:, self.batched_subset_indices]
            if not self.need_sharding:
                self.subset_codes = self.subset_codes.cuda()
        else:
            self.subset_codes = self.subset_codes.index_select(1, new_batch_order)
            if num_paddings > 0:
                self.subset_codes = self.subset_codes[
                    :, :, :new_subset_size
                ].contiguous()

    @torch.jit.export
    def compute_distance_precompute_cpu(
        self, querys: FloatTensor, codes: ByteTensor
    ) -> FloatTensor:
        """Computes distances by the pre-computed lookup table on PQ.

        The distance table `(bsz, num_keys)` are computed as follows:

        1. The given D-dimension querys are mapped to the M*dsub-dimension sub-spaces
            where M is the number of sub-spaces and assigned the byte-codewords.
        2. The distance lookup between the querys and the codewords is computed.
        3. The distances between querys and keys are computed by gathering from the lookup table.

        Args:
            querys (FloatTensor): query vectors of shape `(bsz, D)`.
            codes (ByteTensor): quantized codes of shape `(bsz, num_keys, M)`.

        Returns:
            FloatTensor: distance table of shape `(bsz, num_keys)`.
        """
        if self.pre_transform:
            querys = self.pre_encode(querys)

        # querys (float): (bsz, D) -> (bsz, M, 1, dsub)
        # codewords (float): (M, ksub, dsub) -> (1, M, ksub, dsub)
        bsz = querys.size(0)
        distance_lookup = self.compute_distance(
            querys.view(bsz, self.M, 1, self.dsub), self.codewords[None, :]
        )

        # distance_lookup (float): (bsz, M, ksub) -> (M, bsz, ksub)
        # codes (long): (M, bsz, subset_size)
        distance_lookup = distance_lookup.transpose_(0, 1).contiguous()
        distances = torch.gather(
            distance_lookup, dim=-1, index=codes.long(), out=self.distance_buf(codes)
        ).sum(dim=0)

        # distances (float): (bsz, num_keys)
        return distances

    @torch.jit.export
    def compute_distance_precompute_gpu(
        self, querys: FloatTensor, codes: ByteTensor
    ) -> FloatTensor:
        """Computes distances by the pre-computed lookup table on PQ.

        The distance table `(bsz, num_keys)` are computed as follows:

        1. The given D-dimension querys are mapped to the M*dsub-dimension sub-spaces
            where M is the number of sub-spaces and assigned the byte-codewords.
        2. The distance lookup between the querys and the codewords is computed.
        3. The distances between querys and keys are computed by gathering from the lookup table.

        Args:
            querys (FloatTensor): query vectors of shape `(bsz, D)`.
            codes (ByteTensor): quantized codes of shape `(bsz, M, num_keys)`.

        Returns:
            FloatTensor: distance table of shape `(bsz, num_keys)`.
        """
        if self.pre_transform:
            querys = self.pre_encode(querys)

        # querys (float): (bbsz, D) -> (bbsz, M, 1, dsub)
        # codewords (float): (M, ksub, dsub) -> (1, M, ksub, dsub)
        bbsz = querys.size(0)
        bsz = bbsz // self.beam_size
        distance_lookup = self.compute_distance(
            querys.view(bbsz, self.M, 1, self.dsub), self.codewords[None, :]
        )

        # distance_lookup (float): (bbsz, M, ksub) -> (M, bsz, beam, ksub)
        # codes (long): (M, bsz, beam, subset_size)
        distance_lookup = (
            distance_lookup.transpose_(0, 1)
            .contiguous()
            .view(self.M, bsz, self.beam_size, self.ksub)
        )

        def compute_shard_distance(
            shard_codes: ByteTensor, distance_lookup: Tensor
        ) -> Tensor:
            """Computes the distance of sharded codes.

            Args:
                shard_codes (ByteTensor): Sharded codes of shape `(M, bsz, shard_size)`.
                distance_lookup (Tensor): Lookup table of shape `(M, bsz, beam, ksub)`

            Returns:
                Tensor: The distance tensor of shape `(bsz, beam, shard_size)`
            """
            return (
                distance_lookup.gather(
                    dim=-1,
                    index=shard_codes.long()
                    .unsqueeze(2)
                    .expand(self.M, bsz, self.beam_size, shard_codes.size(2)),
                )
                .float()
                .sum(dim=0)
            )

        distances = []
        subset_size = codes.size(2)
        if self.need_sharding:
            shard_size = self.shard_size // bbsz
            last_shard_size = subset_size % shard_size
            for i in range(subset_size // shard_size):
                codes_i = codes[:, :, shard_size * i : shard_size * (i + 1)].cuda()
                distances.append(compute_shard_distance(codes_i, distance_lookup))
            if last_shard_size > 0:
                codes_i = codes[:, :, -last_shard_size:].cuda()
                distances.append(compute_shard_distance(codes_i, distance_lookup))
        else:
            distances.append(compute_shard_distance(codes, distance_lookup))

        distances = torch.cat(distances, dim=-1).view(bbsz, subset_size)

        # distances (float): (bbsz, num_keys)
        return distances

    @torch.jit.export
    def compute_distance_gpu(
        self, querys: FloatTensor, codes: ByteTensor
    ) -> FloatTensor:
        """Computes distances by the pre-computed lookup table on PQ.

        The distance table `(bsz, num_keys)` are computed as follows:

        1. The given D-dimension querys are mapped to the M*dsub-dimension sub-spaces
            where M is the number of sub-spaces and assigned the byte-codewords.
        2. The distance lookup between the querys and the codewords is computed.
        3. The distances between querys and keys are computed by gathering from the lookup table.

        Args:
            querys (FloatTensor): query vectors of shape `(bsz, D)`.
            codes (ByteTensor): quantized codes of shape `(bsz, num_keys, M)`.

        Returns:
            FloatTensor: distance table of shape `(bsz, num_keys)`.
        """

        def compute_shard_distance(
            querys: Tensor,
            shard_codes: ByteTensor,
        ) -> Tensor:
            """Computes the distance of sharded codes.

            Args:
                querys (Tensor): Query tensor of shape `(bbsz, D)`
                shard_codes (ByteTensor): Sharded codes of shape `(M, bsz, shard_size)`.

            Returns:
                Tensor: The distance tensor of shape `(bbsz, shard_size)`
            """
            # keys: (bsz, shard_size, D)
            keys = self.decode(shard_codes.permute(1, 2, 0).contiguous())
            return self.compute_distance_table(
                querys.view(keys.size(0), self.beam_size, -1).float(), keys.float()
            )

        bbsz = querys.size(0)
        distances = []
        subset_size = codes.size(2)
        if self.need_sharding:
            shard_size = self.shard_size // bbsz
            last_shard_size = subset_size % shard_size
            for i in range(subset_size // shard_size):
                codes_i = codes[:, :, shard_size * i : shard_size * (i + 1)].cuda()
                distances.append(compute_shard_distance(querys, codes_i))
            if last_shard_size > 0:
                codes_i = codes[:, :, -last_shard_size:].cuda()
                distances.append(compute_shard_distance(querys, codes_i))
        else:
            distances.append(compute_shard_distance(querys, codes))

        distances = torch.cat(distances, dim=-1).view(bbsz, subset_size)

        # distances (float): (bbsz, num_keys)
        return distances

    @torch.jit.export
    def query(
        self, querys: FloatTensor, k: int = 1
    ) -> Tuple[FloatTensor, Dict[str, LongTensor]]:
        """Querys the k-nearest vectors to the index.

        Args:
            querys (FloatTensor): query vectors of shape `(n, D)`.
            k (int): number of nearest neighbors.

        Returns:
            FloatTensor: top-k distances.
            Dict[str, LongTensor]: top-k indices.
        """
        if self.batched_subset_indices is None or self.subset_codes is None:
            raise NotImplementedError("TorchPQ class only supports subset search.")
        subset_indices = self.batched_subset_indices

        # subset_codes: (bsz, subset_size, M)
        if self.precompute:
            if not self.use_half:
                querys = querys.float()

            if querys.is_cuda:
                distances = self.compute_distance_precompute_gpu(
                    querys, self.subset_codes
                )
                # distances = self.compute_distance_gpu(querys, codes)
            else:
                distances = self.compute_distance_precompute_cpu(
                    querys, self.subset_codes
                )
            # end = time()
            # logger.info(f"End: Compute distance: {end - start}")
        else:
            querys = querys.half()
            keys = self.subset_vectors.half()
            # keys: [bsz, subset_size, D]
            # querys: [bsz, D]
            distances = self.compute_distance(querys[:, None], keys).float()

        distances = distances.view(-1, self.beam_size, distances.size(-1))
        distances = distances.masked_fill_(
            subset_indices.eq(self.padding_idx).unsqueeze(1), float("-inf")
        )
        distances = distances.view(-1, distances.size(-1))

        indices: Dict[str, LongTensor] = {}
        topk_distances, topk_indices = torch.topk(distances, k=k, dim=1)
        indices["k_indices"] = topk_indices

        bsz, subset_size = self.batched_subset_indices.size()
        subset_indices = subset_indices.unsqueeze(1).expand(
            bsz, self.beam_size, subset_size
        )
        indices["subset_indices"] = torch.gather(
            subset_indices, dim=2, index=topk_indices.view(-1, self.beam_size, k)
        ).view(-1, k)
        return topk_distances, indices
