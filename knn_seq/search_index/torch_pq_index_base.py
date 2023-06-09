import logging
from typing import Any, Dict, Optional, Tuple, Union

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ByteTensor, FloatTensor, LongTensor, Tensor

from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.search_index import SearchIndex

logger = logging.getLogger(__name__)


class TorchPQIndexBase(SearchIndex, nn.Module):
    """Base class for Product Quantizer implemented on PyTorch.

    c.f. https://gist.github.com/mdouze/94bd7a56d912a06ac4719c50fa5b01ac

    Args:
        faiss_index (FaissIndex): wrapped faiss index.
        padding_idx (int): the padding index for `subset_indices`. (default: -1)
    """

    def __init__(
        self,
        faiss_index: FaissIndex,
        padding_idx: int = -1,
        precompute: bool = True,
    ):
        assert faiss_index.is_trained
        nn.Module.__init__(self)
        super().__init__(faiss_index, faiss_index.config)

        self.padding_idx = padding_idx
        self.precompute = precompute

        index = faiss_index.index
        if isinstance(index, faiss.IndexPreTransform):
            vtrans = faiss.downcast_VectorTransform(index.chain.at(0))
            assert isinstance(vtrans, faiss.LinearTransform)
            self.pre_transform = True
            self.A = torch.from_numpy(
                faiss.vector_to_array(vtrans.A).reshape(vtrans.d_out, vtrans.d_in)
            )
            self.b = torch.from_numpy(faiss.vector_to_array(vtrans.b))
            pq_index = faiss.downcast_index(index.index)
        else:
            self.pre_transform = False
            pq_index = index

        assert isinstance(pq_index, faiss.IndexPQ)
        pq = pq_index.pq
        codewords = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
        assert pq.nbits == 8

        self._codewords = torch.from_numpy(codewords)
        self._codes = torch.from_numpy(
            faiss.vector_to_array(pq_index.codes).reshape(-1, pq.M)
        )
        self._dim = codewords.shape[0] * codewords.shape[2]

        self.nprobe = 1

    def __len__(self) -> int:
        """Returns the number of indexed data."""
        return len(self.codes)

    @property
    def dim(self) -> int:
        """Returns the dimension size."""
        return self._dim

    @property
    def M(self) -> int:
        """Returns the number of sub-vectors."""
        return self.codewords.size(0)

    @property
    def ksub(self) -> int:
        """Returns the number of code bits."""
        return self.codewords.size(1)

    @property
    def dsub(self) -> int:
        """Returns the dimesion size of each sub-vector."""
        return self.codewords.size(2)

    @property
    def codewords(self) -> FloatTensor:
        """Codewords that map byte-codes to float-vectors of shape `(M, ksub, dsub)`."""
        return self._codewords

    @property
    def codes(self) -> ByteTensor:
        """Byte-encoded vectors of shape `(N, M)`."""
        return self._codes

    @classmethod
    def new(cls, metric: str, dim: int, **kwargs) -> "SearchIndex":
        """Builds a new search index instance.

        Args:
            metric (str): distance function.
            dim (int): dimension size of vectors.
            kwargs: backend specific keyword arguments.

        Returns:
            SearchIndex: a new search index instance.
        """
        raise NotImplementedError

    def normalize(self, vectors: Tensor, cpu: bool = True) -> FloatTensor:
        """Normalizes the given vectors.

        Args:
            vectors (Tensor): input vectors of shape `(n, D)`.

        Returns:
            ndarray: normalzied np.float32 array.
        """
        if cpu:
            vectors = vectors.cpu().float()
        if self.metric == "cos":
            assert vectors.dim() == 2
            vectors = F.normalize(vectors.float()).to(vectors)
        return vectors

    def compute_distance(
        self, a: FloatTensor, b: FloatTensor, fn: Optional[str] = None
    ) -> FloatTensor:
        """Computes distance between two vectors.

        The output values are always returned as similarity, so L2-distance will be negatived.

        Args:
            a (FloatTensor): float vectors of shape `(..., D)`.
            b (FloatTensor): float vectors of shape `(..., D)`.
            fn (str, optional): distance function.

        Returns:
            FloatTensor: distances between `a` and `b` of shape `(...,)`.
        """
        fn = fn if fn is not None else self.metric
        if fn == "l2":
            return -((a - b) ** 2).sum(dim=-1)
        if fn == "ip" or fn == "cos":
            return (a * b).sum(dim=-1)
        else:
            raise NotImplementedError

    def compute_distance_table(
        self, a: FloatTensor, b: FloatTensor, fn: Optional[str] = None
    ) -> FloatTensor:
        """Computes distance between two vectors.

        The output values are always returned as similarity, so L2-distance will be negatived.

        Args:
            a (FloatTensor): float vectors of shape `(bsz, m, D)`.
            b (FloatTensor): float vectors of shape `(bsz, n, D)`.
            fn (str, optional): distance function.

        Returns:
            FloatTensor: distances between `a` and `b` of shape `(...,)`.
        """
        fn = fn if fn is not None else self.metric
        if fn == "l2":
            return -((a - b) ** 2).sum(dim=-1)
        if fn == "ip" or fn == "cos":
            return torch.bmm(a, b.transpose(1, 2))
        else:
            raise NotImplementedError

    @property
    def is_trained(self) -> bool:
        """Returns the index is trained or not."""
        return self.index.is_trained

    def pre_encode(self, x: FloatTensor) -> FloatTensor:
        """Pre-encodes the vectors.

        Args:
            x (FloatTensor): `(n, D)`, where `D = M * dsub`.
                M is the number of subspaces, dsub is the dimension of each subspace.

        Returns:
            FloatTensor: pre-transformed vectors of shape `(n, D)`.
        """
        A, b = self.A, self.b
        x = x @ A.t()
        if b.numel() > 0:
            x += b
        return x

    @torch.jit.export
    def encode(self, x: FloatTensor) -> ByteTensor:
        """Encodes the vectors to the codes.

        Args:
            x (FloatTensor): `(n, D)`, where `D = M * dsub`.
                M is the number of subspaces, dsub is the dimension of each subspace.

        Returns:
            ByteTensor: uint8 codes of shape `(n, M)`.
        """
        if self.pre_transform:
            x = self.pre_encode(x)

        n, d = x.shape
        codewords = self.codewords
        M, ksub, dsub = codewords.shape

        x = x.view(n, M, 1, dsub)

        # x: (n, M, 1, dsub)
        # codewords: (1, M, ksub, dsub)
        codewords = codewords.unsqueeze(0)
        # distance: (n, M, ksub)
        distance = self.compute_distance(x, codewords, "l2")
        codes = distance.argmax(dim=2).byte()
        return codes

    @torch.jit.export
    def decode(self, codes: ByteTensor) -> FloatTensor:
        """Decodes the codes to the vectors.

        Args:
            codes (ByteTensor): `(n, M)`, where M is the number of subspaces.

        Returns:
            FloatTensor: (n, D), where `D = M * dsub`, and dsub is the dimension of subspace.
        """
        code_shape = codes.shape[:-1]
        codes = codes.flatten(end_dim=-2)
        n, MM = codes.shape
        codewords = self.codewords
        M, ksub, dsub = codewords.size()
        assert MM == M

        # x[n, m, j] = codewords[m][codes[n][m]][j]
        x = torch.gather(
            codewords[None, :].expand(n, M, ksub, dsub),
            dim=2,
            index=codes.long().unsqueeze(-1).unsqueeze(-1).expand(n, M, 1, dsub),
        )
        # (n, M, 1, dsub) -> (n, D)
        x = x.view(n, -1)

        if self.pre_transform:
            A, b = self.A, self.b
            if b.numel() > 0:
                x -= b
            x = x @ A

        return x.view(*code_shape, -1)

    def train(self, vectors) -> None:
        """Trains the index for some approximate nearest neighbor search algorithms.

        Args:
            vectors (ndarray): input vectors.
        """

    def add(self, vectors: FloatTensor, ids: Optional[LongTensor] = None) -> None:
        """Adds vectors to the index.

        Args:
            vectors (ndarray): indexed vectors.
            ids (Optional[ndarray]): indices of the index.
        """

    def reconstruct(self, indices: Union[LongTensor, slice]) -> FloatTensor:
        codes = self.codes[indices]
        return self.decode(codes)

    def set_nprobe(self, nprobe: int) -> None:
        """Sets the `nprobe` parameter for IVF.

        Args:
            nprobe (int): the number of probes for IVF search.
        """
        self.nprobe = nprobe

    @torch.jit.export
    def postprocess_search(
        self,
        distances: FloatTensor,
        indices: Dict[str, LongTensor],
        idmap: Optional[LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, LongTensor]]:
        """Post-processes the search results.

        Args:
            distances (FloatTensor): top-k distances.
            indices (Dict[str, LongTensor]): top-k indices.
            idmap (LongTensor, optional): if given, maps the ids. (e.g., [3, 5] -> {0: 3, 1: 5})
        """
        if idmap is not None:
            indices["k_ids"] = idmap[indices["k_indices"]]

        if self.metric == "l2":
            distances: torch.FloatTensor = distances.neg()

        return distances, indices

    @torch.jit.export
    def search(
        self,
        querys: Tensor,
        k: int = 1,
        idmap: Optional[LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.LongTensor]]:
        """Searches the k-nearest vectors.

        Args:
            querys (Tensor): query vectors.
            k (int): number of nearest neighbors.
            idmap (LongTensor, optional): if given, maps the ids. (e.g., [3, 5] -> {0: 3, 1: 5})

        Returns:
            FloatTensor: top-k scores or distances.
            Dict[str, torch.LongTensor]: top-k indices.
        """
        assert self.is_trained
        querys = self.normalize(querys, cpu=False)
        distances, indices = self.query(querys, k=k)
        return self.postprocess_search(distances, indices, idmap=idmap)

    def clear(self) -> None:
        """Clears the index."""

    @classmethod
    def load_index(cls, path: str) -> Any:
        """Loads the index.

        Args:
            path (str): index file path.

        Returns:
            Any: the wrapped index.
        """

    def save_index(self, path: str) -> None:
        """Saves the index.

        Args:
            path (str): index file path to save.
        """
