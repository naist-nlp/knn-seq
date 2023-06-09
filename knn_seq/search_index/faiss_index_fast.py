import logging
from typing import Optional

import faiss
import numpy as np
import torch
from numpy import ndarray

from knn_seq.search_index.faiss_index import (
    FaissIndex,
    faiss_index_to_cpu,
    faiss_index_to_gpu,
)
from knn_seq.search_index.search_index import SearchIndexConfig

logger = logging.getLogger(__name__)


class FaissIndexFast(FaissIndex):
    """Wrapper for faiss index.

    This class contains highly experimental codes.
    Currently, this class implements:
      - Acclerate vector addition

    Args:
        index (faiss.Index): Faiss index.
        config (SearchIndexConfig): Search index configuration.

    Attributes:
        A (torch.Tensor, optional): A rotation matrix that is used in OPQ and PCA.
        b (torch.Tensor, optional): A bias vector that is used in PCA.
        gpu_ivf_index (faiss.Index, optional): GPU IVF index.
        gpu_ivf_cq (faiss.Index, optional): GPU coarse quantizer of the IVF index.
    """

    BACKEND_NAME = "faiss_fast"

    def __init__(self, index: faiss.Index, config: SearchIndexConfig, **kwargs) -> None:
        super().__init__(index, config, **kwargs)
        self.A: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.gpu_ivf_index: Optional[faiss.Index] = None
        self.gpu_ivf_cq: Optional[faiss.Index] = None

        if self.is_trained and (self.use_opq or self.use_pca):
            vt: faiss.LinearTransform = faiss.downcast_VectorTransform(
                faiss.downcast_index(index).chain.at(0)
            )
            self.A = torch.from_numpy(
                faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in).T
            )
            self.b = torch.from_numpy(faiss.vector_to_array(vt.b))

    def _to_gpu_rotation(self, fp16: bool = False) -> None:
        """Transfers the faiss index to GPUs for adding vectors.

        Args:
            fp16 (bool): Compute vector pre-transformation on fp16.
        """
        if self.A is not None and self.b is not None:
            self.A = self.A.cuda()
            self.b = self.b.cuda()
            if fp16:
                self.A = self.A.half()
                self.b = self.b.half()
            logger.info(
                f"Rotation matrices are on GPU: A={self.A.shape}, b={self.b.shape}"
            )

    def to_gpu_add(
        self,
        fp16: bool = True,
        gpu_ivf_full: bool = True,
        gpu_ivf_cq: bool = False,
    ) -> None:
        """Transfers the faiss index to GPUs for adding vectors.

        Args:
            fp16 (bool): Compute vector pre-transformation on fp16.
            gpu_ivf_full (bool): Use GPU for IVF vector addition.
            gpu_ivf_cq (bool): Use GPU for an IVF coarse quantizer.
        """
        self._to_gpu_rotation(fp16=fp16)

        if self.ivf is None:
            return

        if gpu_ivf_full and gpu_ivf_cq:
            raise ValueError(
                f"gpu_ivf_full and gpu_ivf_cq cannot be set True at the same time."
            )
        if gpu_ivf_full:
            if self.use_hnsw:
                ivf_index = self.ivf
                hnsw_quantizer = faiss.downcast_index(self.ivf.quantizer)
                ivf_index.quantizer = faiss.downcast_index(hnsw_quantizer.storage)
                self.gpu_ivf_index = faiss_index_to_gpu(ivf_index, shard=True)
                ivf_index.quantizer = hnsw_quantizer
            else:
                self.gpu_ivf_index = faiss_index_to_gpu(self.ivf, shard=True)
            self.gpu_ivf_index.reset()
        elif gpu_ivf_cq:
            coarse_quantizer = self.ivf.quantizer
            if self.use_hnsw:
                coarse_quantizer = faiss.downcast_index(coarse_quantizer).storage
            self.gpu_ivf_cq = faiss_index_to_gpu(faiss.downcast_index(coarse_quantizer))

    def rotate(self, x: torch.Tensor, shard_size: int = 2**20) -> torch.Tensor:
        """Rotate the input vectors.

        Args:
            x (torch.Tensor): Input vectors of shpae `(n, D)`
            shard_size (int): Number of rotated vectors at once.
              The default size is 2**20 (Each shard would take 2 GiB when D=256).

        Returns:
            torch.Tensor: Pre-transformed vectors of shape `(n, D)`.
        """
        if self.A is None:
            return x

        x = x.type(self.A.dtype)
        x_device = x.device
        A_device = self.A.device

        # Compute rotation of `x[i:j]` while `i < n`.
        results = []
        n = x.size(0)
        if shard_size <= 0:
            shard_size = n
        i = 0
        while i < n:
            j = min(i + shard_size, n)
            xs = x[i:j]
            xs = xs.to(A_device)
            xs @= self.A
            if self.b is not None and self.b.numel() > 0:
                xs += self.b
            results.append(xs.to(x_device))
            i = j

        return torch.cat(results, dim=0)

    def add_gpu_ivf_index(self, x: ndarray) -> None:
        """Adds vectors to the index with the full GPU IVF index.

        This method runs as follows:
        1. Transfers a trained IVF index to GPU devices. The index has only centroid
          information and does not have any vectors.
        2. Adds the input vectors to the temporary GPU index.
        3. Copies a replica of the GPU index to CPU.
        4. Copies the added vectors with their IVF lists from 3. to the real index.
        5. Empties the storage of the GPU index. Here, the GPU index has only centroids.

        Args:
            x (ndarray): Input vectors of shape `(n, D)`.
        """
        assert self.ivf is not None and self.gpu_ivf_index is not None

        a0 = self.ivf.ntotal
        a1 = a0 + x.shape[0]
        self.gpu_ivf_index.add_with_ids(x, np.arange(a0, a1))
        cpu_ivf_index: faiss.IndexIVF = faiss_index_to_cpu(self.gpu_ivf_index)
        assert cpu_ivf_index.ntotal == x.shape[0]
        faiss.extract_index_ivf(cpu_ivf_index).copy_subset_to(self.ivf, 0, a0, a1)
        self.gpu_ivf_index.reset()

    def add_gpu_ivf_cq(self, x: ndarray) -> None:
        """Adds vectors to the index with the GPU coarse quantizer of the IVF index.

        This method runs as follows:
        1. Transfers the coarse quantizer of the IVF index to GPU devices as
          `faiss.IndexFlat`. The coarse quantizer has centroid vectors.
        2. Searches the nearest neighbor centroid of the input vectors with the GPU
          coarse quantizer.
        3. Adds the input vectors and their nearest neighbor centroid IDs that are
          assigned by 2. to the IVF index.

        Args:
            x (ndarray): Input vectors of shape `(n, D)`.
        """
        assert self.ivf is not None and self.gpu_ivf_cq is not None
        assign = self.gpu_ivf_cq.search(x, k=1)[1].ravel()
        self.ivf.add_core(x.shape[0], faiss.swig_ptr(x), None, faiss.swig_ptr(assign))

    def add(self, vectors: np.ndarray) -> None:
        """Adds vectors to the index.

        Args:
            vectors (np.ndarray): Input vectors.
        """
        vectors = torch.from_numpy(vectors)
        if self.metric == "cos":
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        vectors = self.rotate(vectors)
        np_vectors = self.convert_to_numpy(vectors)

        index = self.index
        if self.use_opq or self.use_pca:
            index = faiss.downcast_index(self.index.index)

        if self.use_ivf and self.gpu_ivf_index is not None:
            self.add_gpu_ivf_index(np_vectors)
        elif self.use_ivf and self.gpu_ivf_cq is not None:
            self.add_gpu_ivf_cq(np_vectors)
        else:
            index.add(np_vectors)

        if self.use_opq or self.use_pca:
            self.index.ntotal = index.ntotal
