import logging

import faiss
import numpy as np
import torch
from numpy import ndarray

logger = logging.getLogger(__name__)

from .faiss_index import FaissIndex


class FaissIndexFast(FaissIndex):
    """Wrapper for faiss index.

    This class contains highly experimental codes.

    Args:
        index (faiss.Index): faiss index.
        config (SearchIndexConfig): search index configuration.
    """

    def rotate(
        self,
        x: np.ndarray,
        use_gpu: bool = False,
        shard_size: int = 2**20,
        fp16: bool = False,
    ) -> np.ndarray:
        """Rotate the vectors.

        Args:
            x (np.array): `(n, D)`, where `D = M * dsub`.
                M is the number of subspaces, dsub is the dimension of each subspace.

        Returns:
            np.array: pre-transformed vectors of shape `(n, D)`.
        """
        if not (self.use_opq or self.use_pca):
            return x

        if not hasattr(self, "A") or not hasattr(self, "b"):
            vt: faiss.VectorTransform = faiss.downcast_VectorTransform(
                faiss.downcast_index(self.index).chain.at(0)
            )
            self.A = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in).T
            self.b = faiss.vector_to_array(vt.b)

        if not use_gpu:
            x = x @ self.A
            if self.b.size > 0:
                x += self.b
            return x

        if not hasattr(self, "A_gpu") or not hasattr(self, "b_gpu"):
            self.A_gpu = torch.from_numpy(self.A).cuda()
            self.b_gpu = torch.from_numpy(self.b).cuda()
            if fp16:
                self.A_gpu = self.A_gpu.half()
                self.b_gpu = self.b_gpu.half()

            logger.info(f"Rotate matrix on GPU: {self.A_gpu.shape}, {self.b_gpu.shape}")

        shards = []
        x = torch.from_numpy(x)
        if fp16:
            x = x.half()
        n = x.size(0)
        ns = n // shard_size
        for i in range(ns):
            xs = x[i * shard_size : (i + 1) * shard_size]
            xs = xs.cuda()
            xs = xs @ self.A_gpu
            if self.b_gpu.numel() > 0:
                xs += self.b_gpu
            shards.append(xs.cpu())
        if n % shard_size > 0:
            xs = x[ns * shard_size :]
            xs = xs.cuda()
            xs = xs @ self.A_gpu
            if self.b_gpu.numel() > 0:
                xs += self.b_gpu
            shards.append(xs.cpu())
        x = torch.cat(shards, dim=0)
        return x.float().numpy()

    def add_ivf_gpu(self, xb: ndarray, full_gpu: bool = True):
        logger.info(f"Rotate vectors: {xb.shape}")
        xb = self.rotate(xb, use_gpu=True, fp16=True)
        ivf_index: faiss.IndexIVF = self.ivf
        assert ivf_index is not None

        if getattr(self, "_gpu_ivf", None) is None:
            if self.use_hnsw:
                hnsw_quantizer = faiss.downcast_index(ivf_index.quantizer)
                ivf_index.quantizer = faiss.downcast_index(hnsw_quantizer.storage)
            self._gpu_ivf = faiss_index_to_gpu(
                ivf_index, reserve_vecs=xb.shape[0], shard=True, precompute=True
            )
            if self.use_hnsw:
                ivf_index.quantizer = hnsw_quantizer
            self._gpu_ivf.reset()
            self.use_gpu = True

        logger.info(f"Add vectors")
        a0 = ivf_index.ntotal
        a1 = a0 + xb.shape[0]
        self._gpu_ivf.add_with_ids(xb, np.arange(a0, a1))
        cpu_ivf: faiss.IndexIVF = faiss_index_to_cpu(self._gpu_ivf)
        assert cpu_ivf.ntotal == xb.shape[0]
        faiss.extract_index_ivf(cpu_ivf).copy_subset_to(ivf_index, 0, a0, a1)
        if self.use_opq or self.use_pca:
            self.index.ntotal = ivf_index.ntotal
        logger.info(f"Copied GPU-to-CPU: [{a0}, {a1})")
        self._gpu_ivf.reset()

    def add_pq_gpu(self, xb: ndarray):
        logger.info(f"Rotate vectors: {xb.shape}")
        xb = self.rotate(xb, use_gpu=True, fp16=True)
        index = self.index
        if self.use_opq or self.use_pca:
            index = faiss.downcast_index(self.index.index)
        assert isinstance(index, faiss.IndexPQ)

        logger.info(f"Add vectors")
        index.add(xb)
        if self.use_opq or self.use_pca:
            self.index.ntotal = index.ntotal
        return

    def add(self, vectors: ndarray, verbose: bool = False) -> None:
        """Adds vectors to the index.

        Args:
            vectors (ndarray): indexed vectors.
        """
        self.set_param("verbose", verbose)

        vectors = self.normalize(vectors)
        if self.use_gpu:
            if self.use_ivf:
                return self.add_ivf_gpu(vectors)
            if self.use_pq:
                return self.add_pq_gpu(vectors)
        return self.index.add(vectors)

    def ivf_to_gpu(self):
        ivf_index: faiss.IndexIVF = self.ivf
        if self.use_hnsw:
            ivf_index.quantizer = faiss.downcast_index(
                faiss.downcast_index(ivf_index.quantizer).storage
            )
        self.index = faiss_index_to_gpu(self.index, shard=True)
        self.use_gpu = True
