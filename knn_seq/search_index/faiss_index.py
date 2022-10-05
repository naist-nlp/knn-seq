import logging
from typing import Any, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from numpy import ndarray
from typing_extensions import TypeAlias

from knn_seq.search_index.search_index import SearchIndex, SearchIndexConfig

logger = logging.getLogger(__name__)

# We wrap the faiss types here to prevent errors when running with faiss-cpu
# faiss-cpu doesn't have faiss.GpuIndex
GpuIndex: TypeAlias = faiss.GpuIndex if hasattr(faiss, "GpuIndex") else faiss.Index


def faiss_index_to_gpu(
    index: faiss.Index,
    num_gpus: int = -1,
    reserve_vecs: Optional[int] = None,
    shard: bool = False,
    precompute: bool = False,
) -> GpuIndex:
    """Transfers the index from CPU to GPU.

    Args:
        index (faiss.Index): faiss index.
        num_gpus (int): number of GPUs to use.
        reserve_vecs (int, optional): number of the max index size.
        shard (bool): builds an IndexShards.

    Returns:
        faiss.GpuIndex: faiss index.
    """
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = True
    co.indicesOptions = faiss.INDICES_CPU
    if precompute:
        logger.info(f"Use precompute table on GPU.")
        co.usePrecomputed = precompute
    if reserve_vecs is not None:
        co.reserveVecs = reserve_vecs

    if shard:
        co.shard = True
        logger.warning("The index will be sharded.")

    return faiss.index_cpu_to_all_gpus(index, co, ngpu=num_gpus)


def faiss_index_to_cpu(index: GpuIndex) -> faiss.Index:
    """Transfers the index from GPU to CPU.

    Args:
        index (faiss.GpuIndex): faiss index.

    Returns:
        faiss.Index: faiss index.
    """
    return faiss.index_gpu_to_cpu(index)


class FaissIndex(SearchIndex):
    """Wrapper for faiss index.

    Args:
        index (faiss.Index): faiss index.
        config (SearchIndexConfig): search index configuration.
        nprobe (int): number of searched clusters (default: 16)
    """

    METRICS_MAP = {
        "l2": faiss.METRIC_L2,
        "ip": faiss.METRIC_INNER_PRODUCT,
        "cos": faiss.METRIC_INNER_PRODUCT,
    }

    def __len__(self) -> int:
        return self.index.ntotal

    def set_param(self, name: str, param: Any):
        if self.use_gpu:
            faiss.GpuParameterSpace().set_index_parameter(self.index, name, param)
        return faiss.ParameterSpace().set_index_parameter(self.index, name, param)

    def set_nprobe(self, nprobe: int):
        if self.use_ivf:
            self.set_param("nprobe", nprobe)

    def set_efsearch(self, efsearch: int):
        if self.use_hnsw:
            self.set_param("efSearch", efsearch)

    @property
    def dim(self) -> int:
        """Returns the dimension size."""
        return self.index.d

    @property
    def ivf(self) -> Optional[faiss.IndexIVF]:
        if not self.use_ivf:
            return None
        return faiss.extract_index_ivf(self.index)

    @classmethod
    def new(
        cls,
        metric: str,
        dim: int,
        hnsw_edges: int = 0,
        ivf_lists: int = 0,
        pq_subvec: int = 0,
        transform_dim: int = -1,
        use_opq: bool = False,
        use_pca: bool = False,
        **kwargs,
    ) -> "FaissIndex":
        """Builds a new faiss index instance.

        Args:
            metric (str): distance function.
            dim (int): dimension size of vectors.

        Returns:
            FaissIndex: a new faiss index instance.
        """

        config = SearchIndexConfig(
            backend="faiss",
            metric=metric,
            hnsw_edges=hnsw_edges,
            ivf_lists=ivf_lists,
            pq_subvec=pq_subvec,
            transform_dim=transform_dim,
            use_opq=use_opq,
            use_pca=use_pca,
        )
        faiss_metric_type = FaissIndex.METRICS_MAP[metric]

        assert not (config.use_opq and config.use_pca)
        vtrans = None
        if config.use_opq:
            vtrans = faiss.OPQMatrix(dim, M=pq_subvec, d2=transform_dim)
            if transform_dim > 0:
                dim = transform_dim
        elif config.use_pca:
            vtrans = faiss.PCAMatrix(dim, d_out=transform_dim)
            if transform_dim > 0:
                dim = transform_dim

        if hnsw_edges <= 0:
            quantizer = faiss.IndexFlat(dim, faiss_metric_type)
        else:
            quantizer = faiss.IndexHNSWFlat(dim, hnsw_edges, faiss_metric_type)

        if ivf_lists <= 0:
            if pq_subvec <= 0:
                index = quantizer
            else:
                if hnsw_edges <= 0:
                    index = faiss.IndexPQ(dim, pq_subvec, 8, faiss_metric_type)
                else:
                    index = faiss.IndexHNSWPQ(dim, pq_subvec, hnsw_edges)
        else:
            if pq_subvec <= 0:
                index = faiss.IndexIVFFlat(quantizer, dim, ivf_lists, faiss_metric_type)
            else:
                index = faiss.IndexIVFPQ(
                    quantizer, dim, ivf_lists, pq_subvec, 8, faiss_metric_type
                )

        if vtrans is not None:
            index = faiss.IndexPreTransform(vtrans, index)

        return cls(index, config)

    def to_gpu(self) -> None:
        """Transfers the faiss index to GPUs."""
        if not self.is_trained:
            if self.use_pq:
                index = self.index.index if self.use_opq or self.use_pca else self.index
                pq = faiss.downcast_index(index).pq
                assign_index = faiss_index_to_gpu(
                    faiss.IndexFlatL2(pq.dsub), num_gpus=1
                )
                pq.assign_index = assign_index
                if self.use_opq:
                    opq = faiss.downcast_VectorTransform(self.index.chain.at(0))
                    opq.pq = pq

                self.use_gpu = True

            ivf = self.ivf
            if ivf is not None:
                dim = (
                    self.config.transform_dim
                    if self.config.transform_dim > 0
                    else self.dim
                )
                clustering_index = faiss_index_to_gpu(
                    faiss.IndexFlat(dim, self.METRICS_MAP[self.metric]), shard=True
                )
                ivf.clustering_index = clustering_index
                self.use_gpu = True
        else:
            if self.use_ivf or self.use_pq:
                self.use_gpu = True

    def to_cpu(self) -> None:
        """Transfers the faiss index to CPUs."""
        self.index = faiss_index_to_cpu(self.index)
        self.use_gpu = False

    @property
    def is_trained(self) -> bool:
        """Returns the faiss index is trained or not."""
        return self.index.is_trained

    def train(self, vectors: ndarray, verbose: bool = False) -> None:
        """Trains the faiss index for some approximate nearest neighbor search algorithms.

        Args:
            vectors (ndarray): input vectors.
            verbose (bool): display the verbose messages.
        """
        vectors = self.normalize(vectors)
        logger.info("Training on {}".format("GPU" if self.use_gpu else "CPU"))

        self.set_param("verbose", verbose)
        self.index.train(vectors)
        self.set_param("verbose", False)

    @property
    def gpu_quantizer(self):
        if not hasattr(self, "_gpu_quantizer"):
            ivf: faiss.IndexIVF = faiss.extract_index_ivf(self.index)
            quantizer = ivf.quantizer
            if self.use_hnsw:
                cq = faiss.downcast_index(faiss.downcast_index(quantizer).storage)
            else:
                cq = faiss.downcast_index(quantizer)

            self._gpu_quantizer = faiss_index_to_gpu(cq, num_gpus=1)
            self.use_gpu = True
        return self._gpu_quantizer

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
        full_gpu = full_gpu and self.config.pq_subvec <= 96

        if not full_gpu:
            logger.info(f"Assign clusters")
            assign = self.gpu_quantizer.search(xb, k=1)[1].ravel()
            nb = xb.shape[0]
            logger.info(f"Add vectors")
            ivf_index.verbose = False
            ivf_index.add_core(nb, faiss.swig_ptr(xb), None, faiss.swig_ptr(assign))
            ivf_index.verbose = True
            if self.use_opq or self.use_pca:
                self.index.ntotal = ivf_index.ntotal
            return

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

    def postprocess_search(
        self, distances: ndarray, indices: ndarray, idmap: Optional[ndarray] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Post-processes the search results.

        Args:
            distances (ndarray): top-k distances.
            indices (ndarray): top-k indices.
            idmap (ndarray, optional): if given, maps the ids. (e.g., [3, 5] -> {0: 3, 1: 5})

        Returns:
            - torch.FloatTensor: top-k scores.
            - torch.LongTensor: top-k indices.
        """
        if idmap is not None:
            indices = idmap[indices]

        distances_tensor = torch.FloatTensor(distances)
        indices_tensor = torch.LongTensor(indices)

        if self.metric == "l2":
            distances_tensor = distances_tensor.neg()
        elif (
            self.use_hnsw and self.use_pq and not self.use_ivf
        ) and self.metric == "cos":
            # HNSWPQ index does not support IP metric.
            distances_tensor = (2 - distances_tensor) / 2
        return distances_tensor, indices_tensor

    def query(self, querys: ndarray, k: int = 1) -> Tuple[ndarray, ndarray]:
        """Querys the k-nearest vectors to the index.

        Args:
            querys (ndarray): query vectors.
            k (int): number of nearest neighbors.

        Returns:
            ndarray: top-k distances.
            ndarray: top-k indices.
        """
        return self.index.search(querys, k=k)

    def clear(self) -> None:
        """Clears the index."""
        self.index.reset()

    def reset(self) -> None:
        """Clears the index."""
        self.index.reset()

    @classmethod
    def load_index(cls, path: str) -> faiss.Index:
        """Loads the index.

        Args:
            path (str): index file path.

        Returns:
            faiss.Index: the faiss index.
        """
        return faiss.read_index(path)

    def save_index(self, path: str) -> None:
        """Saves the index.

        Args:
            path (str): index file path to save.
        """
        return faiss.write_index(self.index, path)
