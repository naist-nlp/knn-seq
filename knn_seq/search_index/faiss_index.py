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
    """

    METRICS_MAP = {
        "l2": faiss.METRIC_L2,
        "ip": faiss.METRIC_INNER_PRODUCT,
        "cos": faiss.METRIC_INNER_PRODUCT,
    }

    def __len__(self) -> int:
        return self.index.ntotal

    def set_nprobe(self, nprobe: int):
        """Set nprobe parameter for IVF* indexes.

        Args:
            nprobe (int): Number of nearest neighbor clusters that are
                probed in search time.

        Raises:
            ValueError: When `nprobe` is smaller than 1.
        """
        if nprobe < 1:
            raise ValueError("`nprobe` must be greater than or equal to 1.")
        if self.use_ivf:
            if isinstance(self.index, faiss.IndexPreTransform):
                index = self.index.index
            else:
                index = self.index
            if isinstance(index, faiss.GpuIndexIVF):
                faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
            else:
                faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)

    def set_efsearch(self, efsearch: int):
        """Set efSearch parameter for HNSW indexes.

        Args:
            efsearch (int): The depth of exploration of the search.

        Raises:
            ValueError: When `nprobe` is smaller than 1.
        """
        if efsearch < 1:
            raise ValueError("`efsearch` must be greater than or equal to 1.")
        if self.use_hnsw:
            faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", efsearch)

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

    def to_gpu_train(self) -> None:
        """Transfers the faiss index to GPUs for training."""
        if self.use_pq:
            index = self.index.index if self.use_opq or self.use_pca else self.index
            pq = faiss.downcast_index(index).pq
            assign_index = faiss_index_to_gpu(faiss.IndexFlatL2(pq.dsub), num_gpus=1)
            pq.assign_index = assign_index
            if self.use_opq:
                opq = faiss.downcast_VectorTransform(self.index.chain.at(0))
                opq.pq = pq
            self.use_gpu = True

        ivf = self.ivf
        if ivf is not None:
            dim = (
                self.config.transform_dim if self.config.transform_dim > 0 else self.dim
            )
            clustering_index = faiss_index_to_gpu(
                faiss.IndexFlat(dim, self.METRICS_MAP[self.metric]), shard=True
            )
            ivf.clustering_index = clustering_index
            self.use_gpu = True

    def to_gpu_add(self) -> None:
        """Transfers the faiss index to GPUs for adding."""
        raise NotImplementedError(
            "Please use `FaissIndexFast` class to add vectors on GPUs."
        )

    def to_gpu_search(self) -> None:
        """Transfers the faiss index to GPUs for searching."""
        if self.use_ivf and self.use_hnsw:
            ivf_index: faiss.IndexIVF = self.ivf
            ivf_index.quantizer = faiss.downcast_index(
                faiss.downcast_index(ivf_index.quantizer).storage
            )
        self.index = faiss_index_to_gpu(self.index, shard=True)
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

        faiss.ParameterSpace().set_index_parameter(self.index, "verbose", verbose)
        self.index.train(vectors)
        faiss.ParameterSpace().set_index_parameter(self.index, "verbose", False)

    def add(self, vectors: ndarray, verbose: bool = False) -> None:
        """Adds vectors to the index.

        Args:
            vectors (ndarray): indexed vectors.
        """
        faiss.ParameterSpace().set_index_parameter(self.index, "verbose", verbose)

        vectors = self.normalize(vectors)
        return self.index.add(vectors)

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
