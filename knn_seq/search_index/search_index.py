import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, List, Literal, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SearchIndexConfig:
    """Search index config dataclass."""

    backend: Literal["faiss"] = "faiss"
    metric: str = "l2"
    hnsw_edges: int = 0
    ivf_lists: int = 0
    pq_subvec: int = 0
    transform_dim: int = -1
    use_opq: bool = False
    use_pca: bool = False

    def save(self, path: str) -> None:
        with open(path, mode="w") as f:
            json.dump(asdict(self), f, indent=True)

    @classmethod
    def load(cls, path: str) -> "SearchIndexConfig":
        with open(path, mode="r") as f:
            config = json.load(f)
        return cls(**config)


class SearchIndex(ABC):
    """Search index class.

    Args:
        index (Any): search index.
        metric (str): distance function.
    """

    METRICS = {"l2", "ip", "cos"}

    def __init__(self, index: Any, config: SearchIndexConfig, **kwargs) -> None:
        self.index = index
        self.config = config
        self.backend = config.backend
        self.metric = config.metric
        self.is_ivf = config.ivf_lists > 0
        self.is_hnsw = config.hnsw_edges > 0
        self.is_pq = config.pq_subvec > 0
        self.is_opq = config.use_opq
        self.is_pca = config.use_pca

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of indexed data."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Returns the dimension size."""

    @classmethod
    @abstractmethod
    def new(cls, metric: str, dim: int, **kwargs) -> "SearchIndex":
        """Builds a new search index instance.

        Args:
            metric (str): distance function.
            dim (int): dimension size of vectors.
            kwargs: backend specific keyword arguments.

        Returns:
            SearchIndex: a new search index instance.
        """

    def convert_to_numpy(self, vectors: Union[Tensor, ndarray]) -> ndarray:
        """Convers a tensor or a numpy array to np.float32 arrary.

        Args:
            vectors (Union[Tensor, ndarray]): input vectors.

        Returns:
            ndarray: np.float32 array.
        """
        if torch.is_tensor(vectors):
            vectors = vectors.cpu().numpy()
        if not np.issubdtype(vectors.dtype, np.float32):
            vectors = np.array(vectors, dtype=np.float32)
        return vectors

    def normalize(self, vectors: Union[Tensor, ndarray]) -> ndarray:
        """Normalizes vectors.

        Converts a tensor or a numpy array to np.float32 array and normalizes vector.

        Args:
            vectors (Union[Tensor, ndarray]): input vectors.

        Returns:
            ndarray: normalzied np.float32 array.
        """
        vectors = self.convert_to_numpy(vectors)
        if self.metric == "cos":
            faiss.normalize_L2(vectors)
        return vectors

    def to_gpu(self, *args, **kwargs):
        """Transfers the faiss index to GPUs."""
        pass

    def to_cpu(self) -> None:
        """Transfers the faiss index to CPUs."""
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Returns the index is trained or not."""

    @abstractmethod
    def train(self, vectors) -> None:
        """Trains the index for some approximate nearest neighbor search algorithms.

        Args:
            vectors (ndarray): input vectors.
        """

    @abstractmethod
    def add(self, vectors: ndarray, ids: Optional[ndarray] = None) -> None:
        """Adds vectors to the index.

        Args:
            vectors (ndarray): indexed vectors.
            ids (Optional[ndarray]): indices of the index.
        """

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
            distances_tensor: torch.FloatTensor = distances_tensor.neg()

        return distances_tensor, indices_tensor

    @abstractmethod
    def query(self, querys: ndarray, k: int = 1) -> Tuple[ndarray, ndarray]:
        """Querys the k-nearest vectors to the index.

        Args:
            querys (ndarray): query vectors.
            k (int): number of nearest neighbors.

        Returns:
            - ndarray: top-k distances.
            - ndarray: top-k indices.
        """

    def search(
        self,
        querys: Union[Tensor, ndarray],
        k: int = 1,
        idmap: Optional[ndarray] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Searches the k-nearest vectors.

        Args:
            querys (Union[Tensor, ndarray]): query vectors.
            k (int): number of nearest neighbors.
            idmap (ndarray, optional): if given, maps the ids. (e.g., [3, 5] -> {0: 3, 1: 5})

        Returns:
            - FloatTensor: top-k scores.
            - LongTensor: top-k indices.
        """
        querys = self.normalize(querys)
        distances, indices = self.query(querys, k=k)
        return self.postprocess_search(distances, indices, idmap=idmap)

    @abstractmethod
    def clear(self) -> None:
        """Clears the index."""

    @classmethod
    def load(cls, path: str, **kawrgs) -> "SearchIndex":
        """Loads the index.

        Args:
            path (str): index file path.
            kwargs: other keyword arguments.

        Returns:
            SearchIndex: the index.
        """

        config = cls.load_config(path)
        index = cls.load_index(path)
        return cls(index, config)

    def save(self, path: str) -> None:
        """Saves the index.

        Args:
            path (str): index file path to save.
        """
        self.save_config(path)
        self.save_index(path)

    @classmethod
    def load_config(cls, path: str) -> SearchIndexConfig:
        """Loads the index configuration.

        Args:
            path (str): index file path.

        Returns:
            SearchIndexConfig: index configuration.
        """
        return SearchIndexConfig.load(os.path.splitext(path)[0] + ".json")

    def save_config(self, path: str) -> None:
        """Saves the index configuration.

        Args:
            path (str): index file path.
        """
        self.config.save(os.path.splitext(path)[0] + ".json")

    @classmethod
    @abstractmethod
    def load_index(cls, path: str) -> Any:
        """Loads the index.

        Args:
            path (str): index file path.

        Returns:
            Any: the wrapped index.
        """

    @abstractmethod
    def save_index(self, path: str) -> None:
        """Saves the index.

        Args:
            path (str): index file path to save.
        """
