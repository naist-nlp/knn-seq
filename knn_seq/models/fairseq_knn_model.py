from __future__ import annotations

import logging
from typing import List, Optional

from torch import Tensor

from knn_seq import utils
from knn_seq.data.token_storage import TokenStorage
from knn_seq.models.fairseq_knn_model_base import FairseqKNNModelBase
from knn_seq.search_index.search_index import SearchIndex

logger = logging.getLogger(__name__)


class FairseqKNNModel(FairseqKNNModelBase):
    """A wrapper for kNN-MT."""

    def set_index(
        self,
        val: TokenStorage,
        indexes: List[SearchIndex],
        knn_topk: int = 16,
        knn_temperature: float = 1.0,
        knn_threshold: Optional[float] = None,
        knn_weight: float = 0.5,
    ):
        self.val = val
        self.indexes = indexes
        self.knn_topk = knn_topk
        self.knn_temperature = knn_temperature

        self.knn_threshold = knn_threshold
        self.knn_weight = knn_weight

    def search(self, querys: Tensor, index_id: int = 0) -> FairseqKNNModel.KNNOutput:
        """Search k-nearest-neighbors.

        Args:
            querys (Tensor): A query tensor of shape `(batch_size, dim)`.
            index_id (int): Index ID for ensemble.

        Output:
            KNNOutput: A kNN output object.
        """
        scores, indices = self.indexes[index_id].search(
            querys, k=self.knn_topk, idmap=self.val.tokens
        )
        if self.knn_threshold is not None:
            scores[scores < self.knn_threshold] = float("-inf")
        probs = utils.softmax(scores / self.knn_temperature).type_as(querys)
        indices = indices.to(querys.device)
        return self.KNNOutput(scores, probs, indices)
