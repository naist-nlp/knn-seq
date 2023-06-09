from typing import Tuple

import numpy as np
import pytest
import torch
from numpy import ndarray

from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.torch_pq_index_base import TorchPQIndexBase

D = 8
M = 4
nbit = 8
ksub = 2**nbit
N = ksub * M * 4


class TorchPQMockIndex(TorchPQIndexBase):
    def query(self, querys: ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        pass


@pytest.fixture(scope="module")
def pq_index() -> FaissIndex:
    np.random.seed(0)
    index = FaissIndex.new("l2", D, pq_subvec=M)
    keys = np.random.rand(N, D).astype(np.float32)
    index.train(keys)
    index.add(keys)
    return index


class TestTorchPQIndexBase:
    def test___init__(self):
        pass

    def test___len__(self, pq_index: FaissIndex):
        assert len(TorchPQMockIndex(pq_index)) == N

    def test_dim(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert index.dim == D

    def test_M(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert index.M == M

    def test_ksub(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert index.ksub == ksub

    def test_dsub(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert index.dsub == D // M

    def test_codewords(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert list(index.codewords.shape) == [M, ksub, D // M]

    def test_codes(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert index.codes.dtype == torch.uint8
        assert list(index.codes.shape) == [N, M]
