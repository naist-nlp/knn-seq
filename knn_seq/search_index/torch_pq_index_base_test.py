from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.torch_pq_index_base import TorchPQIndexBase

D = 8
M = 4
nbit = 8
ksub = 2**nbit
N = ksub * M * 4


class TorchPQMockIndex(TorchPQIndexBase):
    def query(self, querys: Tensor, k: int = 1) -> Tuple[Tensor, Tensor]:
        pass


@pytest.fixture(scope="module")
def pq_index() -> FaissIndex:
    np.random.seed(0)
    index = FaissIndex.new("l2", D, pq_subvec=M, use_opq=True)
    keys = np.random.rand(N, D).astype(np.float32)
    index.train(keys)
    index.add(keys)
    return index


class TestTorchPQIndexBase:
    def test___init__(self):
        # TODO(deguchi): add tests
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

    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    def test_normalize(self, pq_index: FaissIndex, metric: str):
        index = TorchPQMockIndex(pq_index)
        index.metric = metric
        x = torch.rand(N, D)

        if metric == "cos":
            assert torch.equal(
                index.normalize(x), x / x.square().sum(dim=-1, keepdim=True).sqrt()
            )
        else:
            assert torch.equal(index.normalize(x), x)

    def test_compute_distance(self, pq_index: FaissIndex):
        n = 3
        a = torch.rand(n, D)
        b = torch.rand(1, D)

        index = TorchPQMockIndex(pq_index)

        with pytest.raises(NotImplementedError):
            index.compute_distance(a, b, fn="dummy")

        assert torch.equal(
            index.compute_distance(a, b, fn=None),
            index.compute_distance(a, b, fn=index.metric),
        )

        expected_shape = [n]
        torch.testing.assert_close(
            index.compute_distance(a, b, fn="l2"), (a - b).square().sum(dim=-1)
        )
        assert list(index.compute_distance(a, b, fn="l2").shape) == expected_shape

        for metric in ["ip", "cos"]:
            torch.testing.assert_close(
                index.compute_distance(a, b, fn=metric), (a * b).sum(dim=-1)
            )
            assert list(index.compute_distance(a, b, fn=metric).shape) == expected_shape

    def test_compute_distance_table(self, pq_index: FaissIndex):
        bsz = 5
        n = 3
        m = 4
        a = torch.rand(bsz, n, D)
        b = torch.rand(bsz, m, D)

        index = TorchPQMockIndex(pq_index)

        with pytest.raises(NotImplementedError):
            index.compute_distance_table(a, b, fn="dummy")

        assert torch.equal(
            index.compute_distance_table(a, b, fn=None),
            index.compute_distance_table(a, b, fn=index.metric),
        )

        expected_shape = [bsz, n, m]
        torch.testing.assert_close(
            index.compute_distance_table(a, b, fn="l2"),
            (a[:, :, None] - b[:, None, :]).square().sum(dim=-1),
        )
        assert list(index.compute_distance_table(a, b, fn="l2").shape) == expected_shape

        for metric in ["ip", "cos"]:
            torch.testing.assert_close(
                index.compute_distance_table(a, b, fn=metric),
                torch.einsum("bnd,bmd->bnm", a, b),
            )
            assert (
                list(index.compute_distance_table(a, b, fn=metric).shape)
                == expected_shape
            )

    def test_is_trained(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)
        assert index.is_trained

    def test_pre_encode(self, pq_index: FaissIndex):
        index = TorchPQMockIndex(pq_index)

        A, b = index.A, index.b
        assert list(A.shape) == [D, D]
        assert b.numel() == 0 or list(A.shape) == [D]

        x = torch.rand(N, D)
        expected = torch.einsum("...i,oi->...o", x, A)
        if b.numel() > 0:
            expected += b
        assert torch.equal(index.pre_encode(x), expected)
