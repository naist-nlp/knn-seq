import pytest
import torch

from knn_seq.search_index.faiss_index_fast import FaissIndexFast

D = 8
M = 4
nbit = 8
ksub = 2**nbit
N = ksub * M * 4
metric = "l2"


class TestFaissIndexFast:
    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    def test_normalize(self, metric: str):
        index = FaissIndexFast.new(metric, D)
        n = 3
        x = torch.rand(n, D)

        if index.metric == "cos":
            torch.testing.assert_close(
                index.normalize(x), x / x.square().sum(-1, keepdim=True).sqrt()
            )
        else:
            assert torch.equal(index.normalize(x), x)

    def test_rotate(self):
        x = torch.rand(N, D)

        # do nothing
        index = FaissIndexFast.new(metric, D)
        assert torch.equal(index.rotate(x), x)

        # rotate vectors
        index = FaissIndexFast.new(metric, D, pq_subvec=M, use_opq=True)
        index.train(x)
        index = FaissIndexFast(index.index, index.config)
        expected = x @ index.A
        r = index.rotate(x)
        torch.testing.assert_close(r, expected)
        for shard_size in [1, 0, -1]:
            r = index.rotate(x, shard_size=shard_size)
            torch.testing.assert_close(r, expected)
