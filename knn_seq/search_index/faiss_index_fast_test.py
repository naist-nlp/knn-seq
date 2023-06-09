import faiss
import pytest
import torch

from knn_seq.search_index.faiss_index_fast import FaissIndexFast

D = 8
M = 4
nbit = 8
ksub = 2**nbit
N = ksub * M * 4
metric = "l2"


@pytest.fixture(scope="module")
def opq_index() -> FaissIndexFast:
    index = FaissIndexFast.new(metric, D, pq_subvec=M, use_opq=True)
    x = torch.rand(N, D)
    index.train(x)
    return index


class TestFaissIndexFast:
    def test___init__(self, opq_index: FaissIndexFast):
        assert opq_index.is_trained
        index = FaissIndexFast(opq_index.index, opq_index.config)
        assert index.is_trained
        assert index.A is not None and list(index.A.shape) == [D, D]
        assert index.b is not None and list(index.b.shape) == [0]
        vt: faiss.LinearTransform = faiss.downcast_VectorTransform(
            faiss.downcast_index(index.index).chain.at(0)
        )
        assert torch.equal(
            index.A,
            torch.from_numpy(faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in).T),
        )
        assert torch.equal(index.b, torch.from_numpy(faiss.vector_to_array(vt.b)))

    def test_rotate(self, opq_index: FaissIndexFast):
        x = torch.rand(N, D)

        # do nothing when A is None
        index = FaissIndexFast.new(metric, D)
        assert torch.equal(index.rotate(x), x)

        # rotate vectors
        opq_index = FaissIndexFast(opq_index.index, opq_index.config)
        expected = x @ opq_index.A
        r = opq_index.rotate(x)
        torch.testing.assert_close(r, expected)
        for shard_size in [1, 0, -1]:
            r = opq_index.rotate(x, shard_size=shard_size)
            torch.testing.assert_close(r, expected)
