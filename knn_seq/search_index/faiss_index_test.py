import itertools

import faiss
import numpy as np
import pytest
import torch

from knn_seq.search_index.faiss_index import (
    FaissIndex,
    faiss_index_to_cpu,
    faiss_index_to_gpu,
)
from knn_seq.search_index.search_index import SearchIndexConfig

N = 3
D = 8


def is_gpu_index(index):
    concrete_index = faiss.downcast_index(index)
    if isinstance(concrete_index, (faiss.IndexReplicas, faiss.IndexShards)):
        return any(
            isinstance(faiss.downcast_index(index.at(i)), faiss.GpuIndex)
            for i in range(index.count())
        )
    return isinstance(concrete_index, faiss.GpuIndex)


def mkparams_index():
    for index in [
        faiss.IndexFlatL2(D),
        faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, 8),
        faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, 8, 4, 8),
    ]:
        yield index


def mkparams_index_and_config():
    for index_and_config in [
        (faiss.IndexFlatL2(D), SearchIndexConfig()),
        (
            faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, 8),
            SearchIndexConfig(ivf_lists=8),
        ),
        (
            faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, 8, 4, 8),
            SearchIndexConfig(ivf_lists=8, pq_subvec=4),
        ),
    ]:
        yield index_and_config


@pytest.mark.parametrize("index", mkparams_index())
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available.")
def test_faiss_index_to_gpu(index):
    gpu_index = faiss_index_to_gpu(index)
    assert is_gpu_index(gpu_index)

    # Signle GPU
    gpu_index = faiss_index_to_gpu(index, num_gpus=1)
    assert is_gpu_index(gpu_index)

    # Multi GPU (Replica)
    ngpus = min(faiss.get_num_gpus(), 2)
    gpu_index = faiss_index_to_gpu(index, num_gpus=ngpus)
    assert is_gpu_index(gpu_index)
    assert faiss.downcast_index(gpu_index).count() == ngpus

    # Multi GPU (Shard)
    gpu_index = faiss_index_to_gpu(index, num_gpus=ngpus, shard=True)
    assert is_gpu_index(gpu_index)
    assert faiss.downcast_index(gpu_index).count() == ngpus

    # Reserve fixed memory
    gpu_index = faiss_index_to_gpu(index, num_gpus=1, reserve_vecs=1024)
    assert is_gpu_index(gpu_index)

    # Pre-compute option
    gpu_index = faiss_index_to_gpu(index, num_gpus=1, precompute=True)
    assert is_gpu_index(gpu_index)


@pytest.mark.parametrize("index", mkparams_index())
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available.")
def test_faiss_index_to_cpu(index):
    # CPU -> CPU
    assert not is_gpu_index(faiss_index_to_cpu(index))

    # Single GPU -> CPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    assert not is_gpu_index(faiss_index_to_cpu(gpu_index))

    # Multi-GPU (Replica) -> CPU
    gpu_index = faiss.index_cpu_to_all_gpus(index, ngpu=min(faiss.get_num_gpus(), 2))
    assert not is_gpu_index(faiss_index_to_cpu(gpu_index))

    # Multi-GPU (Shard) -> CPU
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(
        index, co=co, ngpu=min(faiss.get_num_gpus(), 2)
    )
    assert not is_gpu_index(faiss_index_to_cpu(gpu_index))


class TestFaissIndex:
    @pytest.mark.parametrize("index", mkparams_index())
    def test___len__(self, index):
        faiss_index = FaissIndex(index, SearchIndexConfig())
        assert len(faiss_index) == 0

        index.train(np.zeros((256, D), dtype=np.float32))

        index.add(np.zeros((N, D), dtype=np.float32))
        assert len(faiss_index) == N
        index.add(np.zeros((N, D), dtype=np.float32))
        assert len(faiss_index) == 2 * N

    @pytest.mark.parametrize("index, config", mkparams_index_and_config())
    @pytest.mark.parametrize("nprobe", [-1, 0, 1, 8])
    def test_set_nprobe(
        self, index: faiss.Index, config: SearchIndexConfig, nprobe: int
    ):
        faiss_index = FaissIndex(index, config)
        if nprobe < 1:
            with pytest.raises(ValueError):
                faiss_index.set_nprobe(nprobe)
        else:
            faiss_index.set_nprobe(nprobe)
            if config.ivf_lists > 0:
                assert faiss.extract_index_ivf(index).nprobe == nprobe

    @pytest.mark.parametrize("index", mkparams_index())
    def test_dim(self, index):
        faiss_index = FaissIndex(index, SearchIndexConfig())
        assert faiss_index.dim == D

    @pytest.mark.parametrize("index", mkparams_index())
    def test_is_trained(self, index):
        faiss_index = FaissIndex(index, SearchIndexConfig())
        assert faiss_index.is_trained == index.is_trained
        if isinstance(index, faiss.IndexFlat):
            assert faiss_index.is_trained == True
        else:
            assert faiss_index.is_trained == False

        index.train(np.zeros((256, D), dtype=np.float32))
        assert faiss_index.is_trained == index.is_trained
        assert faiss_index.is_trained == True

    @pytest.mark.parametrize(
        ("idmap", "metric"),
        itertools.product(
            [
                (None, np.array([2, 0, 1])),
                (np.arange(3), np.array([2, 0, 1])),
                (np.array([1, 2, 0]), np.array([0, 1, 2])),
            ],
            ["l2", "ip", "cos"],
        ),
    )
    def test_postprocess_search(self, idmap, metric):
        index = FaissIndex.new(metric, dim=D)
        distances = np.random.rand(3, D)
        indices = np.array([2, 0, 1])
        mapping, expected_ids = idmap
        processed_distances, processed_indices = index.postprocess_search(
            distances, indices, idmap=mapping
        )
        assert np.array_equal(np.array(processed_indices), expected_ids)

        if metric == "l2":
            assert np.allclose(np.array(processed_distances), -distances)
        else:
            assert np.allclose(np.array(processed_distances), distances)

    @pytest.mark.parametrize("index", mkparams_index())
    def test_clear(self, index):
        faiss_index = FaissIndex(index, SearchIndexConfig())
        index.train(np.zeros((256, D), dtype=np.float32))
        index.add(np.zeros((N, D), dtype=np.float32))
        assert len(faiss_index) == N
        faiss_index.clear()
        assert len(faiss_index) == index.ntotal == 0

    @pytest.mark.parametrize("index", mkparams_index())
    def test_reset(self, index):
        faiss_index = FaissIndex(index, SearchIndexConfig())
        index.train(np.zeros((256, D), dtype=np.float32))
        index.add(np.zeros((N, D), dtype=np.float32))
        assert len(faiss_index) == N
        faiss_index.reset()
        assert len(faiss_index) == index.ntotal == 0

    @pytest.mark.parametrize("index", mkparams_index())
    def test_save_index(self, index, tmp_path):
        faiss_index = FaissIndex(index, SearchIndexConfig())
        index.train(np.zeros((256, D), dtype=np.float32))
        index.add(np.zeros((N, D), dtype=np.float32))

        index_path = str(tmp_path / "index.bin")
        faiss_index.save_index(index_path)

        assert faiss.read_index(index_path).ntotal == N

    @pytest.mark.parametrize("index", mkparams_index())
    def test_load_index(self, index, tmp_path):
        index.train(np.zeros((256, D), dtype=np.float32))
        index.add(np.zeros((N, D), dtype=np.float32))

        index_path = str(tmp_path / "index.bin")
        faiss.write_index(index, index_path)

        assert FaissIndex.load_index(index_path).ntotal == N
