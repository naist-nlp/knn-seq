import itertools
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from knn_seq.search_index.search_index import SearchIndex, SearchIndexConfig


class TestSearchIndexConfig:
    def test_save(self, tmp_path):
        cfg = SearchIndexConfig()
        cfg_path = tmp_path / "test_index_config.json"
        cfg.save(cfg_path)
        with open(cfg_path, mode="r") as f:
            assert asdict(cfg) == json.load(f)

    def test_load(self, tmp_path):
        cfg = SearchIndexConfig()
        cfg_path = tmp_path / "test_index_config.json"
        with open(cfg_path, mode="w") as f:
            json.dump(asdict(cfg), f, indent=True)

        assert cfg == SearchIndexConfig.load(cfg_path)


N = 3
D = 8


class TestSearchIndex:
    class SearchIndexMock(SearchIndex):
        def __len__(self):
            pass

        @property
        def dim(self):
            pass

        @classmethod
        def new(cls):
            pass

        @property
        def is_trained(self):
            pass

        def train(self):
            pass

        def add(self):
            pass

        def query(self):
            pass

        def clear(self):
            pass

        @classmethod
        def load_index(cls):
            pass

        def save_index(self):
            pass

    @pytest.mark.parametrize(
        ("vectors"),
        [
            torch.rand(N, D, dtype=torch.float32),
            torch.rand(N, D, dtype=torch.float16),
            np.random.rand(N, D).astype(np.float32),
            np.random.rand(N, D).astype(np.float16),
        ],
    )
    def test_convert_to_numpy(self, vectors):
        index = TestSearchIndex.SearchIndexMock(object, SearchIndexConfig())
        ndarray = index.convert_to_numpy(vectors)
        assert isinstance(ndarray, np.ndarray) and np.issubdtype(
            ndarray.dtype, np.float32
        )
        assert np.array_equal(np.array(vectors).astype(np.float32), ndarray)

    @pytest.mark.parametrize(
        ("vectors", "metric"),
        itertools.product(
            [
                torch.rand(N, D, dtype=torch.float32),
                torch.rand(N, D, dtype=torch.float16),
                np.random.rand(N, D).astype(np.float32),
                np.random.rand(N, D).astype(np.float16),
            ],
            ["l2", "ip", "cos"],
        ),
    )
    def test_normalize(self, vectors, metric):
        index = TestSearchIndex.SearchIndexMock(
            object, SearchIndexConfig(metric=metric)
        )
        # Copy input vectors because faiss.normalize_L2() is in-place operation.
        inputs = np.array(vectors).astype(np.float32).copy()
        normalized_vectors = index.normalize(vectors)
        assert isinstance(normalized_vectors, np.ndarray) and np.issubdtype(
            normalized_vectors.dtype, np.float32
        )

        if metric == "cos":
            norms = np.linalg.norm(normalized_vectors, axis=-1)
            assert np.allclose(norms, np.ones_like(norms))
            assert np.allclose(
                inputs / np.linalg.norm(inputs, axis=-1, keepdims=True),
                normalized_vectors,
            )
        else:
            assert np.array_equal(inputs, normalized_vectors)
