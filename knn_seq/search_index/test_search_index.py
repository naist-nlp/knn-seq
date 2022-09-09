import itertools
import json
import os
from dataclasses import asdict
from typing import Tuple

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

        def query(
            self, querys: np.ndarray, k: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            if self.metric == "l2":
                return np.array([0.1, 0.2, 0.3]), np.array([2, 0, 1])
            return np.array([0.3, 0.2, 0.1]), np.array([2, 0, 1])

        def clear(self):
            pass

        @classmethod
        def load_index(cls, path):
            pass

        def save_index(self, path):
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
        index = TestSearchIndex.SearchIndexMock(
            object, SearchIndexConfig(metric=metric)
        )
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
    def test_search(self, idmap, metric):
        index = TestSearchIndex.SearchIndexMock(
            object, SearchIndexConfig(metric=metric)
        )
        querys = np.random.rand(3, D)
        mapping, expected_ids = idmap
        distances, indices = index.search(querys, k=3, idmap=mapping)
        assert np.array_equal(np.array(indices), expected_ids)

    def test_save_config(self, tmp_path):
        index_path = tmp_path / "test_index.bin"
        index = TestSearchIndex.SearchIndexMock(object, SearchIndexConfig())
        index.save_config(index_path)
        assert os.path.exists(os.path.splitext(index_path)[0] + ".json")

    def test_load_config(self, tmp_path):
        index_path = tmp_path / "test_index.bin"
        cfg = SearchIndexConfig()
        cfg.save(os.path.splitext(index_path)[0] + ".json")
        assert cfg == TestSearchIndex.SearchIndexMock.load_config(index_path)

    def test_save(self, tmp_path):
        index_path = tmp_path / "test_index.bin"
        index = TestSearchIndex.SearchIndexMock(object, SearchIndexConfig())
        index.save(index_path)
        assert os.path.exists(os.path.splitext(index_path)[0] + ".json")

    def test_load(self, tmp_path):
        index_path = tmp_path / "test_index.bin"
        cfg = SearchIndexConfig()
        cfg.save(os.path.splitext(index_path)[0] + ".json")
        assert cfg == TestSearchIndex.SearchIndexMock.load(index_path).config
