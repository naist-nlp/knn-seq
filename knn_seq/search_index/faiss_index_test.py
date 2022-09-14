import itertools

import numpy as np
import pytest

from knn_seq.search_index.faiss_index import FaissIndex

N = 3
D = 8


class TestFaissIndex:
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
