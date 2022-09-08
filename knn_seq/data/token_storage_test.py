import numpy as np
import pytest

from knn_seq.data.token_storage import TokenStorage, make_offsets


def test_make_offsets():
    arr1 = np.array([1, 1, 1, 1, 1, 5, 4, 3, 2, 1])
    arr2 = np.array([0, 1, 2, 3, 4, 5, 10, 14, 17, 19, 20])
    assert np.array_equal(make_offsets(arr1), arr2)
