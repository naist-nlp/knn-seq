import random
from itertools import chain
from typing import List

import numpy as np
import pytest

from knn_seq.data.token_storage import TokenStorage, make_offsets


def test_make_offsets():
    arr1 = np.array([1, 1, 1, 1, 1, 5, 4, 3, 2, 1])
    arr2 = np.array([0, 1, 2, 3, 4, 5, 10, 14, 17, 19, 20])
    assert np.array_equal(make_offsets(arr1), arr2)


POPULATION = list(range(100))


def make_sentence():
    length = random.randint(10, 50)
    return random.choices(
        population=POPULATION, k=length, weights=range(1, len(POPULATION) + 1)
    )


def invest_match(orig_list: List[List[int]], ts: TokenStorage) -> bool:
    """Verify that the tokens of the instance can be restored to the original data.

    Args:
        orig_list (List): source tokens of sentences.
        ts (:class:`TokenStorage`): TokenStorage object.

    Returns:
        bool: whether `ts` can be restored to `orig_list`.
    """
    ts_tokens = []
    for i in ts.orig_order:
        ts_tokens.append(list(ts[i]))
    return np.array_equal(orig_list, ts_tokens)


class TestTokenStorage:
    @pytest.fixture
    def data(self):
        tokens = [make_sentence() for _ in range(10)]
        lengths = np.array([len(l) for l in tokens])
        sort_order = np.argsort(lengths, kind="mergesort")[::-1]
        lengths = lengths[sort_order]
        orig_tokens = tokens
        tokens = np.array(
            list(chain.from_iterable([tokens[s_i] for s_i in sort_order]))
        )
        return tokens, lengths, sort_order, orig_tokens
