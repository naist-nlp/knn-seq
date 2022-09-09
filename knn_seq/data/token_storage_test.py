from itertools import chain
from typing import List

import numpy as np
import pytest

from knn_seq.data.token_storage import TokenStorage, make_offsets


def test_make_offsets():
    arr1 = np.array([1, 1, 1, 1, 1, 5, 4, 3, 2, 1])
    arr2 = np.array([0, 1, 2, 3, 4, 5, 10, 14, 17, 19, 20])
    assert np.array_equal(make_offsets(arr1), arr2)


def make_sentence():
    length = np.random.randint(10, 50 + 1)  # randint is [x, y)
    return np.random.randint(0, 100, (length,))


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
        tokens = [list(make_sentence()) for _ in range(10)]
        lengths = np.array([len(l) for l in tokens])
        sort_order = np.argsort(lengths, kind="mergesort")[::-1]
        lengths = lengths[sort_order]
        orig_tokens = tokens
        tokens = np.array(
            list(chain.from_iterable([tokens[s_i] for s_i in sort_order]))
        )
        return tokens, lengths, sort_order, orig_tokens

    def test__getitem__(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        for i in range(len(lengths)):
            begin, end = make_offsets(lengths)[i], make_offsets(lengths)[i + 1]
            assert np.array_equal(ts[i], tokens[begin:end])

    def test__len__(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        assert np.equal(len(ts), len(lengths))

    def test_size(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        assert np.equal(ts.size, len(tokens))

    def test_tokens(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        assert np.array_equal(ts.tokens, tokens)

    def test_sort_order(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        assert np.array_equal(ts.sort_order, sort_order)

    def test_orig_order(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        orig_order = np.zeros_like(sort_order)
        orig_order[sort_order] = np.arange(len(sort_order))
        assert np.array_equal(ts.orig_order, orig_order)

    def test_get_interval(self, data):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        for i in range(len(lengths)):
            ds_idx = ts.orig_order[i]
            assert np.array_equal(
                ts.get_interval(i),
                np.arange(
                    make_offsets(lengths)[ds_idx], make_offsets(lengths)[ds_idx + 1]
                ),
            )

    def test_binarize(self, data):
        tokens, lengths, sort_order, orig_tokens = data
        ts = TokenStorage(tokens, lengths, sort_order)
        ts_b = TokenStorage.binarize(orig_tokens)
        assert np.array_equal(ts.tokens, ts_b.tokens)
        assert np.array_equal(ts.lengths, ts_b.lengths)
        assert np.array_equal(ts.offsets, ts_b.offsets)
        assert np.array_equal(ts.sort_order, ts_b.sort_order)
        assert np.array_equal(ts.orig_order, ts_b.orig_order)
