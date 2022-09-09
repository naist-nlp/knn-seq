import random
from itertools import chain
from typing import List

import h5py
import numpy as np
import pytest
import torch
from fairseq.data import Dictionary
from fairseq.data.language_pair_dataset import LanguagePairDataset

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


def dummy_dictionary(vocab_size=len(POPULATION), prefix="token_"):
    d = Dictionary()
    for i in range(vocab_size):
        token = prefix + str(i)
        d.add_symbol(token)
    d.finalize(padding_factor=1)  # don't add extra padding symbols
    return d


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
    print(ts_tokens)
    print(orig_list)
    return np.array_equal(orig_list, ts_tokens)


class TmpDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.sizes = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def fairseq_dataset(tmp_data):
    _, _, _, orig_data = tmp_data
    test_data = TmpDataset(orig_data)
    length = [len(l) for l in orig_data]
    dict = dummy_dictionary()
    dataset = LanguagePairDataset(
        test_data,
        length,
        dict,
        tgt=test_data,
        tgt_sizes=length,
        tgt_dict=dict,
    )
    return dataset, tmp_data


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

    @pytest.mark.parametrize("target", [True, False])
    def test_load_from_fairseq_dataset(self, data, target):
        dataset, (tokens, lengths, sort_order, orig_tokens) = fairseq_dataset(data)
        ts = TokenStorage(tokens, lengths, sort_order)
        ts_b = TokenStorage.load_from_fairseq_dataset(dataset, tgt=target)
        assert invest_match(orig_tokens, ts)
        assert invest_match(orig_tokens, ts_b)

    def test_save(self, data, tmp_path):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        ts.save(tmp_path)
        with h5py.File(tmp_path / "values.bin", mode="r") as f:
            assert np.array_equal(tokens, f["tokens"][()])
            assert np.array_equal(lengths, f["lengths"][()])
            assert np.array_equal(sort_order, f["sort_order"][()])

    def test_load(self, data, tmp_path):
        tokens, lengths, sort_order, _ = data
        ts = TokenStorage(tokens, lengths, sort_order)
        ts.save(tmp_path)
        ts_b = TokenStorage.load(tmp_path)
        assert np.array_equal(ts.tokens, ts_b.tokens)
        assert np.array_equal(ts.lengths, ts_b.lengths)
        assert np.array_equal(ts.offsets, ts_b.offsets)
        assert np.array_equal(ts.sort_order, ts_b.sort_order)
        assert np.array_equal(ts.orig_order, ts_b.orig_order)

    @pytest.fixture
    def create_path(self, tmp_path_factory):
        save_dir = tmp_path_factory.mktemp("save")
        other_dir = tmp_path_factory.mktemp("other")
        return save_dir, other_dir

    def test_merge(self, data, create_path):
        tokens, lengths, sort_order, orig_tokens = data
        save_dir, other_dir = create_path
        ts = TokenStorage(tokens, lengths, sort_order)
        ts.save(other_dir)
        ts.merge(save_dir, other_dir)
        with h5py.File(save_dir / "values.bin", mode="r") as f:
            assert np.array_equal(np.concatenate([tokens, tokens]), f["tokens"][()])
            assert np.array_equal(np.concatenate([lengths, lengths]), f["lengths"][()])
            assert np.array_equal(
                np.concatenate([sort_order, sort_order + len(sort_order)]),
                f["sort_order"][()],
            )
        ts_b = TokenStorage.load(save_dir)
        assert invest_match(orig_tokens * 2, ts_b)
