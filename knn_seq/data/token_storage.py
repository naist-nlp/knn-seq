from __future__ import annotations

import logging
import os
from itertools import chain
from typing import Any, Iterable, List

import h5py
import numpy as np
from fairseq.data.language_pair_dataset import LanguagePairDataset
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_offsets(lengths: ArrayLike) -> NDArray:
    """Makes an offsets array from a lengths array.

    Args:
        lengths (NDArray): elements represent each data length.

    Returns:
        NDArray: offsets.
    """
    offsets = np.cumsum(lengths)
    offsets = np.insert(offsets, 0, 0)
    return offsets


class TokenStorage:
    """Token storage class.

    All tokens are stored to a flattened 1-D array and accessed by each sequence offset.
    Sequences are ordered in descending order of their length for efficient computation.
    """

    def __init__(self, tokens: NDArray, lengths: NDArray, sort_order: NDArray) -> None:
        self._tokens = tokens
        self.lengths = lengths
        self.offsets = make_offsets(lengths)
        self._sort_order = sort_order
        orig_order = np.zeros_like(self._sort_order)
        orig_order[self._sort_order] = np.arange(len(self._sort_order))
        self._orig_order = orig_order

    def __getitem__(self, idx: int) -> NDArray:
        begin, end = self.offsets[idx], self.offsets[idx + 1]
        return self._tokens[begin:end]

    def __len__(self) -> int:
        return self.lengths.size

    @property
    def size(self) -> int:
        """Returns the number of all tokens."""
        return self._tokens.size

    @property
    def tokens(self) -> NDArray:
        """Flattened tokens array.

        Each element maps an unique token index to a token ID. [sort_token_idx -> vocab]
        """
        return self._tokens

    @property
    def sort_order(self) -> NDArray:
        """Datastore order array.

        Each element maps a datastore index to a sequnece index. [sort_seq_idx -> orig_seq_idx]
        """
        return self._sort_order

    @property
    def orig_order(self) -> NDArray:
        """Original order array.

        Each element maps a sequence index to a datastore index. [orig_seq_idx -> sort_seq_idx]
        """
        return self._orig_order

    def get_interval(self, idx: int) -> NDArray:
        """Gets the datastore keys of the given sequence index.

        Args:
            idx (int): sequence index.

        Returns:
            NDArray: datastore keys.
        """
        ds_idx = self._orig_order[idx]
        return np.arange(self.offsets[ds_idx], self.offsets[ds_idx + 1])

    @classmethod
    def binarize(cls, raw_seq: List[List[int]]) -> "TokenStorage":
        lengths = np.array([len(l) for l in raw_seq])
        sort_order = np.argsort(lengths, kind="mergesort")[::-1]
        lengths = lengths[sort_order]
        tokens = np.array(
            list(chain.from_iterable([raw_seq[sort_idx] for sort_idx in sort_order]))
        )
        return cls(tokens, lengths, sort_order)

    @classmethod
    def load_from_fairseq_dataset(
        cls, dataset: LanguagePairDataset, tgt: bool = True, progress: bool = True
    ) -> "TokenStorage":
        """Load from fairseq :class:`LanguagePairDataset`.

        Args:
            dataset (:class:`LanguagePairDataset`): A binarized dataset.
            tgt (bool): If True, load the target side data, otherwise the source side. (default: True)
            progress (bool): Show the progress bar when loading the dataset. (default: True)
        """

        def progress_bar(iterable: Iterable[Any]):
            if progress:
                return tqdm(iterable)
            return iterable

        dataset.shuffle = False
        dataset.buckets = None
        if dataset.supports_prefetch:
            dataset.prefetch(range(len(dataset)))

        sizes = dataset.tgt_sizes if tgt else dataset.src_sizes
        assert sizes is not None
        data = dataset.tgt if tgt else dataset.src

        sort_order = dataset.ordered_indices()
        lengths = sizes[sort_order]
        tokens = np.concatenate([data[i] for i in progress_bar(sort_order)])
        return cls(tokens, lengths, sort_order)

    def save(self, save_dir: str) -> None:
        """Saves the binarized :class:`TokenStorage` to the given directory.

        Args:
            save_dir (str): directory path.
        """
        path = os.path.join(save_dir, "values.bin")
        logger.info(f"Saving the binarized sequences to `{path}'")
        with h5py.File(path, mode="w") as f:
            f.create_dataset("tokens", data=self._tokens, dtype=np.int32)
            f.create_dataset("lengths", data=self.lengths, dtype=np.int32)
            f.create_dataset("sort_order", data=self._sort_order)
        logger.info("Done")

    @classmethod
    def load(cls, save_dir: str) -> "TokenStorage":
        """Loads the binarized :class:`TokenStorage` from the given directory.

        Args:
            save_dir (str): directory path.

        Returns:
            IndexedSequences: this class.
        """
        path = os.path.join(save_dir, "values.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        logger.info(f"Loading `{path}'")
        with h5py.File(path, mode="r") as f:
            tokens = f["tokens"][()]
            lengths = f["lengths"][()]
            sort_order = f["sort_order"][()]
        self = cls(tokens, lengths, sort_order)
        logger.info("Loaded")
        return self

    def merge(self, save_dir: str, other_dir: str):
        other = TokenStorage.load(other_dir)
        new_tokens = np.concatenate([self._tokens, other._tokens])
        new_lengths = np.concatenate([self.lengths, other.lengths])
        new_sort_order = np.concatenate(
            [self._sort_order, other._sort_order + len(self._sort_order)]
        )
        self._tokens = new_tokens
        self.lengths = new_lengths
        self._sort_order = new_sort_order
        self.save(save_dir)
