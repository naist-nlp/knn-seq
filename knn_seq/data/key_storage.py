import contextlib
import logging
from typing import Optional, Union

import h5py
import numpy as np
from numpy.typing import DTypeLike, NDArray

logger = logging.getLogger(__name__)


class KeyStorage:
    """Key storage class."""

    def __init__(self, hdf5: h5py.File, memory: h5py.Dataset) -> None:
        self.hdf5 = hdf5
        self._memory = memory
        self._write_pointer = 0

    def __len__(self) -> int:
        """Returns the number of data."""
        return self._memory.shape[0]

    @property
    def size(self) -> int:
        """Returns the number of data."""
        return self._memory.shape[0]

    @property
    def dim(self) -> int:
        """Returns the dimension size."""
        return self._memory.shape[1]

    @property
    def dtype(self) -> DTypeLike:
        """Returns the dtype."""
        return self._memory.dtype

    @property
    def is_fp16(self) -> bool:
        """Returns whether the vectors are represented by fp16."""
        return np.issubdtype(self.dtype, np.float16)

    def __getitem__(self, indices: Union[int, slice, NDArray]) -> NDArray:
        return self._memory[indices]

    @property
    def shape(self):
        """Returns the shape of the storage."""
        return self._memory.shape

    @classmethod
    def _open(
        cls,
        path: str,
        size: Optional[int] = None,
        dim: Optional[int] = None,
        dtype: DTypeLike = np.float32,
        readonly: bool = True,
        compress: bool = False,
    ) -> "KeyStorage":
        """Opens the key storage.

        Args:
            path (str): path of the key storage.
            size (int): the number of data.
            dim (Optional[int]): dimension size.
            dtype (DtypeLike): numpy dtype. (default: np.float32)
            readonly (bool): open as read only.
            compress (bool): compress the memory.

        Returns:
            KeyStorage: this class.
        """
        f = h5py.File(path, mode="r" if readonly else "a")
        if readonly:
            memory = f["memory"]
        else:
            assert size is not None and dim is not None
            memory = f.create_dataset(
                "memory",
                shape=(size, dim),
                dtype=dtype,
                compression="gzip" if compress else None,
            )

        self = cls(f, memory)
        return self

    def close(self):
        """Closes the key storage stream."""
        self._memory.flush()
        self.hdf5.close()

    @classmethod
    @contextlib.contextmanager
    def open(
        cls,
        path: str,
        size: Optional[int] = None,
        dim: Optional[int] = None,
        dtype: DTypeLike = np.float32,
        readonly: bool = True,
        compress: bool = False,
    ):
        """Opens the key storage.

        Args:
            path (str): path of the key storage.
            size (int): the number of data.
            dim (Optional[int]): dimension size.
            dtype (DtypeLike): numpy dtype. (default: np.float32)
            readonly (bool): open as read only.
            compress (bool): compress the memory.
        """
        logger.info("Opens the key storage from {}".format(path))
        self = KeyStorage._open(
            path, size=size, dim=dim, dtype=dtype, readonly=readonly, compress=compress
        )
        logger.info("Number of datapoints: {:,}".format(len(self)))
        yield self
        self.close()

    def add(self, keys: NDArray) -> None:
        """Adds key vectors to the storage.

        Vectors are written continuously.

        Args:
            keys (NDArray): key vectors.
        """
        length = len(keys)
        assert self._write_pointer + length <= len(self)
        self._memory[self._write_pointer : self._write_pointer + length] = keys
        self._write_pointer += length

    def write_range(self, keys: NDArray, begin: int, end: int) -> None:
        """Writes key vectors to the storage.

        Args:
            keys (NDArray): key vectors.
            begin (int): the beginning position. (inclusive)
            end (int): the end position. (exclusive)
        """
        assert len(keys) == end - begin
        self._memory[begin:end] = keys
