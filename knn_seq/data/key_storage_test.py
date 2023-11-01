import io

import h5py
import numpy as np
import pytest

from knn_seq.data.key_storage import KeyStorage


@pytest.fixture
def bio():
    bio = io.BytesIO()
    return bio


@pytest.fixture
def data(bio):
    arr = np.random.randn(5, 10).astype("float32")
    f = h5py.File(bio, mode="w")
    data = f.create_dataset("memory", data=arr)
    return data


@pytest.fixture
def create_tmpdata(tmp_path):
    arr = np.random.randn(5, 10).astype("float32")
    tmp_file = tmp_path / "tmp.h5"
    with h5py.File(tmp_file, mode="w") as f:
        f.create_dataset("memory", data=arr)
    return tmp_file


def test_key_storage__init__(bio, data):
    assert KeyStorage(bio, data)


def test_key_storage__len__(bio, data):
    d = KeyStorage(bio, data)
    assert len(d) == 5


def test_key_storage_size(bio, data):
    d = KeyStorage(bio, data)
    assert d.size == 5


def test_key_storage_dim(bio, data):
    d = KeyStorage(bio, data)
    assert d.dim == 10


def test_key_storage_dtype(bio, data):
    d = KeyStorage(bio, data)
    assert np.issubdtype(d.dtype, np.float32)


def test_key_storage___getitem__(bio, data):
    d = KeyStorage(bio, data)
    assert np.array_equal(d[0], data[0])


def test_key_storage_shape(bio, data):
    d = KeyStorage(bio, data)
    assert np.array_equal(d.shape, data.shape)


@pytest.mark.parametrize("readonly", [True, False])
def test_key_storage__open(create_tmpdata, readonly):
    if readonly:
        assert KeyStorage._open(create_tmpdata, readonly=readonly)
    else:
        create_bio = io.BytesIO()
        assert KeyStorage._open(create_bio, size=5, dim=10, readonly=readonly)


def test_key_storage_close(create_tmpdata):
    d = KeyStorage._open(create_tmpdata, readonly=True)
    d.close()
    assert not d._memory and not d.hdf5


@pytest.mark.parametrize("readonly", [True, False])
def test_key_storage_open(create_tmpdata, readonly):
    if readonly:
        assert KeyStorage.open(create_tmpdata, readonly=readonly)
    else:
        create_bio = io.BytesIO()
        assert KeyStorage._open(create_bio, size=5, dim=10, readonly=readonly)


def test_key_storage_add(bio, data):
    d = KeyStorage(bio, data)
    k = np.random.randn(2, 10).astype("float32")
    d.add(k)
    assert np.array_equal(d[1], k[1])
    assert d._write_pointer == 2


def test_key_storage_write_range(bio, data):
    d = KeyStorage(bio, data)
    k = np.random.randn(2, 10).astype("float32")
    d.write_range(k, 0, 2)
    assert np.array_equal(d[1], k[1])
