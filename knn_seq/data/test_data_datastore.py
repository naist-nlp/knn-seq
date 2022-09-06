import pytest

import io
import os
import tempfile
import h5py
import numpy as np

from knn_seq.data.datastore import Datastore

@pytest.fixture
def bio():
    bio = io.BytesIO()
    return bio

@pytest.fixture
def data(bio):
    arr = np.random.randn(5,10).astype("float32")
    f = h5py.File(bio, mode="w")
    data = f.create_dataset("memory", data=arr)
    return data

@pytest.fixture
def create_tmpdata():
    arr = np.random.randn(5,10).astype("float32")
    _, tf = tempfile.mkstemp()
    with h5py.File(tf, mode="w")as f:
        f.create_dataset("memory", data=arr)
    yield tf
    os.remove(tf)
        
def test_datastore__init__(bio, data):
    assert Datastore(bio, data)

def test_datastore__len__(bio, data):
    d = Datastore(bio, data)
    assert len(d)==5

def test_datastore_size(bio, data):
    d = Datastore(bio, data)
    assert d.size==5

def test_datastore_dim(bio, data):
    d = Datastore(bio, data)
    assert d.dim==10

def test_datastore_dtype(bio, data):
    d = Datastore(bio, data)
    assert np.equal(d.dtype, np.dtype("float32"))
    
def test_datastore___getitem__(bio, data):
    d = Datastore(bio, data)
    assert np.equal(d[0], data[0]).all()

def test_datastore_shape(bio, data):
    d = Datastore(bio, data)
    assert np.equal(d.shape, data.shape).all()

@pytest.mark.parametrize("readonly", [True, False])
def test_datastore__open(create_tmpdata, readonly):
    if readonly:
        assert Datastore._open(create_tmpdata, readonly=readonly)
    else:
        create_bio = io.BytesIO()
        assert Datastore._open(create_bio, size=5, dim=10, readonly=readonly)

    
def test_datastore_close(create_tmpdata):
    d = Datastore._open(create_tmpdata, readonly=True)
    d.close()
    assert not d._memory and not d.hdf5
    
    
@pytest.mark.parametrize("readonly", [True, False])
def test_datastore_open(create_tmpdata, readonly):
    if readonly:
        assert Datastore.open(create_tmpdata, readonly=readonly)
    else:
        create_bio = io.BytesIO()
        assert Datastore._open(create_bio, size=5, dim=10, readonly=readonly)
    
#@pytest.mark.parametrize("k_dim", [2, 8])
def test_datastore_add(bio, data):
    d = Datastore(bio, data)
    k = np.random.randn(2,10).astype("float32")
    d.add(k)
    assert np.equal(d[1],k[1]).all()
    assert d._write_pointer == 2
    

    
    



    
