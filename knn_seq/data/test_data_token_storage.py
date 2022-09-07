import pytest

import random
import numpy as np
from itertools import chain

from knn_seq.data.token_storage import (
    make_offsets, TokenStorage
)

def test_make_offsets():
    arr1=np.array(
        [1,1,1,1,1,5,4,3,2,1]
    )
    arr2=np.array(
        [0,1,2,3,4,5,10,14,17,19,20]
    )
    assert np.equal(make_offsets(arr1),arr2).all()
    


#########################################
##### Test class TokenStorage
#########################################
POPULATION = list(range(100))

def make_sentence():
    length = random.randint(10, 50)
    return random.choices(
        population=POPULATION, k=length, weights=range(1, len(POPULATION) + 1)
    )

@pytest.fixture
def data():
    tokens = [make_sentence() for _ in range(10)]
    lengths = np.array([len(l) for l in tokens])
    sort_order = np.argsort(lengths, kind="mergesort")[::-1]
    lengths = lengths[sort_order]
    tokens = np.array(
        list(chain.from_iterable([tokens[s_i]for s_i in sort_order]))
    )
    return tokens, lengths, sort_order

def test__init__(data):
   tokens, lengths, sort_order = data
   assert TokenStorage(tokens, lengths, sort_order)
    
def test__getitem__(data):
    tokens, lengths, sort_order = data
    ts = TokenStorage(tokens, lengths, sort_order)
    for i in range(len(lengths)):
        begin, end = make_offsets(lengths)[i], make_offsets(lengths)[i+1]
        assert np.equal(ts[i], tokens[begin:end]).all()

def test__len__(data):
    tokens, lengths, sort_order  = data
    ts = TokenStorage(tokens, lengths, sort_order)
    assert np.equal(len(ts), len(lengths))

def test_size(data):
    tokens, lengths, sort_order  = data
    ts = TokenStorage(tokens, lengths, sort_order)
    assert np.equal(ts.size, len(tokens))

def test_tokens(data):
    tokens, lengths, sort_order  = data
    ts = TokenStorage(tokens, lengths, sort_order)
    assert np.equal(ts.tokens, tokens).all()    

def test_sort_order(data):
    tokens, lengths, sort_order = data
    ts = TokenStorage(tokens, lengths, sort_order)
    assert np.equal(ts._sort_order, sort_order).all()

def test_orig_order(data):
    tokens, lengths, sort_order = data
    ts = TokenStorage(tokens, lengths, sort_order)
    orig_order = np.zeros_like(ts._sort_order)
    orig_order[ts._sort_order] = np.arange(len(ts._sort_order))
    assert np.equal(ts._orig_order, orig_order).all()
    
#def test_get_interval():
#    tokens, lengths, sort_order = data
#    ts = TokenStorage(tokens, lengths, sort_order)
#    for i in range(len(lengths)):
        
        
    
#def test_binarize():
#    
#def test_load_from_fairseq_dataset():
#    
#def test_save():
#    
#def test_load():
#
#def test_merge():
    
