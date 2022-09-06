import pytest

import string
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
    
POPULATION = string.ascii_letters + string.digits

def make_sentence():
    length = random.randint(10, 50)
    return random.choices(
        population=POPULATION, k=length, weights=range(1, len(POPULATION) + 1)
    )


@pytest.fixture
def pre_tokens():
    data = (
        [make_sentence() for _ in range(10)]
    )
    return data

@pytest.fixture
def sort_order(pre_tokens):
    lengths = np.array([len(l) for l in pre_tokens])
    sort_order = np.argsort(lengths, kind="stable")[::-1]
    return sort_order

@pytest.fixture
def lengths(pre_tokens, sort_order):
    lengths = np.array([len(l) for l in pre_tokens])
    lengths = lengths[sort_order]
    return lengths

@pytest.fixture
def tokens(pre_tokens, sort_order):
    tokens = np.array(
        list(chain.from_iterable(pre_tokens[s_i] for s_i in sort_order))
    )
    return tokens
class TestTokenStorage:

    
    def test__init__(tokens, lengths, sort_order):
        assert TokenStorage(tokens, lengths, sort_order)
        
    #def test__getitem__():
    #    
    #def test__len__():
    #    
    #def test_size():
    #
    #def test_tokens():
    #
    #def test_sort_order():
    #
    #def test_orig_order():
    #    
    #def test_get_interval():
    #    
    #def test_binarize():
    #    
    #def test_load_from_fairseq_dataset():
    #    
    #def test_save():
    #    
    #def test_load():
    #
    #def test_merge():
    #    
    #