from knn_seq.dataset_wrapper import (
    LanguagePairDatasetWithRawSentence,
    LanguagePairDatasetWithOriginalOrder,
)

import pytest
import os
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import Dictionary, data_utils
import numpy as np
import torch

DATABIN_DIR = os.path.join("data", "data-bin")


@pytest.fixture(scope="module")
def testdata_src_dict():
    dict_path = os.path.join(DATABIN_DIR, "dict.en.txt")
    return Dictionary.load(dict_path)


@pytest.fixture(scope="module")
def testdata_tgt_dict():
    dict_path = os.path.join(DATABIN_DIR, "dict.ja.txt")
    return Dictionary.load(dict_path)


@pytest.fixture(scope="module")
def testdata_src_sents():
    with open(os.path.join("data", "train.en"), "r") as open_in:
        src_sents = [s.strip() for s in open_in.readlines()]
    return src_sents


@pytest.fixture(scope="module")
def testdata_tgt_sents():
    with open(os.path.join("data", "train.ja"), "r") as open_in:
        tgt_sents = [s.strip() for s in open_in.readlines()]
    return tgt_sents


@pytest.fixture(scope="module")
def testdata_langpair_dataset(testdata_src_dict, testdata_tgt_dict):
    return load_langpair_dataset(
        DATABIN_DIR,
        "train",
        "en",
        testdata_src_dict,
        "ja",
        testdata_tgt_dict,
        combine=True,
        dataset_impl="mmap",
        upsample_primary=-1,
        left_pad_source=False,
        left_pad_target=False,
        max_source_positions=1024,
        max_target_positions=1024,
        load_alignments=False,
        truncate_source=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
    )

def pad(tensor_to_pad, pad_length, pad_value=1):
    num_to_add = pad_length - tensor_to_pad.shape[-1]
    return torch.nn.functional.pad(tensor_to_pad, (0, num_to_add), value=pad_value)

@pytest.fixture
def expected_src():
    return torch.LongTensor(
        [
            [  4, 105,  13,  26,  45,  11,  15,  26,   4,  19,  29,   9,  15,  57,
                57,   7,   7,  43,  72,  13,  59,  45,  11,  15,  26,   6,   2],
            [ 12,   4,  10,   5,  19,  46,   4,  10,  25,  11,   9,  15,  64,   4,
                45,  26,   4,  80,   4,  17,  11,   7,  14,  43,   6,   2,   1],
            [ 12,   7,   4,  21,   5,  18,  20,  10,   4,  13,   8,  25,   8,  10,
                10,  85,  26,  90,  22,  16,  42,   6,   2,   1,   1,   1,   1],
            [ 12,   4,  19,  70,  14,   4,  10,  14,   8,  86,  31,   4,   5,   9,
                24,  50,  59,  19,   6,   2,   1,   1,   1,   1,   1,   1,   1],
            [ 12,  73,   4,  21,   5,  18,  20,   7,  61,   4,  17,  27,  25,  14,
                62,  46,   6,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  4,  74,  29,   4,  29,  32,  14,   7,  71,  16,  47,  73,   8,  72,
                7,   6,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 12,  62,   7,  56,  47,   4, 110,  27,  85,  31,   9,  10,  91,  81,
                6,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  4, 115,  49,  12,   4,   7,   5,   9,  34,  91,  22,   4,  30,   2,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 55,  36,  22,   4,  19,   5,  13,   7,  16,  36,   4,  30,   2,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  4, 109,  51,   4,  25,  29,  23,  10,   8,   6,   2,   1,   1,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        ]
    )

@pytest.fixture
def expected_tgt():
    return torch.LongTensor(
        [
            [ 50,  34,  49,  73,  12,  43,  18,   9,   7,  37,  36,   6,  53,  26,
                29,   5,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  4,  17,   6,  11,  20,  11,  14,  62,  31, 111,  51,  34,  10,  24,
                31,  19,   4,  30,  22,  14,   4,  21,   4,  75,   8,   5,   2],
            [  4,  55,   6,   4,  59,  31,   9,  42,  30,  72,   9,  19,   9,   7,
                5,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 41,  20,   4, 106,  39,  13,  18,  30,  72,   9, 111,   5,   2,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 28,  56,  36,  11,  10,  68,   7,   4,  77,   4,  34,   5,   2,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  4,  55,   6,  57,   4,  21,  60,   4,  99,  39,  12,   6,   9,  54,
                8,  14,  96,   5,   2,   1,   1,   1,   1,   1,   1,   1,   1],
            [  4,  17,   6,  37,  25,  41,  33,   4,  42,  34,  31,  60,  22,   5,
                2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 28,  49,  47,  33,  62,  65,  26,  24,  13,  81,  12,  32,  15,   5,
                2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 28,  49,  47,   6,  38,   4,  21,  24,   8,   7,  12,  32,  15,   5,
                2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [ 35,  69,  58,  34,  84,   5,   2,   1,   1,   1,   1,   1,   1,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
        ]
    )

@pytest.fixture
def expected_prev_out():
    return torch.LongTensor(
        [
            [  2,  50,  34,  49,  73,  12,  43,  18,   9,   7,  37,  36,   6,  53,
                26,  29,   5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,   4,  17,   6,  11,  20,  11,  14,  62,  31, 111,  51,  34,  10,
                24,  31,  19,   4,  30,  22,  14,   4,  21,   4,  75,   8,   5],
            [  2,   4,  55,   6,   4,  59,  31,   9,  42,  30,  72,   9,  19,   9,
                7,   5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,  41,  20,   4, 106,  39,  13,  18,  30,  72,   9, 111,   5,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,  28,  56,  36,  11,  10,  68,   7,   4,  77,   4,  34,   5,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,   4,  55,   6,  57,   4,  21,  60,   4,  99,  39,  12,   6,   9,
                54,   8,  14,  96,   5,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,   4,  17,   6,  37,  25,  41,  33,   4,  42,  34,  31,  60,  22,
                5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,  28,  49,  47,  33,  62,  65,  26,  24,  13,  81,  12,  32,  15,
                5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,  28,  49,  47,   6,  38,   4,  21,  24,   8,   7,  12,  32,  15,
                5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
            [  2,  35,  69,  58,  34,  84,   5,   1,   1,   1,   1,   1,   1,   1,
                1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        ]
    )

class TestLanguagePairDatasetWithRawSentence:
    @pytest.fixture
    def testdata_dataset(
        self, testdata_langpair_dataset, testdata_src_sents, testdata_tgt_sents
    ):
        return LanguagePairDatasetWithRawSentence(
            testdata_langpair_dataset,
            src_sents=testdata_src_sents,
            tgt_sents=testdata_tgt_sents,
        )

    def test_dataset_init(self, testdata_dataset, testdata_src_dict, testdata_tgt_dict, testdata_src_sents, testdata_tgt_sents):
        assert testdata_dataset.src_dict == testdata_src_dict
        assert testdata_dataset.tgt_dict == testdata_tgt_dict
        assert np.array_equal(
            testdata_dataset.src_sizes[:10],
            np.array([14, 16, 23, 26, 17, 18, 27, 20, 11, 13]),
        )
        assert np.array_equal(
            testdata_dataset.tgt_sizes[:10],
            np.array([15, 15, 16, 27, 19, 13, 17, 13,  7, 15]),
        )

        assert testdata_dataset.src_sents == testdata_src_sents
        assert testdata_dataset.tgt_sents == testdata_tgt_sents

    @pytest.mark.parametrize(
        "pad_to_length",
        [None, {"source": 1, "target": 1}, {"source": 30, "target": 30}],
    )
    def test_collater(self, testdata_dataset, pad_to_length, testdata_src_sents, testdata_tgt_sents, expected_src, expected_tgt, expected_prev_out):
        collator = testdata_dataset.collater(
            [testdata_dataset[i] for i in range(10)], pad_to_length=pad_to_length
        )
        
        assert torch.equal(collator["id"], torch.tensor([6, 3, 2, 7, 5, 4, 1, 0, 9, 8]))
        assert collator["ntokens"] == 157

        if pad_to_length is None or pad_to_length["source"] < 27:
            assert torch.equal(collator["net_input"]["src_tokens"], expected_src)
        else:
            expected_src_padded = pad(expected_src, pad_to_length["source"], testdata_dataset.src_dict.pad())
            assert torch.equal(collator["net_input"]["src_tokens"], expected_src_padded)

        assert torch.equal(
            collator["net_input"]["src_lengths"],
            torch.tensor([27, 26, 23, 20, 18, 17, 16, 14, 13, 11]),
        )

        if pad_to_length is None or pad_to_length["target"] < 27:
            assert torch.equal(collator["target"], expected_tgt)
            assert torch.equal(collator["net_input"]["prev_output_tokens"], expected_prev_out)
        else:
            expected_tgt_padded = pad(expected_tgt, pad_to_length['target'], testdata_dataset.src_dict.pad())
            assert torch.equal(collator["target"], expected_tgt_padded)
            expected_prev_out_padded = pad(expected_prev_out, pad_to_length['target'], testdata_dataset.src_dict.pad())
            assert torch.equal(collator["net_input"]["prev_output_tokens"], expected_prev_out_padded)
            
        assert collator["src_sents"] == [testdata_src_sents[i.item()] for i in collator["id"]]
        assert collator["tgt_sents"] == [testdata_tgt_sents[i.item()] for i in collator["id"]]


class TestLanguagePairDatasetWithOriginalOrder:
    @pytest.fixture
    def testdata_dataset(self, testdata_langpair_dataset):
        return LanguagePairDatasetWithOriginalOrder(testdata_langpair_dataset)

    def test_dataset_init(self, testdata_dataset, testdata_src_dict, testdata_tgt_dict):
        assert testdata_dataset.src_dict == testdata_src_dict
        assert testdata_dataset.tgt_dict == testdata_tgt_dict
        assert np.array_equal(
            testdata_dataset.src_sizes[:10],
            np.array([14, 16, 23, 26, 17, 18, 27, 20, 11, 13]),
        )
        assert np.array_equal(
            testdata_dataset.tgt_sizes[:10],
            np.array([15, 15, 16, 27, 19, 13, 17, 13,  7, 15]),
        )

    @pytest.mark.parametrize(
        "pad_to_length",
        [None, {"source": 1, "target": 1}, {"source": 35, "target": 30}],
    )
    def test_collater(self, testdata_dataset, pad_to_length, expected_src, expected_tgt, expected_prev_out):
        collator = testdata_dataset.collater(
            [testdata_dataset[i] for i in range(10)], pad_to_length=pad_to_length
        )
        assert torch.equal(collator["id"], torch.tensor([6, 3, 2, 7, 5, 4, 1, 0, 9, 8]))
        assert collator["ntokens"] == 157

        if pad_to_length is None or pad_to_length["source"] < 27:
            assert torch.equal(collator["net_input"]["src_tokens"], expected_src)
        else:
            expected_src_padded = pad(expected_src, pad_to_length["source"], testdata_dataset.src_dict.pad())
            assert torch.equal(collator["net_input"]["src_tokens"], expected_src_padded)

        assert torch.equal(
            collator["net_input"]["src_lengths"],
            torch.tensor([27, 26, 23, 20, 18, 17, 16, 14, 13, 11]),
        )

        if pad_to_length is None or pad_to_length["target"] < 27:
            assert torch.equal(collator["target"], expected_tgt)
            assert torch.equal(collator["net_input"]["prev_output_tokens"], expected_prev_out)
        else:
            expected_tgt_padded = pad(expected_tgt, pad_to_length['target'], testdata_dataset.src_dict.pad())
            assert torch.equal(collator["target"], expected_tgt_padded)
            expected_prev_out_padded = pad(expected_prev_out, pad_to_length['target'], testdata_dataset.src_dict.pad())
            assert torch.equal(collator["net_input"]["prev_output_tokens"], expected_prev_out_padded)

        assert torch.equal(
            collator["orig_order"], torch.tensor([7, 6, 2, 1, 5, 4, 0, 3, 9, 8])
        )

        expected_collator = collate([testdata_dataset[i] for i in range(10)], testdata_dataset.src_dict, pad_to_length=pad_to_length)

        assert torch.equal(
            collator["net_input"]["src_tokens"].index_select(0, collator['orig_order']),
            expected_collator["net_input"]["src_tokens"]
        )
        assert torch.equal(
            collator["net_input"]["prev_output_tokens"].index_select(0, collator['orig_order']),
            expected_collator["net_input"]["prev_output_tokens"]
        )
        assert torch.equal(
            collator["target"].index_select(0, collator['orig_order']),
            expected_collator["target"]
        )
        
        testdata_dataset.ordered_indices()
        #test nesting: LanguagePairWithOriginalOrder(LanguagePairWithRawSentences(langpair_dataet))?

def collate(
    samples,
    src_dict,
    pad_to_length=None,
):
    if len(samples) == 0:
        return {}

    if pad_to_length == None:
        pad_to_length = {"source": None, "target": None}

    def merge(key, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            src_dict.pad(),
            src_dict.eos(),
            False,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    src_tokens = merge(
        "source",
        pad_to_length=pad_to_length["source"]
    )
    target = merge(
        "target",
        pad_to_length=pad_to_length["target"]
    )
    prev_output_tokens = merge(
        "target",
        move_eos_to_beginning=True,
        pad_to_length=pad_to_length["target"]
    )

    batch = {
        "net_input": {
            "src_tokens": src_tokens,
            "prev_output_tokens": prev_output_tokens,
        },
        "target": target,
    }

    return batch
