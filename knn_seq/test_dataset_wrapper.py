from knn_seq.dataset_wrapper import LanguagePairDatasetWithRawSentence
import pytest
import os
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import Dictionary
from fairseq.data.indexed_dataset import get_available_dataset_impl
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
def testdata_samples(
    testdata_src_dict, testdata_tgt_dict, testdata_src_sents, testdata_tgt_sents
):
    samples = [
        {
            "id": i,
            "source": testdata_src_dict.encode_line(src),
            "target": testdata_tgt_dict.encode_line(tgt),
        }
        for i, (src, tgt) in enumerate(zip(testdata_src_sents[:10], testdata_tgt_sents[:10]))
    ]
    return samples


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

        with open(os.path.join("data", "train.en"), "r") as open_in:
            src_sents = [s.strip() for s in open_in.readlines()]

        with open(os.path.join("data", "train.ja"), "r") as open_in:
            tgt_sents = [s.strip() for s in open_in.readlines()]

        assert testdata_dataset.src_sents == src_sents
        assert testdata_dataset.tgt_sents == tgt_sents

    @pytest.mark.parametrize(
        "pad_to_length",
        [None, {"source": 1, "target": 1}, {"source": 30, "target": 30}],
    )
    def test_collater(self, testdata_dataset, testdata_samples, pad_to_length):
        collator = testdata_dataset.collater(
            testdata_samples, pad_to_length=pad_to_length
        )
        
        assert torch.equal(collator["id"], torch.tensor([6, 3, 2, 7, 5, 4, 1, 0, 9, 8]))
        assert collator["ntokens"] == 157

        if pad_to_length is None or pad_to_length["source"] < 27:
            expected_src = torch.tensor(
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
                ],
                dtype=torch.int32,
            )
        elif pad_to_length["source"] == 30:
            expected_src = torch.tensor(
                [
                    [  4, 105,  13,  26,  45,  11,  15,  26,   4,  19,  29,   9,  15,  57,
                        57,   7,   7,  43,  72,  13,  59,  45,  11,  15,  26,   6,   2, 1, 1, 1],
                    [ 12,   4,  10,   5,  19,  46,   4,  10,  25,  11,   9,  15,  64,   4,
                        45,  26,   4,  80,   4,  17,  11,   7,  14,  43,   6,   2,   1, 1, 1, 1],
                    [ 12,   7,   4,  21,   5,  18,  20,  10,   4,  13,   8,  25,   8,  10,
                        10,  85,  26,  90,  22,  16,  42,   6,   2,   1,   1,   1,   1, 1, 1, 1],
                    [ 12,   4,  19,  70,  14,   4,  10,  14,   8,  86,  31,   4,   5,   9,
                        24,  50,  59,  19,   6,   2,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 12,  73,   4,  21,   5,  18,  20,   7,  61,   4,  17,  27,  25,  14,
                        62,  46,   6,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  4,  74,  29,   4,  29,  32,  14,   7,  71,  16,  47,  73,   8,  72,
                        7,   6,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 12,  62,   7,  56,  47,   4, 110,  27,  85,  31,   9,  10,  91,  81,
                        6,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  4, 115,  49,  12,   4,   7,   5,   9,  34,  91,  22,   4,  30,   2,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 55,  36,  22,   4,  19,   5,  13,   7,  16,  36,   4,  30,   2,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  4, 109,  51,   4,  25,  29,  23,  10,   8,   6,   2,   1,   1,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1]
                ],
                dtype=torch.int32,
            )
        assert torch.equal(collator["net_input"]["src_tokens"], expected_src)
        assert torch.equal(
            collator["net_input"]["src_lengths"],
            torch.tensor([27, 26, 23, 20, 18, 17, 16, 14, 13, 11]),
        )

        if pad_to_length is None or pad_to_length["target"] < 27:
            expected_tgt = torch.tensor(
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
                ],
                dtype=torch.int32,
            )
            shifted_tgt = torch.tensor(
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
                ],
                dtype=torch.int32,
            )
        elif pad_to_length["target"] == 30:
            expected_tgt = torch.tensor(
                [
                    [ 50,  34,  49,  73,  12,  43,  18,   9,   7,  37,  36,   6,  53,  26,
                        29,   5,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  4,  17,   6,  11,  20,  11,  14,  62,  31, 111,  51,  34,  10,  24,
                        31,  19,   4,  30,  22,  14,   4,  21,   4,  75,   8,   5,   2, 1, 1, 1],
                    [  4,  55,   6,   4,  59,  31,   9,  42,  30,  72,   9,  19,   9,   7,
                        5,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 41,  20,   4, 106,  39,  13,  18,  30,  72,   9, 111,   5,   2,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 28,  56,  36,  11,  10,  68,   7,   4,  77,   4,  34,   5,   2,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  4,  55,   6,  57,   4,  21,  60,   4,  99,  39,  12,   6,   9,  54,
                        8,  14,  96,   5,   2,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  4,  17,   6,  37,  25,  41,  33,   4,  42,  34,  31,  60,  22,   5,
                        2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 28,  49,  47,  33,  62,  65,  26,  24,  13,  81,  12,  32,  15,   5,
                        2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 28,  49,  47,   6,  38,   4,  21,  24,   8,   7,  12,  32,  15,   5,
                        2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [ 35,  69,  58,  34,  84,   5,   2,   1,   1,   1,   1,   1,   1,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                ],
                dtype=torch.int32,
            )
            shifted_tgt = torch.tensor(
                [
                    [  2,  50,  34,  49,  73,  12,  43,  18,   9,   7,  37,  36,   6,  53,
                        26,  29,   5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,   4,  17,   6,  11,  20,  11,  14,  62,  31, 111,  51,  34,  10,
                        24,  31,  19,   4,  30,  22,  14,   4,  21,   4,  75,   8,   5, 1, 1, 1],
                    [  2,   4,  55,   6,   4,  59,  31,   9,  42,  30,  72,   9,  19,   9,
                        7,   5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,  41,  20,   4, 106,  39,  13,  18,  30,  72,   9, 111,   5,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,  28,  56,  36,  11,  10,  68,   7,   4,  77,   4,  34,   5,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,   4,  55,   6,  57,   4,  21,  60,   4,  99,  39,  12,   6,   9,
                        54,   8,  14,  96,   5,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,   4,  17,   6,  37,  25,  41,  33,   4,  42,  34,  31,  60,  22,
                        5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,  28,  49,  47,  33,  62,  65,  26,  24,  13,  81,  12,  32,  15,
                        5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,  28,  49,  47,   6,  38,   4,  21,  24,   8,   7,  12,  32,  15,
                        5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1],
                    [  2,  35,  69,  58,  34,  84,   5,   1,   1,   1,   1,   1,   1,   1,
                        1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 1, 1]
                ],
                dtype=torch.int32,
            )
        assert torch.equal(collator["target"], expected_tgt)
        assert torch.equal(collator["net_input"]["prev_output_tokens"], shifted_tgt)

        assert collator["src_sents"] == [
            '▁ A n y b o d y ▁ w ou l d ▁be ▁be t t er ▁tha n ▁no b o d y ▁.', 
            '▁I ▁ s a w ▁him ▁ s c o l d ed ▁ b y ▁ his ▁ m o t h er ▁.', 
            '▁I t ▁ & a pos ; s ▁ n e c e s s ar y ▁for ▁you ▁to ▁go ▁.', 
            '▁I ▁ w is h ▁ s h e ▁we re ▁ a l i ve ▁no w ▁.', 
            '▁I ▁don ▁ & a pos ; t ▁see ▁ m u c h ▁of ▁him ▁.', 
            '▁ Y ou ▁ ou g h t ▁not ▁to ▁have ▁don e ▁tha t ▁.', 
            '▁I ▁of t en ▁have ▁ q u ar re l s ▁with ▁her ▁.', 
            '▁ M ay ▁I ▁ t a l k ▁with ▁you ▁ ?', 
            '▁What ▁do ▁you ▁ w a n t ▁to ▁do ▁ ?', 
            '▁ O f ▁ c ou r s e ▁.'
        ]
        assert collator["tgt_sents"] == [
            '▁ど ん な ▁人 ▁で も ▁い ▁な い ▁よ り ▁は ▁ま し ▁だ ▁。', 
            '▁ 私 ▁は ▁彼 ▁が ▁彼 ▁の ▁お か あ さ ん ▁に ▁し か ら ▁ れ る ▁の ▁ を ▁ 見 ▁た ▁。', 
            '▁ 君 ▁は ▁ 行 か ▁な け れ ▁ば ▁な ら ▁な い ▁。', 
            '▁彼女 ▁が ▁ 生 き ▁て ▁い れ ▁ば ▁な あ ▁。', 
            '▁あ ま り ▁彼 ▁に ▁会 い ▁ ませ ▁ ん ▁。', 
            '▁ 君 ▁は ▁それ ▁ を ▁す ▁ べ き ▁で ▁は ▁な かっ ▁た ▁の に ▁。', 
            '▁ 私 ▁は ▁よ く ▁彼女 ▁と ▁ け ん か ▁す る ▁。', 
            '▁あ な た ▁と ▁お 話 し ▁し ▁て ▁いい ▁で す ▁か ▁。', 
            '▁あ な た ▁は ▁何 ▁ を ▁し ▁た い ▁で す ▁か ▁。', 
            '▁も ち ろ ん ▁さ ▁。'
        ]
