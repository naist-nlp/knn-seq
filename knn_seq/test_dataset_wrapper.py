from knn_seq.dataset_wrapper import LanguagePairDatasetWithRawSentence
import pytest
import os
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import Dictionary
from fairseq.data.indexed_dataset import get_available_dataset_impl
import numpy as np
import torch


DATABIN_DIR = os.path.join('data', "data-bin")

@pytest.fixture(scope="module")
def testdata_src_dict():
    dict_path = os.path.join(DATABIN_DIR, "dict.en.txt")
    return Dictionary.load(dict_path)


@pytest.fixture(scope="module")
def testdata_tgt_dict():
    dict_path = os.path.join(DATABIN_DIR, "dict.ja.txt")
    return Dictionary.load(dict_path)

@pytest.fixture(scope="module")
def testdata_samples(testdata_src_dict, testdata_tgt_dict):
    with open(os.path.join('data', "train.en"), "r") as open_in:
        src_sents = [
            testdata_src_dict.encode_line(s.strip()) for s in open_in.readlines()
        ]

    with open(os.path.join('data', "train.ja"), "r") as open_in:
        tgt_sents = [
            testdata_tgt_dict.encode_line(s.strip()) for s in open_in.readlines()
        ]

    samples = []
    for i in range(10):
        samples.append({"id": i, "source": src_sents[i], "target": tgt_sents[i]})
    return samples

class TestLanguagePairDatasetWithRawSentence:
    @pytest.fixture
    def testdata_dataset(self, testdata_src_dict, testdata_tgt_dict):
        with open(os.path.join('data', "train.en"), "r") as open_in:
            src_sents = [s.strip() for s in open_in.readlines()]

        with open(os.path.join('data', "train.ja"), "r") as open_in:
            tgt_sents = [s.strip() for s in open_in.readlines()]

        dataset = load_langpair_dataset(
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
        language_pair_dataset = LanguagePairDatasetWithRawSentence(
            dataset, src_sents=src_sents, tgt_sents=tgt_sents
        )
        return language_pair_dataset

    def test_dataset_init(self, testdata_dataset, testdata_src_dict, testdata_tgt_dict):
        assert testdata_dataset.src_dict == testdata_src_dict
        assert testdata_dataset.tgt_dict == testdata_tgt_dict
        assert np.array_equal(
            testdata_dataset.src_sizes[:10],
            np.array([13, 10, 8, 17, 20, 16, 20, 15, 17, 9]),
        )
        assert np.array_equal(
            testdata_dataset.tgt_sizes[:10],
            np.array([9, 6, 8, 7, 9, 9, 8, 5, 6, 8]),
        )
        
        with open(os.path.join('data', "train.en"), "r") as open_in:
            src_sents = [s.strip() for s in open_in.readlines()]

        with open(os.path.join('data', "train.ja"), "r") as open_in:
            tgt_sents = [s.strip() for s in open_in.readlines()]

        assert testdata_dataset.src_sents == src_sents
        assert testdata_dataset.tgt_sents == tgt_sents

    @pytest.mark.parametrize(
        ("pad_to_length"),
        [
            None, 
            {"source": 1, "target": 1}, 
            {"source": 10, "target": 6}
        ],
    ) 
    def test_collater(self, testdata_dataset, testdata_samples, pad_to_length):
        collator = testdata_dataset.collater(testdata_samples, pad_to_length=pad_to_length)
        assert torch.equal(collator["id"], torch.tensor([4, 6, 8, 5, 7, 1, 2, 3, 9, 0]))
        assert collator["ntokens"] == sum([3, 4, 4, 4, 8, 5, 6, 5, 6, 4])
        
        if not pad_to_length or pad_to_length['source'] < 10:
            expected_src = torch.tensor(
                [
                    [138, 26, 139, 140, 141, 140, 142, 121, 2],
                    [138, 145, 33, 146, 147, 121, 2, 1, 1],
                    [151, 152, 33, 153, 154, 121, 2, 1, 1],
                    [128, 143, 131, 144, 121, 2, 1, 1, 1],
                    [148, 149, 150, 144, 121, 2, 1, 1, 1],
                    [130, 131, 132, 121, 2, 1, 1, 1, 1],
                    [133, 10, 134, 135, 2, 1, 1, 1, 1],
                    [136, 26, 137, 121, 2, 1, 1, 1, 1],
                    [155, 26, 156, 121, 2, 1, 1, 1, 1],
                    [128, 129, 121, 2, 1, 1, 1, 1, 1],
                ],
                dtype=torch.int32,
            )
        elif pad_to_length['source'] == 10:
            expected_src = torch.tensor(
                [
                    [138, 26, 139, 140, 141, 140, 142, 121, 2, 1],
                    [138, 145, 33, 146, 147, 121, 2, 1, 1, 1],
                    [151, 152, 33, 153, 154, 121, 2, 1, 1, 1],
                    [128, 143, 131, 144, 121, 2, 1, 1, 1, 1],
                    [148, 149, 150, 144, 121, 2, 1, 1, 1, 1],
                    [130, 131, 132, 121, 2, 1, 1, 1, 1, 1],
                    [133, 10, 134, 135, 2, 1, 1, 1, 1, 1],
                    [136, 26, 137, 121, 2, 1, 1, 1, 1, 1],
                    [155, 26, 156, 121, 2, 1, 1, 1, 1, 1],
                    [128, 129, 121, 2, 1, 1, 1, 1, 1, 1],
                ],
                dtype=torch.int32,
            )
        assert torch.equal(collator["net_input"]["src_tokens"], expected_src)
        assert torch.equal(
            collator["net_input"]["src_lengths"],
            torch.tensor([9, 7, 7, 6, 6, 5, 5, 5, 5, 4]),
        )

        if not pad_to_length or pad_to_length['target'] < 5:
            expected_tgt = torch.tensor(
                [
                    [937, 938, 939, 930, 2],
                    [941, 942, 943, 930, 2],
                    [332, 51, 946, 930, 2],
                    [940, 47, 929, 930, 2],
                    [944, 945, 930, 2, 1],
                    [931, 932, 315, 930, 2],
                    [933, 934, 26, 930, 2],
                    [935, 173, 936, 930, 2],
                    [947, 332, 948, 930, 2],
                    [170, 928, 929, 930, 2],
                ],
                dtype=torch.int32,
            )
            shifted_tgt = torch.tensor(
                [
                    [2, 937, 938, 939, 930],
                    [2, 941, 942, 943, 930],
                    [2, 332, 51, 946, 930],
                    [2, 940, 47, 929, 930],
                    [2, 944, 945, 930, 1],
                    [2, 931, 932, 315, 930],
                    [2, 933, 934, 26, 930],
                    [2, 935, 173, 936, 930],
                    [2, 947, 332, 948, 930],
                    [2, 170, 928, 929, 930],
                ],
                dtype=torch.int32,
            )
        elif pad_to_length['target'] == 6:
            expected_tgt = torch.tensor(
                [
                    [937, 938, 939, 930, 2, 1],
                    [941, 942, 943, 930, 2, 1],
                    [332, 51, 946, 930, 2, 1],
                    [940, 47, 929, 930, 2, 1],
                    [944, 945, 930, 2, 1, 1],
                    [931, 932, 315, 930, 2, 1],
                    [933, 934, 26, 930, 2, 1],
                    [935, 173, 936, 930, 2, 1],
                    [947, 332, 948, 930, 2, 1],
                    [170, 928, 929, 930, 2, 1],
                ],
                dtype=torch.int32,
            )
            shifted_tgt = torch.tensor(
                [
                    [2, 937, 938, 939, 930, 1],
                    [2, 941, 942, 943, 930, 1],
                    [2, 332, 51, 946, 930, 1],
                    [2, 940, 47, 929, 930, 1],
                    [2, 944, 945, 930, 1, 1],
                    [2, 931, 932, 315, 930, 1],
                    [2, 933, 934, 26, 930, 1],
                    [2, 935, 173, 936, 930, 1],
                    [2, 947, 332, 948, 930, 1],
                    [2, 170, 928, 929, 930, 1],
                ],
                dtype=torch.int32,
            )
        assert torch.equal(collator["target"], expected_tgt)
        assert torch.equal(collator["net_input"]["prev_output_tokens"], shifted_tgt)

        assert collator["src_sents"] == [
            "It is still as cold as ever .",
            "It rains in some places .",
            "Every man in his way .",
            "Please let me know .",
            "I don &apos;t know .",
            "Let me say .",
            "What a pity !",
            "Forewarned is forearmed .",
            "He is kind .",
            "Please respond .",
        ]
        assert collator["tgt_sents"] == [
            "相変わらず まだ 寒い 。",
            "ところ により 雨 。",
            "人 に 一癖 。",
            "知らせ て ください 。",
            "知ら ない 。",
            "一言 言い たい 。",
            "全く 気の毒 だ 。",
            "警戒 は 警備 。",
            "優しい 人 です 。",
            "お 返事 ください 。",
        ]
