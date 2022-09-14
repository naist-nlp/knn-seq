from knn_seq.dataset_wrapper import LanguagePairDatasetWithRawSentence
import pytest
import os
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import Dictionary
from fairseq.data.indexed_dataset import get_available_dataset_impl
import numpy as np
import torch


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data",
)

DATABIN_DIR = os.path.join(DATA_DIR, "data-bin")


class TestLanguagePairDatasetWithRawSentence:
    @pytest.fixture
    def tmp_src_dict(self):
        dict_path = os.path.join(DATABIN_DIR, "dict.en.txt")
        return Dictionary.load(dict_path)

    @pytest.fixture
    def tmp_tgt_dict(self):
        dict_path = os.path.join(DATABIN_DIR, "dict.ja.txt")
        return Dictionary.load(dict_path)

    @pytest.fixture
    def tmp_dataset(self, tmp_src_dict, tmp_tgt_dict):
        with open(os.path.join(DATA_DIR, "train.en"), "r") as open_in:
            src_sents = [s.strip() for s in open_in.readlines()]

        with open(os.path.join(DATA_DIR, "train.ja"), "r") as open_in:
            tgt_sents = [s.strip() for s in open_in.readlines()]

        dataset = load_langpair_dataset(
            DATABIN_DIR,
            "train",
            "en",
            tmp_src_dict,
            "ja",
            tmp_tgt_dict,
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

        assert language_pair_dataset.src_dict == tmp_src_dict
        assert language_pair_dataset.tgt_dict == tmp_tgt_dict
        assert np.array_equal(
            language_pair_dataset.src_sizes[:10],
            np.array([13, 10, 8, 17, 20, 16, 20, 15, 17, 9]),
        )
        assert np.array_equal(
            language_pair_dataset.tgt_sizes[:10],
            np.array([9, 6, 8, 7, 9, 9, 8, 5, 6, 8]),
        )
        assert language_pair_dataset.src_sents == src_sents
        assert language_pair_dataset.tgt_sents == tgt_sents
        return language_pair_dataset

    @pytest.fixture
    def tmp_samples(self, tmp_dataset, tmp_src_dict, tmp_tgt_dict):
        with open(os.path.join(DATA_DIR, "train.en"), "r") as open_in:
            src_sents = [
                tmp_src_dict.encode_line(s.strip()) for s in open_in.readlines()
            ]
        assert torch.equal(
            src_sents[0],
            tmp_src_dict.encode_line("Please respond ."),  # [130, 131, 132, 121,   2]
        )
        assert torch.equal(
            src_sents[9],
            tmp_src_dict.encode_line("He is kind ."),  # [155,  26, 156, 121,   2]
        )

        with open(os.path.join(DATA_DIR, "train.ja"), "r") as open_in:
            tgt_sents = [
                tmp_tgt_dict.encode_line(s.strip()) for s in open_in.readlines()
            ]
        assert torch.equal(
            tgt_sents[0],
            tmp_tgt_dict.encode_line("お 返事 ください 。"),  # [170, 928, 929, 930,   2]
        )
        assert torch.equal(
            tgt_sents[9],
            tmp_tgt_dict.encode_line("優しい 人 です 。"),  # [947, 332, 948, 930,   2]
        )

        samples = []
        for i in range(10):
            samples.append({"id": i, "source": src_sents[i], "target": tgt_sents[i]})
        assert len(samples) == 10
        return samples

    @pytest.mark.parametrize(
        ("pad_to_length"),
        [None, {"source": 1, "target": 1}, {"source": 10, "target": 5}],
    )  # , {'source' : 10, 'target': 10}])
    def test_collater(self, tmp_dataset, tmp_samples, pad_to_length):

        collator = tmp_dataset.collater(tmp_samples, pad_to_length=None)
        assert torch.equal(collator["id"], torch.tensor([4, 6, 8, 5, 7, 1, 2, 3, 9, 0]))
        assert collator["ntokens"] == sum([3, 4, 4, 4, 8, 5, 6, 5, 6, 4])
        assert torch.equal(
            collator["net_input"]["src_tokens"],
            torch.tensor(
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
            ),
        )
        assert torch.equal(
            collator["net_input"]["src_lengths"],
            torch.tensor([9, 7, 7, 6, 6, 5, 5, 5, 5, 4]),
        )
        assert torch.equal(
            collator["target"],
            torch.tensor(
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
            ),
        )
        assert torch.equal(
            collator["net_input"]["prev_output_tokens"],
            torch.tensor(
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
            ),  # Right shifted by 1 tgt tokens
        )
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
        