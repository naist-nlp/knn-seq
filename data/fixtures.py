import os

import pytest
from fairseq import options, tasks
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks.translation import load_langpair_dataset

@pytest.fixture(scope="module")
def init_models():
    """Load pre-trained model for test."""

    filenames = ["data/checkpoints/checkpoint_best.pt"]

    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args=["data/data-bin"])
    cfg = convert_namespace_to_omegaconf(args)
    task = tasks.setup_task(cfg.task)

    ensemble, saved_args = load_model_ensemble(filenames, task=task)

    return ensemble, saved_args


@pytest.fixture(scope="module")
def testdata_src_dict():
    dict_path = os.path.join("data/data-bin", "dict.en.txt")
    return Dictionary.load(dict_path)


@pytest.fixture(scope="module")
def testdata_tgt_dict():
    dict_path = os.path.join("data/data-bin", "dict.ja.txt")
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
        "data/data-bin",
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
