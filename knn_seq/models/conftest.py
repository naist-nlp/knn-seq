import pytest
from fairseq import options, tasks
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


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
