import pytest

from . import fairseq_knn_model_base


class TestFairseqKNNModelBase:
    # @pytest.fixture
    # def init_models(self):
    #     """Load pre-trained model for test."""

    #     scriptdir = os.path.dirname(os.path.abspath(__file__))
    #     os.chdir(scriptdir)

    #     filenames = ["../../data/checkpoints/checkpoint_best.pt"]

    #     parser = options.get_generation_parser()
    #     args = options.parse_args_and_arch(parser, input_args=["../../data/data-bin"])
    #     cfg = convert_namespace_to_omegaconf(args)
    #     task = tasks.setup_task(cfg.task)

    #     models, saved_args = load_model_ensemble(filenames, task=task)
    #     self.knn_base = fairseq_knn_model_base.FairseqKNNModelBase(models)

    #     return saved_args

    def test_fit_embed_dim(self, init_models) -> None:
        """test for get_embed_dim()."""

        models, saved_args = init_models
        self.knn_base = fairseq_knn_model_base.FairseqKNNModelBase(models)

        test_case: int = saved_args["model"].decoder_embed_dim
        embed_dims = self.knn_base.get_embed_dim()
        assert embed_dims == [test_case]
