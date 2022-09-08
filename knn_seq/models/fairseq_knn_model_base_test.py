from . import fairseq_knn_model_base


class TestFairseqKNNModelBase:
    def test_fit_embed_dim(self, init_models) -> None:
        """test for get_embed_dim()."""

        models, saved_args = init_models
        self.knn_base = fairseq_knn_model_base.FairseqKNNModelBase(models)

        test_case: int = saved_args["model"].decoder_embed_dim
        embed_dims = self.knn_base.get_embed_dim()
        assert embed_dims == [test_case]
