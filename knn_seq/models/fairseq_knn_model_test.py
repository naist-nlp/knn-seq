import faiss
import numpy as np
import pytest
import torch

from data.fixtures import (
    testdata_langpair_dataset,
    testdata_models,
    testdata_src_dict,
    testdata_tgt_dict,
)
from knn_seq.data.token_storage import TokenStorage
from knn_seq.dataset_wrapper import LanguagePairDatasetWithOriginalOrder
from knn_seq.models.fairseq_knn_model import FairseqKNNModel
from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.search_index import SearchIndexConfig


class TestFairseqKNNModel:
    @pytest.fixture
    def knn_model(self, testdata_models):
        ensemble, _ = testdata_models
        return FairseqKNNModel(ensemble)

    @pytest.fixture
    def testdata_collator(self, testdata_langpair_dataset):
        dataset = LanguagePairDatasetWithOriginalOrder(testdata_langpair_dataset)
        return dataset.collater([dataset[i] for i in range(10)])

    @pytest.fixture
    def testdata_token_storage(self, testdata_collator):
        return TokenStorage(
            tokens=torch.flatten(testdata_collator["net_input"]["src_tokens"]).numpy(),
            lengths=testdata_collator["net_input"]["src_lengths"],
            sort_order=testdata_collator["id"],
        )

    @pytest.mark.parametrize("topk", [None, 0])
    def test_set_index_bad_values(self, knn_model, testdata_token_storage, topk):
        dim = knn_model.get_embed_dim()[0]
        index = faiss.IndexFlatL2(dim)

        with pytest.raises(ValueError):
            knn_model.set_index(
                testdata_token_storage,
                indexes=[index],
                knn_topk=topk,
                knn_temperature=1.0,
                knn_threshold=None,
                knn_weight=0.5,
            )

    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("threshold", [None, 0.2])
    @pytest.mark.parametrize("temperature", [1.0, 2.5])
    @pytest.mark.parametrize("topk", [1, 3])
    def test_set_index(
        self, knn_model, testdata_token_storage, topk, temperature, threshold, weight
    ):
        dim = knn_model.get_embed_dim()[0]
        index = faiss.IndexFlatL2(dim)

        knn_model.set_index(
            testdata_token_storage,
            indexes=[index],
            knn_topk=topk,
            knn_temperature=temperature,
            knn_threshold=threshold,
            knn_weight=weight,
        )
        assert knn_model.val == testdata_token_storage
        assert knn_model.indexes == [index]
        assert knn_model.knn_topk == topk
        assert knn_model.knn_temperature == temperature

        assert knn_model.knn_threshold == threshold
        assert knn_model.knn_weight == weight
