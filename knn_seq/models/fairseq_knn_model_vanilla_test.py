import faiss
import pytest
import torch

from data.fixtures import (
    testdata_langpair_dataset,
    testdata_models,
    testdata_src_dict,
    testdata_tgt_dict,
)
from knn_seq.data.token_storage import TokenStorage
from knn_seq.models.fairseq_knn_model_vanilla import FairseqKNNModel
from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.search_index import SearchIndexConfig
from knn_seq.tasks.dataset_wrapper import LanguagePairDatasetWithOriginalOrder


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

    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("temperature", [1.0, 2.5])
    @pytest.mark.parametrize("topk", [1, 3])
    @pytest.mark.parametrize("key", ["ffn_in", "ffn_out"])
    def test_search(
        self,
        testdata_models,
        testdata_token_storage,
        testdata_collator,
        key,
        topk,
        temperature,
        weight,
    ):
        torch.manual_seed(0)
        queries_to_test = 4

        ensemble, _ = testdata_models
        knn_model = FairseqKNNModel(ensemble, key=key)

        dim = knn_model.get_embed_dim()[0]

        model_out = knn_model.forward(
            src_tokens=testdata_collator["net_input"]["src_tokens"],
            src_lengths=testdata_collator["net_input"]["src_lengths"],
            prev_output_tokens=testdata_collator["net_input"]["prev_output_tokens"],
        )
        queries = model_out[0][:, queries_to_test]

        all_vectors = torch.flatten(model_out[0], end_dim=-2).numpy()
        index = FaissIndex(
            faiss.IndexFlatL2(dim),
            SearchIndexConfig(),
        )
        index.add(all_vectors)

        knn_model.set_index(
            testdata_token_storage,
            indexes=[index],
            knn_topk=topk,
            knn_temperature=temperature,
            knn_weight=weight,
        )

        output = knn_model.search(queries, 0)
        assert output != None
        assert torch.equal(
            output.indices[:, 0],
            testdata_collator["net_input"]["src_tokens"][:, queries_to_test],
        )

        # scores of all the exact matches should be 0
        assert torch.equal(output.scores[:, 0], torch.zeros((queries.shape[0])))

        for i in range(1, topk):
            assert not torch.equal(output.scores[:, i], torch.zeros((queries.shape[0])))

        offset = torch.normal(mean=0, std=0.1, size=queries.shape)
        distance = -torch.norm(offset, dim=1) ** 2
        output = knn_model.search(queries + offset, 0)
        torch.testing.assert_close(output.scores[:, 0], distance)
