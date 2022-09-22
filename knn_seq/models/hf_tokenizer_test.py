import pytest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from knn_seq.models.hf_tokenizer import HFAutoTokenizer, space_tokenize
from knn_seq.models.sbert import SBERT_MODELS


def test_space_tokenize():
    before = "  AB&c dEF  GH　　IJK l01m., n　"
    after = ["AB&c", "dEF", "GH", "IJK", "l01m.,", "n"]
    assert space_tokenize(before) == after


# wmt16 de-en train_data
example = [
    "You have requested a debate on this subject in the course of the next few days, during this part-session.",
    "Resumption of the session",
    "It will, I hope, be examined in a positive light.",
]


class TestHFAutoTokenizer:
    @pytest.mark.parametrize(
        "name_or_path", ["all-MiniLM-L6-v2", "distilbert-base-uncased"]
    )
    @pytest.mark.parametrize("pretokenized", [True, False])
    def test_encode(self, name_or_path, pretokenized):
        tokenizer = HFAutoTokenizer.build_tokenizer(name_or_path, pretokenized)
        if name_or_path in SBERT_MODELS:
            encoder = SentenceTransformer(name_or_path, device="cpu").tokenizer
        else:
            encoder = AutoTokenizer.from_pretrained(name_or_path)
        for text in example:
            if tokenizer.pretokenized:
                assert tokenizer.encode(text) == encoder.convert_tokens_to_ids(
                    space_tokenize(text)
                )
            else:
                assert tokenizer.encode(text) == encoder.encode(
                    text, add_special_tokens=False
                )

    @pytest.mark.parametrize(
        "name_or_path", ["all-MiniLM-L6-v2", "distilbert-base-uncased"]
    )
    def test_decode(self, name_or_path):
        tokenizer = HFAutoTokenizer.build_tokenizer(name_or_path)
        for text in example:
            decoded_example = tokenizer.decode(tokenizer.encode(text))
            assert decoded_example == tokenizer.tokenizer.tokenize(
                text, add_special_tokens=False
            )
