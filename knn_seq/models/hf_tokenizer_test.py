import pytest
from transformers import AutoTokenizer

from knn_seq.models.hf_tokenizer import HFTokenizer

# wmt16 de-en train_data
_EXAMPLE = [
    "You have requested a debate on this subject in the course of the next few days, during this part-session.",
    "Resumption of the session",
    "It will, I hope, be examined in a positive light.",
]


class TestHFTokenizer:
    @pytest.mark.parametrize(
        "name_or_path",
        ["sentence-transformers/all-MiniLM-L6-v2", "distilbert-base-uncased"],
    )
    def test_encode(self, name_or_path):
        tokenizer = HFTokenizer.build_tokenizer(name_or_path)
        transformers_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        assert tokenizer.tokenizer.__class__ == transformers_tokenizer.__class__
        for text in _EXAMPLE:
            assert tokenizer.encode(text) == transformers_tokenizer.encode(
                text, add_special_tokens=False
            )

    @pytest.mark.parametrize(
        "name_or_path",
        ["sentence-transformers/all-MiniLM-L6-v2", "distilbert-base-uncased"],
    )
    def test_decode(self, name_or_path):
        tokenizer = HFTokenizer.build_tokenizer(name_or_path)
        transformers_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        for text in _EXAMPLE:
            decoded_example = tokenizer.decode(tokenizer.encode(text))
            assert decoded_example == transformers_tokenizer.tokenize(
                text, add_special_tokens=False
            )
