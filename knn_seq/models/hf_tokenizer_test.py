from knn_seq.models.hf_tokenizer import space_tokenize


def test_space_tokenize():
    before = "  AB&c dEF  GH　　IJK l01m., n　"
    after = ["AB&c", "dEF", "GH", "IJK", "l01m.,", "n"]
    assert space_tokenize(before) == after
