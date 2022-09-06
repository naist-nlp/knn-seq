from knn_seq.search_index.faiss_index import FaissIndex

__all__ = ["FaissIndex"]


SEARCH_INDEX_MAP = {"faiss": FaissIndex}


def load_index(path: str, **kwargs):
    from knn_seq.search_index.search_index import SearchIndex

    config = SearchIndex.load_config(path)
    return SEARCH_INDEX_MAP[config.backend].load(path, **kwargs)


def build_index(backend: str, *args, **kwargs):
    return SEARCH_INDEX_MAP[backend].new(*args, **kwargs)
