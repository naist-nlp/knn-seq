from knn_seq.search_index.faiss_index import FaissIndex
from knn_seq.search_index.faiss_index_fast import FaissIndexFast

__all__ = ["FaissIndex", "FaissIndexFast"]


SEARCH_INDEX_MAP = {
    FaissIndex.BACKEND_NAME: FaissIndex,
    FaissIndexFast.BACKEND_NAME: FaissIndexFast,
}


def load_index(path: str, **kwargs):
    from knn_seq.search_index.search_index import SearchIndex

    config = SearchIndex.load_config(path)
    return SEARCH_INDEX_MAP[config.backend].load(path, **kwargs)


def build_index(backend: str, *args, **kwargs):
    return SEARCH_INDEX_MAP[backend].new(*args, **kwargs)
