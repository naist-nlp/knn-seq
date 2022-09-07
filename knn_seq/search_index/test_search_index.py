import json
from dataclasses import asdict

from knn_seq.search_index.search_index import SearchIndexConfig


class TestSearchIndexConfig:
    def test_save(self, tmp_path):
        cfg = SearchIndexConfig()
        cfg_path = tmp_path / "test_index_config.json"
        cfg.save(cfg_path)
        with open(cfg_path, mode="r") as f:
            assert asdict(cfg) == json.load(f)

    def test_load(self, tmp_path):
        cfg = SearchIndexConfig()
        cfg_path = tmp_path / "test_index_config.json"
        with open(cfg_path, mode="w") as f:
            json.dump(asdict(cfg), f, indent=True)

        assert cfg == SearchIndexConfig.load(cfg_path)
