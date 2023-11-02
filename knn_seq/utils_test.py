import logging
import time

import numpy as np
import pytest
import torch

from knn_seq import utils


class TestPad:
    def test_type_errors(self):
        tensor_list = [np.arange(5), np.arange(2)]
        with pytest.raises(TypeError):
            _ = utils.pad(tensor_list)

    @pytest.mark.parametrize(
        ("tensor_list", "padding_idx", "expected_value"),
        [
            ([torch.arange(1), torch.arange(2)], -1, torch.tensor([[0, -1], [0, 1]])),
            ([torch.arange(2), torch.arange(1)], -1, torch.tensor([[0, 1], [0, -1]])),
            ([torch.arange(1), torch.arange(1)], -1, torch.tensor([[0], [0]])),
            (
                [torch.arange(4), torch.arange(1)],
                -1,
                torch.tensor([[0, 1, 2, 3], [0, -1, -1, -1]]),
            ),
        ],
    )
    def test(self, tensor_list, padding_idx, expected_value):
        result = utils.pad(tensor_list, padding_idx)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, expected_value)


class TestStopwatchMeter:
    def test_call(self, monkeypatch):
        stopwatch = utils.Stopwatch()
        assert stopwatch.acc == 0.0
        assert stopwatch.avg == 0.0
        fake_time = 120.0
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)
        with stopwatch():
            fake_time += 5.0
        assert stopwatch.acc == 5.0
        assert stopwatch.avg == 5.0

    def test_multiple_start_stops(self, monkeypatch):
        stopwatch = utils.Stopwatch()
        fake_time = 0.0
        max_time = 5
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)
        assert stopwatch.acc == 0.0
        assert stopwatch.avg == 0.0
        for i in range(max_time):
            delta = float(i + 1)
            with stopwatch():
                fake_time += delta

            assert stopwatch.acc == fake_time
            assert stopwatch.avg == fake_time / float(i + 1)

        assert stopwatch.acc == fake_time
        assert stopwatch.avg == fake_time / float(max_time)

    def test_reset(self, monkeypatch):
        stopwatch = utils.Stopwatch()
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        assert stopwatch.acc == 0.0
        assert stopwatch.avg == 0.0

        delta = 5.0
        for _ in range(3):
            with stopwatch():
                fake_time += delta

        assert stopwatch.acc == delta * 3
        assert stopwatch.avg == delta

        stopwatch.reset()
        assert stopwatch.acc == 0.0
        assert stopwatch.avg == 0.0

    @pytest.mark.parametrize("label", ["", "hi"])
    def test_log_time(self, monkeypatch, caplog, label):
        stopwatch = utils.Stopwatch()
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        stopwatch.log(label=label)
        with stopwatch():
            fake_time += 1
            stopwatch.log(label=label)
        stopwatch.log(label=label)
        with stopwatch():
            fake_time += 3
        stopwatch.log(label=label)
        assert caplog.record_tuples == [
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: acc: 0.000 s, avg: 0.000 s (n=0)".format(label),
            ),
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: acc: 0.000 s, avg: 0.000 s (n=0)".format(label),
            ),
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: acc: 1.000 s, avg: 1.000 s (n=1)".format(label),
            ),
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: acc: 4.000 s, avg: 2.000 s (n=2)".format(label),
            ),
        ]
