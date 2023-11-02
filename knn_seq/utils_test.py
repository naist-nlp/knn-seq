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
    @pytest.fixture
    def stopwatch(self):
        return utils.StopwatchMeter()

    def test_start(self, stopwatch, monkeypatch):
        monkeypatch.setattr(time, "perf_counter", lambda: 120)
        assert stopwatch.start_time == None
        stopwatch.start()
        assert stopwatch.start_time == 120

    @pytest.mark.parametrize(
        ("should_start", "n", "use_prehook"),
        [
            (True, 1, False),
            (False, 1, False),
            (True, 2, False),
            (False, 2, False),
            (True, 0, False),
            (True, 1, True),
            (False, 1, True),
        ],
    )
    def test_stop(self, capsys, monkeypatch, stopwatch, should_start, n, use_prehook):
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        def simple_prehook():
            print("Hello?")

        expected_sum = 0
        expected_n = 0

        assert stopwatch.stop_time == None
        assert stopwatch.sum == expected_sum
        assert stopwatch.n == expected_n

        if should_start:
            stopwatch.start()

        fake_time += 5

        if use_prehook:
            stopwatch.stop(n=n, prehook=simple_prehook)
        else:
            stopwatch.stop(n=n, prehook=None)

        expected_n += n

        if should_start:
            assert stopwatch.stop_time == fake_time
            assert stopwatch.sum == 5
            assert stopwatch.n == expected_n

            expected_sum = stopwatch.sum
        else:
            assert stopwatch.stop_time == None

        if (
            use_prehook and should_start
        ):  # Do we want prehook to only be used if stop is called after start?
            captured_progress = capsys.readouterr().out
            assert captured_progress == "Hello?\n"

    @pytest.mark.parametrize(
        ("n", "use_prehook", "iterations"),
        [(1, False, 2), (2, True, 2), (2, False, 5), (0, False, 2), (0, True, 5)],
    )
    def test_multiple_start_stops(
        self, capsys, monkeypatch, stopwatch, n, use_prehook, iterations
    ):
        def simple_prehook():
            print("Hello?")

        expected_sum = 0
        expected_n = 0
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        assert stopwatch.stop_time == None
        assert stopwatch.sum == expected_sum
        assert stopwatch.n == expected_n

        for i in range(iterations):
            stopwatch.start()

            fake_time += 5

            if use_prehook:
                stopwatch.stop(n=n, prehook=simple_prehook)
            else:
                stopwatch.stop(n=n, prehook=None)

            assert stopwatch.stop_time == fake_time
            assert stopwatch.sum == 5 * (i + 1)
            expected_sum = stopwatch.sum
            expected_n += n

            fake_time += 2

        assert stopwatch.n == expected_n

        if use_prehook:
            captured_progress = capsys.readouterr().out
            assert captured_progress == "Hello?\n" * iterations

    @pytest.mark.parametrize("use_prehook", [True, False])
    @pytest.mark.parametrize("n", [1, 2])
    def test_multiple_stops(self, capsys, monkeypatch, stopwatch, n, use_prehook):
        def simple_prehook():
            print("Hello?")

        expected_n = 0
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        assert stopwatch.stop_time == None
        assert stopwatch.sum == 0
        assert stopwatch.n == expected_n
        stopwatch.start()

        fake_time += 5

        if use_prehook:
            stopwatch.stop(n=n, prehook=simple_prehook)
        else:
            stopwatch.stop(n=n, prehook=None)

        expected_n = n
        fake_time += 5

        # Stop again without calling start
        if use_prehook:
            stopwatch.stop(n=n, prehook=simple_prehook)
        else:
            stopwatch.stop(n=n, prehook=None)

        # The sum / n should not have increased
        assert stopwatch.sum == 5
        assert stopwatch.n == expected_n

        if use_prehook:
            captured_progress = capsys.readouterr().out
            assert captured_progress == "Hello?\n"

    @pytest.mark.parametrize("with_start", [True, False])
    def test_reset(self, stopwatch, monkeypatch, with_start):
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        assert stopwatch.sum == 0
        assert stopwatch.n == 0
        assert stopwatch.start_time == None
        assert stopwatch.stop_time == None

        stopwatch.start()
        assert stopwatch.start_time != None
        fake_time += 5

        stopwatch.stop()
        assert stopwatch.stop_time == fake_time
        assert stopwatch.sum == 5
        assert stopwatch.n == 1

        if with_start:
            stopwatch.start()

        stopwatch.reset()
        assert stopwatch.sum == 0
        assert stopwatch.n == 0
        assert stopwatch.stop_time == None

        if not with_start:
            assert stopwatch.start_time == None

    @pytest.mark.parametrize(
        ("n", "iterations"),
        [(0, 0), (1, 0), (0, 1), (1, 1), (1, 5), (5, 5)],
    )
    def test_avg(self, stopwatch, monkeypatch, n, iterations):
        assert stopwatch.avg == 0

        expected_n = n * iterations
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        for _ in range(iterations):
            stopwatch.start()

            fake_time += 5
            stopwatch.stop(n=n)

            fake_time += 2

        assert stopwatch.sum == 5 * iterations
        assert stopwatch.avg == (5 / (n if n != 0 else 1) if iterations > 0 else 0)

    def test_elapsed_time(self, monkeypatch, stopwatch):
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        assert stopwatch.elapsed_time == 0

        stopwatch.start()

        fake_time += 2
        assert stopwatch.elapsed_time == 2

        fake_time += 2
        assert stopwatch.elapsed_time == 4

        stopwatch.stop()
        assert stopwatch.elapsed_time == 0

    def test_lap_time(self, monkeypatch, stopwatch):
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)
        assert stopwatch.lap_time == 0

        stopwatch.start()

        fake_time += 2
        assert stopwatch.lap_time == 2

        fake_time += 2
        assert stopwatch.lap_time == 4

        stopwatch.stop()
        assert stopwatch.lap_time == 4

        fake_time += 2
        assert stopwatch.lap_time == 4

    @pytest.mark.parametrize("label", ["", "hi"])
    def test_log_time(self, monkeypatch, stopwatch, caplog, label):
        fake_time = 120
        monkeypatch.setattr(time, "perf_counter", lambda: fake_time)

        stopwatch.log_time(label=label)
        stopwatch.start()

        fake_time += 1

        stopwatch.log_time(label=label)
        stopwatch.stop()
        stopwatch.start()

        fake_time += 3

        stopwatch.stop()
        stopwatch.log_time(label=label)
        assert caplog.record_tuples == [
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: 0.000 s, avg: 0.000 s, sum: 0.000 s".format(label),
            ),
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: 1.000 s, avg: 0.000 s, sum: 0.000 s".format(label),
            ),
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: 3.000 s, avg: 2.000 s, sum: 4.000 s".format(label),
            ),
        ]
