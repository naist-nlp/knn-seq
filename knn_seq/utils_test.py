import logging
import math
import time
from collections import UserDict, UserList

import numpy as np
import pytest
import torch

from knn_seq import utils


class TestReadLines:
    @pytest.fixture
    def tmp_file(self, tmp_path) -> str:
        path = tmp_path / "test.txt"
        return path

    def test_no_file(self, tmp_file):
        with pytest.raises(FileNotFoundError):
            for result_lines in utils.read_lines(input=str(tmp_file), buffer_size=1):
                assert result_lines == None

    def test_empty_file(self, tmp_file):
        tmp_file.write_text("")
        result_lines = utils.read_lines(input=str(tmp_file), buffer_size=1)

        with pytest.raises(StopIteration):
            next(result_lines)

    @pytest.mark.parametrize("buffer_size", [-1, 0])
    def test_zero_buffer(self, tmp_file, buffer_size):
        with pytest.raises(ValueError):
            result_lines = utils.read_lines(
                input=str(tmp_file), buffer_size=buffer_size
            )
            next(result_lines)

    def test_can_stop(self, tmp_file):
        lines = ["-"] * 3
        content = "\n".join(lines)
        tmp_file.write_text(content)

        result_lines = utils.read_lines(input=str(tmp_file), buffer_size=2)
        assert len(next(result_lines)) == 2
        assert len(next(result_lines)) == 1
        with pytest.raises(StopIteration):
            next(result_lines)

    @pytest.mark.parametrize(
        ("num_lines", "buffer_size", "progress"),
        [
            (1, 4, True),
            (4, 4, True),
            (16, 4, True),
            (17, 4, True),
            (4, 1, True),
            (1, 4, False),
            (4, 4, False),
            (16, 4, False),
            (17, 4, False),
            (4, 1, False),
        ],
    )
    def test(self, num_lines, buffer_size, progress, tmp_file, capsys):
        expected_repetitions = math.ceil(num_lines / buffer_size)

        lines = ["-"] * num_lines
        content = "\n".join(lines)
        tmp_file.write_text(content)

        num_repetitions = 0
        for result_lines in utils.read_lines(
            input=str(tmp_file), buffer_size=buffer_size, progress=progress
        ):
            num_repetitions += 1
            if num_repetitions == expected_repetitions:
                assert len(result_lines) <= buffer_size
            else:
                assert len(result_lines) == buffer_size

        assert num_repetitions == expected_repetitions

        if progress:
            captured_progress = capsys.readouterr().err
            progress_split = captured_progress.split("\r")
            assert progress_split[-1].startswith("{}it".format(num_lines))
        else:
            assert capsys.readouterr().err == ""


class TestToDevice:
    @pytest.fixture
    def tmp_tensor(self):
        return torch.arange(5)

    @pytest.mark.parametrize("src", ["cuda", "cpu"])
    @pytest.mark.parametrize("dst", ["cuda", "cpu"])
    def test_tensor(self, tmp_tensor, src, dst):
        if not torch.cuda.is_available() and (src == "cuda" or dst == "cuda"):
            pytest.skip("No CUDA available")

        tmp_tensor = tmp_tensor.to(device=src)
        result = utils.to_device(tmp_tensor, device=dst)
        assert result.device.type == dst

    @pytest.fixture
    def tmp_tensor_dict(self):
        temp_dict = {}
        for i in range(5):
            temp_dict[i] = torch.rand((3, 2)).cpu()

        return temp_dict

    @pytest.mark.parametrize("src", ["cuda", "cpu"])
    @pytest.mark.parametrize("dst", ["cuda", "cpu"])
    @pytest.mark.parametrize("is_user_dict", [True, False])
    def test_tensor_dict(self, tmp_tensor_dict, src, dst, is_user_dict):
        if not torch.cuda.is_available() and (src == "cuda" or dst == "cuda"):
            pytest.skip("No CUDA available")

        tmp_tensor_dict = {k: v.to(device=src) for k, v in tmp_tensor_dict.items()}

        if is_user_dict:
            tmp_tensor_dict = UserDict(tmp_tensor_dict)

        result = utils.to_device(tmp_tensor_dict, device=dst)

        for k, v in result.items():
            assert v.device.type == dst

        if is_user_dict:
            assert isinstance(result, UserDict)
        else:
            assert isinstance(result, dict)

    @pytest.fixture
    def tmp_tensor_list(self):
        temp_list = []
        for i in range(5):
            temp_list.append(torch.rand((3, 2)).cpu())

        return temp_list

    @pytest.mark.parametrize("src", ["cuda", "cpu"])
    @pytest.mark.parametrize("dst", ["cuda", "cpu"])
    @pytest.mark.parametrize("is_user_list", [True, False])
    def test_tensor_list(self, tmp_tensor_list, src, dst, is_user_list):
        if not torch.cuda.is_available() and (src == "cuda" or dst == "cuda"):
            pytest.skip("No CUDA available")

        tmp_tensor_list = [x.to(device=src) for x in tmp_tensor_list]

        if is_user_list:
            tmp_tensor_list = UserList(tmp_tensor_list)

        result = utils.to_device(tmp_tensor_list, device=dst)

        for x in result:
            assert x.device.type == dst

        if is_user_list:
            assert isinstance(result, UserList)
        else:
            assert isinstance(result, list)

    @pytest.mark.parametrize("item", ["hello", 1])
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_other_input(self, item, device):
        result = utils.to_device(item, device=device)
        assert result == item

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_nd_array(self, device):
        item = np.arange(5)
        result = utils.to_device(item, device=device)
        assert np.array_equal(result, item)


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
