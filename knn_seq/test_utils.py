from knn_seq import utils
import pytest, warnings
import math
import numpy as np
import torch
from collections import UserDict, UserList
import time


class TestBufferLines:
    @pytest.mark.parametrize("lines,buffer_size", [(4, 4), ([4], "4")])
    def test_type_errors(self, lines, buffer_size):
        with pytest.raises(TypeError):
            for result_lines in utils.buffer_lines(lines=lines, buffer_size=buffer_size):
                assert result_lines == None

    def test_zero_lines(self):
        with pytest.raises(StopIteration):
            result_lines = utils.buffer_lines(lines=[])
            next(result_lines)

    @pytest.mark.parametrize("buffer_size", [-1, 0])
    def test_zero_buffer(self, buffer_size):
        with pytest.raises(ValueError):
            lines = [1] * 20
            result_lines = utils.buffer_lines(lines=lines, buffer_size=buffer_size)
            next(result_lines)

    @pytest.mark.parametrize(
        "num_lines,buffer_size", [(1, 4), (4, 4), (16, 4), (17, 4), (4, 1)]
    )
    def test(self, num_lines, buffer_size):
        expected_repetitions = math.ceil(num_lines / buffer_size)

        lines = [1] * num_lines
        num_repetitions = 0
        for result_lines in utils.buffer_lines(lines=lines, buffer_size=buffer_size):
            num_repetitions += 1
            if num_repetitions == expected_repetitions:
                assert len(result_lines) <= buffer_size
            else:
                assert len(result_lines) == buffer_size

        assert num_repetitions == expected_repetitions
        
class TestReadLines:
    @pytest.fixture
    def tmp_file(self, tmp_path) -> str:
        path = tmp_path / "test.txt"
        return path

    def test_type_error_input(self):
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=12, buffer_size=1):
                assert result_lines == None

    @pytest.mark.parametrize("buffer_size, progress", [("4", True), (1, 100)])
    def test_type_errors(self, tmp_file, buffer_size, progress):
        tmp_file.write_text("-")

        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(
                input=str(tmp_file), buffer_size=buffer_size, progress=progress
            ):
                assert result_lines == ["-"]

    def test_no_file(self, tmp_file):
        with pytest.raises(FileNotFoundError):
            for result_lines in utils.read_lines(input=str(tmp_file), buffer_size=1):
                assert result_lines == None

    def test_empty_file(self, tmp_file):
        tmp_file.write_text("")
        result_lines = utils.read_lines(input=str(tmp_file), buffer_size=1)
        
        with pytest.raises(StopIteration):
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
        "num_lines,buffer_size,progress",
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

class TestPad:
    def test_type_errors(self):
        tensor_list = [np.arange(5), np.arange(2)]
        with pytest.raises(TypeError):
            result = utils.pad(tensor_list)

    def test_value_error(self):
        tensor_list = [torch.eye(2), torch.arange(1)]
        with pytest.raises(ValueError):
            result = utils.pad(tensor_list, -1)

    @pytest.mark.parametrize(
        "tensor_list, padding_idx, expected_value",
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

class TestReadLines:
    @pytest.fixture
    def tmp_file(self, tmp_path) -> str:
        path = tmp_path / "test.txt"
        return path

    def test_type_error_input(self):
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=12, buffer_size=1):
                assert result_lines == None

    @pytest.mark.parametrize("buffer_size, progress", [("4", True), (1, 100)])
    def test_type_errors(self, tmp_file, buffer_size, progress):
        tmp_file.write_text("-")

        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(
                input=str(tmp_file), buffer_size=buffer_size, progress=progress
            ):
                assert result_lines == ["-"]

    def test_no_file(self, tmp_file):
        with pytest.raises(FileNotFoundError):
            for result_lines in utils.read_lines(input=str(tmp_file), buffer_size=1):
                assert result_lines == None

    def test_empty_file(self, tmp_file):
        tmp_file.write_text("")
        result_lines = utils.read_lines(input=str(tmp_file), buffer_size=1)
        
        with pytest.raises(StopIteration):
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
        "num_lines,buffer_size,progress",
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
            
class TestStopwatchMeter:
    @pytest.fixture
    def stopwatch(self):
        return utils.StopwatchMeter()

    def test_start(self, stopwatch):
        assert stopwatch.start_time == None
        stopwatch.start()
        assert stopwatch.start_time != None

    def test_stop_type_errors(self, stopwatch):
        stopwatch.start()
        with pytest.raises(TypeError):
            stopwatch.stop(n=torch.arange(2))
            stopwatch.stop(n=5.0)
            stopwatch.stop(n="2")

    @pytest.mark.parametrize(
        "should_start, n, use_prehook",
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
    def test_stop(self, capsys, stopwatch, should_start, n, use_prehook):
        def simple_prehook():
            print("Hello?")

        expected_sum = 0
        expected_n = 0

        assert stopwatch.stop_time == None
        assert stopwatch.sum == expected_sum
        assert stopwatch.n == expected_n

        if should_start:
            stopwatch.start()

        if use_prehook:
            stopwatch.stop(n=n, prehook=simple_prehook)
        else:
            stopwatch.stop(n=n, prehook=None)

        expected_n += n

        if should_start:
            assert stopwatch.stop_time != None
            assert stopwatch.sum > 0
            assert stopwatch.sum < 0.01
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
        "n, use_prehook, iterations",
        [(1, False, 2), (2, True, 2), (2, False, 5), (0, False, 2), (0, True, 5)],
    )
    def test_multiple_start_stops(self, capsys, stopwatch, n, use_prehook, iterations):
        def simple_prehook():
            print("Hello?")

        expected_sum = 0
        expected_n = 0

        assert stopwatch.stop_time == None
        assert stopwatch.sum == expected_sum
        assert stopwatch.n == expected_n

        for _ in range(iterations):
            stopwatch.start()

            if use_prehook:
                stopwatch.stop(n=n, prehook=simple_prehook)
            else:
                stopwatch.stop(n=n, prehook=None)

            assert stopwatch.stop_time != None
            assert stopwatch.sum > expected_sum
            assert stopwatch.sum - expected_sum < 0.01
            expected_sum = stopwatch.sum
            expected_n += n

        assert stopwatch.n == expected_n

        if use_prehook:
            captured_progress = capsys.readouterr().out
            assert captured_progress == "Hello?\n" * iterations

    @pytest.mark.parametrize(
        "n, use_prehook", [(1, False), (1, True), (2, False), (1, True)]
    )
    def test_multiple_stops(self, capsys, stopwatch, n, use_prehook):
        def simple_prehook():
            print("Hello?")

        expected_sum = 0
        expected_n = 0

        assert stopwatch.stop_time == None
        assert stopwatch.sum == expected_sum
        assert stopwatch.n == expected_n
        stopwatch.start()

        if use_prehook:
            stopwatch.stop(n=n, prehook=simple_prehook)
        else:
            stopwatch.stop(n=n, prehook=None)

        expected_sum = stopwatch.sum
        expected_n = n

        # Stop again without calling start
        if use_prehook:
            stopwatch.stop(n=n, prehook=simple_prehook)
        else:
            stopwatch.stop(n=n, prehook=None)

        # The sum / n should not have increased
        assert stopwatch.sum == expected_sum
        assert stopwatch.n == expected_n

        if use_prehook:
            captured_progress = capsys.readouterr().out
            assert captured_progress == "Hello?\n"

    @pytest.mark.parametrize("with_start", [True, False])
    def test_reset(self, stopwatch, with_start):
        assert stopwatch.sum == 0
        assert stopwatch.n == 0
        assert stopwatch.start_time == None
        assert stopwatch.stop_time == None

        stopwatch.start()
        assert stopwatch.start_time != None
        stopwatch.stop()
        assert stopwatch.stop_time != None
        assert stopwatch.sum != 0
        assert stopwatch.n != 0

        if with_start:
            stopwatch.start()

        stopwatch.reset()
        assert stopwatch.sum == 0
        assert stopwatch.n == 0
        assert stopwatch.stop_time == None

        if not with_start:
            assert stopwatch.start_time == None

    @pytest.mark.parametrize(
        "n, iterations",
        [(0, 0), (1, 0), (0, 1), (1, 1), (1, 5), (5, 5)],
    )
    def test_avg(self, stopwatch, n, iterations):
        assert stopwatch.avg == 0

        expected_n = n * iterations

        for _ in range(iterations):
            stopwatch.start()
            time.sleep(0.01)
            stopwatch.stop(n=n)
            print(stopwatch.sum)

        assert stopwatch.sum >= 0.01 * iterations
        assert stopwatch.sum <= 0.02 * iterations

        expected_avg = stopwatch.sum

        if not isinstance(n, torch.Tensor) and expected_n > 0:
            expected_avg = stopwatch.sum / expected_n

        assert stopwatch.avg == expected_avg

    def test_elapsed_time(self, stopwatch):
        assert stopwatch.elapsed_time == 0
        stopwatch.start()
        assert stopwatch.elapsed_time > 0
        time.sleep(0.01)
        assert stopwatch.elapsed_time > 0.01
        assert stopwatch.elapsed_time < 0.02

        stopwatch.stop()
        assert stopwatch.elapsed_time == 0

    def test_lap_time(self, stopwatch):
        assert stopwatch.lap_time == 0
        stopwatch.start()
        assert stopwatch.lap_time > 0
        time.sleep(0.01)
        assert stopwatch.lap_time > 0.01
        assert stopwatch.lap_time < 0.02

        stopwatch.stop()
        assert stopwatch.lap_time > 0.01
        assert stopwatch.lap_time < 0.02

        time.sleep(0.01)
        assert stopwatch.lap_time < 0.02

    @pytest.mark.parametrize("label", ["", "hi"])
    def test_log_time(self, stopwatch, caplog, label):
        stopwatch.log_time(label=label)
        stopwatch.start()
        time.sleep(0.01)
        stopwatch.log_time(label=label)
        stopwatch.stop()
        stopwatch.start()
        time.sleep(0.03)
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
                "[Time] {}: 0.010 s, avg: 0.000 s, sum: 0.000 s".format(label),
            ),
            (
                "knn_seq.utils",
                logging.INFO,
                "[Time] {}: 0.030 s, avg: 0.020 s, sum: 0.040 s".format(label),
            ),
        ]
