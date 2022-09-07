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
            
class TestSoftmax:
    def test_type_errors(self):
        tensor = np.arange(5)
        with pytest.raises(TypeError):
            result = utils.softmax(tensor)

    @pytest.mark.parametrize(
        "tensor", [torch.eye(5), torch.arange(5), torch.rand((2, 3))]
    )
    def test(self, tensor):
        result = utils.softmax(tensor)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(
            result, torch.nn.functional.softmax(tensor, dim=-1, dtype=torch.float32)
        )


class TestLogSoftmax:
    def test_type_errors(self):
        tensor = np.arange(5)
        with pytest.raises(TypeError):
            result = utils.log_softmax(tensor)

    @pytest.mark.parametrize(
        "tensor", [torch.eye(5), torch.arange(5), torch.rand((2, 3))]
    )
    def test(self, tensor):
        result = utils.log_softmax(tensor)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(
            result, torch.nn.functional.log_softmax(tensor, dim=-1, dtype=torch.float32)
        )

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

