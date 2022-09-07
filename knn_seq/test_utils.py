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
        

global caps, double_it


def caps(strs: [str]) -> [str]:
    return [s.upper() for s in strs]


def double_it(x: float) -> float:
    return x * 2


def scale(x: float, y: float) -> float:
    return x * y


class TestParallelApply:
    def test_no_workers(self):
        def double(x: float) -> float:
            return x * 2

        with pytest.raises(ValueError):
            result = utils.parallel_apply(double, [0.0, 1.0, 2.0, 3.0], 0)
            next(result)

    def test_simple(self):
        result = utils.parallel_apply(double_it, [0.0, 1.0, 2.0, 3.0])
        assert next(result) == 0
        assert next(result) == 2.0
        assert next(result) == 4.0
        assert next(result) == 6.0
        with pytest.raises(StopIteration):
            next(result)

    def test_simple_with_two_workers(self):
        result = utils.parallel_apply(double_it, [0.0, 1.0, 2.0, 3.0], 2)
        assert next(result) == [0.0, 2.0]
        assert next(result) == [4.0, 6.0]
        with pytest.raises(StopIteration):
            next(result)

    def test_with_args(self):
        result = utils.parallel_apply(scale, [0.0, 1.0, 2.0, 3.0], 1, 3)
        assert next(result) == 0
        assert next(result) == 3.0
        assert next(result) == 6.0
        assert next(result) == 9.0
        with pytest.raises(StopIteration):
            next(result)

    def test_with_kwargs(self):
        result = utils.parallel_apply(scale, [0.0, 1.0, 2.0, 3.0], 1, y=3)
        assert next(result) == 0
        assert next(result) == 3.0
        assert next(result) == 6.0
        assert next(result) == 9.0
        with pytest.raises(StopIteration):
            next(result)

    def test_with_bad_args(self):
        with pytest.raises(TypeError):
            result = utils.parallel_apply(scale, [0.0, 1.0, 2.0, 3.0], 1, 3, 5)
            next(result)

    def test_with_bad_kwargs(self):
        with pytest.raises(TypeError):
            result = utils.parallel_apply(scale, [0.0, 1.0, 2.0, 3.0], 1, z=3)
            next(result)

    def test_with_no_input(self):
        def simple() -> int:
            return 1

        with pytest.raises(TypeError):
            result = utils.parallel_apply(simple, [0.0, 1.0, 2.0, 3.0], 1)
            next(result)

    @pytest.fixture
    def tmp_file(self, tmp_path) -> str:
        lines = ["a"] * 5
        content = "\n".join(lines)
        path = tmp_path / "test.txt"
        path.write_text(content)
        return str(path)

    def test_with_read_lines(self, tmp_file):
        result = utils.parallel_apply(caps, utils.read_lines(tmp_file, 3), 1)
        assert next(result) == ["A\n", "A\n", "A\n"]
        assert next(result) == ["A\n", "A"]
        with pytest.raises(StopIteration):
            next(result)

    def test_with_two_workers(self, tmp_file):
        result = utils.parallel_apply(caps, utils.read_lines(tmp_file, 3), 2)

        assert next(result) == ["A\n", "A\n", "A\n", "A\n", "A"]
        with pytest.raises(StopIteration):
            next(result)

    def test_with_two_workers2(self, tmp_file):
        result = utils.parallel_apply(caps, utils.read_lines(tmp_file, 1), 2)

        assert next(result) == ["A\n", "A\n"]
        assert next(result) == ["A\n", "A\n"]
        assert next(result) == ["A"]
        with pytest.raises(StopIteration):
            next(result)

    def test_with_too_many_workers(self, tmp_file):
        result = utils.parallel_apply(caps, utils.read_lines(tmp_file, 3), 12)

        assert next(result) == ["A\n", "A\n", "A\n", "A\n", "A"]
        with pytest.raises(StopIteration):
            next(result)

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

