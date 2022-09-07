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
