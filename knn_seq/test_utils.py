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
        
class TestToDevice:
    # Note:  The documentation of this function says 'arbitrary data structures', 
    # but not every kind of data structure is available... 
    # Maybe we need to make the documentation more explicit?

    @pytest.fixture
    def tmp_tensor(self):
        return torch.arange(5)

    @pytest.mark.parametrize(
        "from_gpu,to_gpu", [(True, True), (True, False), (False, True), (False, False)]
    )
    def test_tensor(self, tmp_tensor, from_gpu, to_gpu):
        if not torch.cuda.is_available() and (to_gpu or from_gpu):
            warnings.warn("No CUDA available, this test always passes")
            return

        if from_gpu:
            tmp_tensor.cuda()
        else:
            tmp_tensor.cpu()

        result = utils.to_device(tmp_tensor, use_gpu=to_gpu)

        if to_gpu:
            assert result.device.type == "cuda"
        else:
            assert result.device.type == "cpu"

    @pytest.fixture
    def tmp_tensor_dict(self):
        temp_dict = {}
        for i in range(5):
            temp_dict[i] = torch.rand((3, 2)).cpu()

        return temp_dict

    @pytest.mark.parametrize(
        "from_gpu,to_gpu,is_user_dict",
        [
            (True, True, False),
            (True, False, False),
            (False, True, False),
            (False, False, False),
            (True, True, True),
            (True, False, True),
            (False, True, True),
            (False, False, True),
        ],
    )
    def test_tensor_dict(self, tmp_tensor_dict, from_gpu, to_gpu, is_user_dict):
        if not torch.cuda.is_available() and (to_gpu or from_gpu):
            pytest.skip("No CUDA available")

        if from_gpu:
            tmp_tensor_dict = {k: v.cuda() for k, v in tmp_tensor_dict.items()}

        if is_user_dict:
            tmp_tensor_dict = UserDict(tmp_tensor_dict)

        result = utils.to_device(tmp_tensor_dict, use_gpu=to_gpu)

        if to_gpu:
            for k, v in result.items():
                assert v.device.type == "cuda"
        else:
            for k, v in result.items():
                assert v.device.type == "cpu"

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

    @pytest.mark.parametrize(
        "from_gpu,to_gpu,is_user_list",
        [
            (True, True, False),
            (True, False, False),
            (False, True, False),
            (False, False, False),
            (True, True, True),
            (True, False, True),
            (False, True, True),
            (False, False, True),
        ],
    )
    def test_tensor_list(self, tmp_tensor_list, from_gpu, to_gpu, is_user_list):
        if not torch.cuda.is_available() and (to_gpu or from_gpu):
            pytest.skip("No CUDA available")

        if from_gpu:
            tmp_tensor_list = [x.cuda() for x in tmp_tensor_list]

        if is_user_list:
            tmp_tensor_list = UserList(tmp_tensor_list)

        result = utils.to_device(tmp_tensor_list, use_gpu=to_gpu)

        if to_gpu:
            for x in result:
                assert x.device.type == "cuda"
        else:
            for x in result:
                assert x.device.type == "cpu"

        if is_user_list:
            assert isinstance(result, UserList)
        else:
            assert isinstance(result, list)

    @pytest.mark.parametrize(
        "item, use_gpu", [("hello", True), ("hello", False), (1, True), (1, False)]
    )
    def test_other_input(self, item, use_gpu):
        result = utils.to_device(item, use_gpu=use_gpu)
        assert result == item

    @pytest.mark.parametrize(
        "item, use_gpu", [(np.arange(5), True), (np.arange(5), False)]
    )
    def test_nd_array(self, item, use_gpu):
        result = utils.to_device(item, use_gpu=use_gpu)
        assert np.array_equal(result, item)

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

