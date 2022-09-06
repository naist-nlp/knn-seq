from knn_seq import utils
import pytest, warnings
import math
import numpy as np
import torch


class TestBufferLines:
    @pytest.mark.parametrize('lines,buffer_size', [(4, 4), ([4], '4')])
    def test_type_errors(self, lines, buffer_size):
        with pytest.raises(TypeError):
            for result_lines in utils.buffer_lines(lines=4):
                assert result_lines == None
                                
    def test_zero_lines(self):
        with pytest.raises(StopIteration):
            result_lines = utils.buffer_lines(lines=[])
            next(result_lines)
            
    #Fails but shouldn't
    @pytest.mark.parametrize('buffer_size', [-1, 0])
    def test_zero_buffer(self, buffer_size):
        with pytest.raises(ValueError):
            lines = [1] * 20
            result_lines = utils.buffer_lines(lines=lines, buffer_size=0)
            next(result_lines)
            
    @pytest.mark.parametrize('num_lines,buffer_size', [(1, 4), (4, 4), (16, 4), (17, 4), (4, 1)])
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
    def filepath(self, tmp_path) -> str:
        path = tmp_path / "test.txt"
        return path
    
    def test_type_error_input(self):
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=12, buffer_size=1):
                assert result_lines == None
    
    @pytest.mark.parametrize('buffer_size, progress', [('4', True), (1, 100)])                
    def test_type_errors(self, filepath, buffer_size, progress):
        filepath.write_text('-')
        
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=filepath, buffer_size=buffer_size, progress=progress):
                assert result_lines == ['-']
                                
    def test_no_file(self, filepath):
        with pytest.raises(FileNotFoundError):
            for result_lines in utils.read_lines(input=filepath, buffer_size=1):
                assert result_lines == None
                
    def test_empty_file(self, filepath):
        filepath.write_text("")
        
        with pytest.raises(StopIteration):
            result_lines = utils.read_lines(input=filepath, buffer_size=1)
            next(result_lines)
            
    def test_can_stop(self, filepath):
        lines = ["-"] * 3
        content = "\n".join(lines)
        filepath.write_text(content)
        
        result_lines = utils.read_lines(input=filepath, buffer_size=2)
        assert len(next(result_lines)) == 2
        assert len(next(result_lines)) == 1
        with pytest.raises(StopIteration):
            next(result_lines)
        
    @pytest.mark.parametrize('num_lines,buffer_size,progress', [(1, 4, True), (4, 4, True), (16, 4, True), (17, 4, True), (4, 1, True), (1, 4, False), (4, 4, False), (16, 4, False), (17, 4, False), (4, 1, False)])
    def test(self, num_lines, buffer_size, progress, tmp_path, capsys):
        expected_repetitions = math.ceil(num_lines / buffer_size)
        
        lines = ["-"] * num_lines
        content = "\n".join(lines)
        path = tmp_path / "test.txt"
        path.write_text(content)
        assert path.read_text() == content

        num_repetitions = 0
        for result_lines in utils.read_lines(input=path, buffer_size=buffer_size, progress=progress):
            num_repetitions += 1
            if num_repetitions == expected_repetitions:
                assert len(result_lines) <= buffer_size
            else:
                assert len(result_lines) == buffer_size
                
        assert num_repetitions == expected_repetitions
        
        if progress:
            captured_progress = capsys.readouterr().err
            progress_split = captured_progress.split('\r')
            assert progress_split[-1].startswith('{}it'.format(num_lines))
        else:
            assert capsys.readouterr().err == ''
      
global caps, double_it
        
def caps(strs: [str]) -> [str]:
    return [s.upper() for s in strs]
    
def double_it(x: float) -> float:
    return x * 2
    
def scale(x: float, y: float) -> float:
    return x * y
            
class TestParallelApply:
    #Fails but shouldn't
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
            
    #This fails and I wonder if it should...
    #There's nothing in the documentation for the function that prohibts this
    #It works with a single worker, but not with multiple
    def test_simple_with_two_workers(self):
        result = utils.parallel_apply(double_it, [0.0, 1.0, 2.0, 3.0], 2)
        assert next(result) == 0
        assert next(result) == 2.0
        assert next(result) == 4.0
        assert next(result) == 6.0
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
    def tmp_file(self, tmp_path):
        lines = ["a"] * 5
        content = "\n".join(lines)
        path = tmp_path / "test.txt"
        path.write_text(content)
        return path
        
    def test_with_read_lines(self, tmp_file):
        result = utils.parallel_apply( 
            caps,
            utils.read_lines(tmp_file, 3),
            1)
        assert next(result) == ["A\n", "A\n", "A\n"]
        assert next(result) == ["A\n", "A"]
        with pytest.raises(StopIteration):
            next(result)
            
          
    def test_with_two_workers(self, tmp_file):
        result = utils.parallel_apply( 
                caps,
                utils.read_lines(tmp_file, 3),
                2)
                
        assert next(result) == ["A\n", "A\n", "A\n", "A\n", "A"]
        with pytest.raises(StopIteration):
            next(result)
            
    def test_with_two_workers2(self, tmp_file):
        result = utils.parallel_apply( 
                caps,
                utils.read_lines(tmp_file, 1),
                2)
                
        assert next(result) == ["A\n", "A\n"]
        assert next(result) == ["A\n", "A\n"]
        assert next(result) == ["A"]
        with pytest.raises(StopIteration):
            next(result)
            
    def test_with_too_many_workers(self, tmp_file):
        result = utils.parallel_apply( 
                caps,
                utils.read_lines(tmp_file, 3),
                12)
                
        assert next(result) == ["A\n", "A\n", "A\n", "A\n", "A"]
        with pytest.raises(StopIteration):
            next(result)

class TestToNDArray:
    def test_ndarray(self):
        array = np.arange(5)
        result = utils.to_ndarray(array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.arange(5))
        
    def test_torch_cpu(self):
        array = torch.arange(5).to('cpu')
        result = utils.to_ndarray(array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.arange(5))
    
    def test_torch_gpu(self):
        if not torch.cuda.is_available():
            warnings.warn("No CUDA available, this test always passes")
            return
        
        array = torch.arange(5).to('cuda')
        result = utils.to_ndarray(array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.arange(5))
        
    def test_list(self):
        array = [0, 1, 2, 3, 4]
        result = utils.to_ndarray(array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.arange(5))
        
    def test_empty(self):
        array = torch.tensor([])
        result = utils.to_ndarray(array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([]))
        
    def test_wrong_type(self):
        array = "0, 1, 2, 3, 4"
        with pytest.raises(TypeError):
            result = utils.to_ndarray(array)      
      
class TestToDevice:
    #Note:  The documentation of this function says 'arbitrary data structures', but not every kind of data structure is available... Maybe we need to make the documentation more explicit?
    
    @pytest.fixture
    def tmp_tensor(self):
        return torch.arange(5)
        
    @pytest.mark.parametrize('from_gpu,to_gpu', [(True, True), (True, False), (False, True), (False, False)])
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
            assert result.device == torch.device(type='cuda')
        else:
            assert result.device == torch.device(type='cpu')
            
    @pytest.fixture
    def tmp_tensor_dict(self):
        temp_dict = {}
        for i in range(5):
            temp_dict[i] = torch.rand((3, 2)).cpu()
            
        return temp_dict
        
    @pytest.mark.parametrize('from_gpu,to_gpu', [(True, True), (True, False), (False, True), (False, False)])
    def test_tensor_dict(self, tmp_tensor_dict, from_gpu, to_gpu):
        if not torch.cuda.is_available() and (to_gpu or from_gpu):
            warnings.warn("No CUDA available, this test always passes")
            return
            
        if from_gpu:
            tmp_tensor_dict = {k : v.cuda() for k, v in tmp_tensor_dict.items()}
            
        result = utils.to_device(tmp_tensor_dict, use_gpu=to_gpu)
        
        if to_gpu:
            for k, v in tmp_tensor_dict:
                assert v.device == torch.device(type='cuda')
        else:
            for k, v in tmp_tensor_dict:
                assert v.device == torch.device(type='cpu')
                