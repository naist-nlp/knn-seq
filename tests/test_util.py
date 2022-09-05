from knn_seq import utils
import pytest
import math

class TestBufferLines:            
    def test_type_error1(self):
        with pytest.raises(TypeError):
            for result_lines in utils.buffer_lines(lines=4):
                assert result_lines == None
                
    def test_type_error2(self):
        with pytest.raises(TypeError):
            for result_lines in utils.buffer_lines(lines=[4], buffer_size='a'):
                assert result_lines == None
                
    def test_zero_lines(self):
        with pytest.raises(StopIteration):
            result_lines = utils.buffer_lines(lines=[])
            next(result_lines)
            
    #Fails but shouldn't
    def test_zero_buffer(self):
        with pytest.raises(ValueError):
            lines = [1] * 20
            result_lines = utils.buffer_lines(lines=lines, buffer_size=0)
            next(result_lines)
        
    #Fails but shouldn't
    def test_negative_buffer(self):
        with pytest.raises(ValueError):
            lines = [1] * 20
            result_lines = utils.buffer_lines(lines=lines, buffer_size=-1)
            next(result_lines)
            
    def test1(self):
        self.generic_test(4, 4)
            
    def test2(self):
        self.generic_test(16, 4)
    
    def test3(self):
        self.generic_test(1, 4)
    
    def test4(self):
        self.generic_test(17, 4)
        
    def test5(self):
        self.generic_test(4, 1)
        
    def generic_test(self, num_lines, buffer_size):
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