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
        

class TestReadLines:
    def test_type_error1(self):
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=12, buffer_size=1):
                assert result_lines == None
    
    def test_type_error2(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("-")
        
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=path, buffer_size='1'):
                assert result_lines == ['-']
                
    def test_type_error3(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("-")
        
        with pytest.raises(TypeError):
            for result_lines in utils.read_lines(input=path, buffer_size=1, progress=100):
                assert result_lines == ['-']
                
    def test_no_file(self, tmp_path):
        path = tmp_path / "test.txt"
        
        with pytest.raises(FileNotFoundError):
            for result_lines in utils.read_lines(input=path, buffer_size=1):
                assert result_lines == None
                
    def test_empty_file(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("")
        
        with pytest.raises(StopIteration):
            result_lines = utils.read_lines(input=path, buffer_size=1)
            next(result_lines)
            
    def test_can_stop(self, tmp_path):
        lines = ["-"] * 4
        content = "\n".join(lines)
        path = tmp_path / "test.txt"
        path.write_text(content)
        assert path.read_text() == content
        
        result_lines = utils.read_lines(input=path, buffer_size=1)
        assert len(next(result_lines)) == 1
        assert len(next(result_lines)) == 1
        assert len(next(result_lines)) == 1
        assert len(next(result_lines)) == 1
        with pytest.raises(StopIteration):
            next(result_lines)
            
    def test1(self, tmp_path, capsys):
        self.generic_test(4, 4, True, tmp_path, capsys)
        
    def test2(self, tmp_path, capsys):
        self.generic_test(16, 4, False, tmp_path, capsys)
        
    def test3(self, tmp_path, capsys):
        self.generic_test(1, 4, True, tmp_path, capsys)
    
    def test4(self, tmp_path, capsys):
        self.generic_test(17, 4, True, tmp_path, capsys)
        
    def test5(self, tmp_path, capsys):
        self.generic_test(4, 1, True, tmp_path, capsys)
        
    
    def generic_test(self, num_lines, buffer_size, progress, tmp_path, capsys):
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