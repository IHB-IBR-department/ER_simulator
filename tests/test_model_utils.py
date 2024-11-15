import pytest
import numpy as np
from tmfc_simulation.model_utils import adjustArrayShape  # Assuming your function is here

def test_adjustArrayShape_scalar():
    target = np.zeros((2, 3))
    original = 5
    result = adjustArrayShape(original, target)
    assert np.all(result == 5)
    assert result.shape == (2, 3)

def test_adjustArrayShape_list():
    target = np.zeros((3, 2))
    original = [1, 2]
    result = adjustArrayShape(original, target)
    expected = np.array([[1, 2], [1, 2], [1, 2]])
    assert np.all(result == expected)
    assert result.shape == (3, 2)


def test_adjustArrayShape_1d_array():
    target = np.zeros((4, 1))
    original = np.array([1, 2, 3])
    result = adjustArrayShape(original, target)
    expected = np.array([[3], [3], [3], [3]])
    assert np.all(result == expected)
    assert result.shape == (4, 1)

def test_adjustArrayShape_2d_array_smaller():
    target = np.zeros((3, 4))
    original = np.array([[1, 2], [3, 4]])
    result = adjustArrayShape(original, target)
    expected = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2]])
    assert np.all(result == expected)
    assert result.shape == (3, 4)


def test_adjustArrayShape_2d_array_larger():
    target = np.zeros((2, 2))
    original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = adjustArrayShape(original, target)
    expected = np.array([[2, 3], [5, 6]])
    assert np.all(result == expected)
    assert result.shape == target.shape


def test_adjustArrayShape_invalid_input_type():
    target = np.zeros((2, 2))
    original = "invalid"
    with pytest.raises(TypeError):
        adjustArrayShape(original, target)

def test_adjustArrayShape_invalid_target_type():
  original = np.array([1,2])
  target = [1,2]
  with pytest.raises(TypeError):
        adjustArrayShape(original, target)


