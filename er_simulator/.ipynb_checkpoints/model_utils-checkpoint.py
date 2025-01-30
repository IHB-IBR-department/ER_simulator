import numpy as np
# Adapted from https://github.com/neurolib-dev/neurolib

def adjustArrayShape(original: np.ndarray | list | float, target: np.ndarray) -> np.ndarray:
    """
    Adjusts the shape of an array (or list or float) to match the shape of a target array.

    This function tiles the original array until it is larger than or equal to the target array in each dimension,
    and then cuts it from the end to match the target shape precisely.  This ensures compatibility, for example,
    when applying an external current (which might be a scalar, list, or a differently-shaped array)
    to a rate array in a neural simulation.

    Args:
        original: The array, list, or float to be reshaped.  This represents the input data that needs to be adjusted.
        target: The target array whose shape the original array should match. This provides the desired output_type shape.

    Returns:
        A NumPy array with the same shape as the target array, containing the tiled and cut values of the original data.
        If the original input is a scalar, it's returned as a NumPy array with the target's shape filled with that scalar value.

    Raises:
        TypeError: if input `original` is not a NumPy array, list or a scalar (int, float).
        TypeError: if input `target` is not a NumPy array.
    """


    if not isinstance(target, np.ndarray):
        raise TypeError("Target must be a NumPy array.")

    if not isinstance(original, (np.ndarray, list, int, float)):
        raise TypeError("Original must be a NumPy array, list, or a scalar value.")



    # Convert to NumPy array for consistent handling. Ensures scalar inputs are handled correctly
    if not hasattr(original, "__len__"):
        original = [original]
    original = np.array(original)

    # repeat original in y until larger (or same size) as target

    # tile until N

    # either (x,) shape or (y,x) shape
    if len(original.shape) == 1:
        # if original.shape[0] > 1:
        rep_y = target.shape[0]
    elif target.shape[0] > original.shape[0]:
        rep_y = int(target.shape[0] / original.shape[0]) + 1
    else:
        rep_y = 1

    # tile once so the array has shape (N,1)
    original = np.tile(original, (rep_y, 1))

    # tile until t

    if target.shape[1] > original.shape[1]:
        rep_x = int(target.shape[1] / original.shape[1]) + 1
    else:
        rep_x = 1
    original = np.tile(original, (1, rep_x))

    # cut from end because the beginning can be initial condition
    original = original[: target.shape[0], -target.shape[1]:]
    return original


def computeDelayMatrix(lengthMat: np.ndarray, signalV: float, segmentLength: float = 1) -> np.ndarray:
    """
    Compute the delay matrix from the fiber length matrix and the signal
    velocity.

    Args:
        lengthMat (np.ndarray): A matrix containing the connection length in segments.
        signalV (float): Signal velocity in meters per second (m/s).
        segmentLength (float, optional): Length of a single segment in millimeters (mm). Defaults to 1.

    Returns:
        np.ndarray: A matrix of connection delays in milliseconds (ms).
    """

    normalizedLenMat = lengthMat * segmentLength
    if signalV > 0:
        Dmat = normalizedLenMat / signalV  # Interareal delays in ms
    else:
        Dmat = lengthMat * 0.0
    return Dmat

