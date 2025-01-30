import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d, UnivariateSpline
from typing import Tuple, Union


def normalize(signal):
    x = np.asarray(signal)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    x_mean = np.mean(x, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    return (x - x_mean) / x_std


def env_peak(x: Union[np.ndarray, list], n: int) -> np.ndarray:
    """
    Compute the upper envelope of a signal using peak detection and spline interpolation.

    Parameters:
    x : Union[np.ndarray, list]
        Input signal, shape (samples,) or (samples, channels).
    n : int
        Minimum peak separation (in samples).

    Returns:
    np.ndarray
        Upper envelope of the signal, same shape as x.

    Examples:
    --------
    >>> # Example 1: Single-channel signal
    >>> x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    >>> yupper = env_peak(x, 50)
    >>> yupper.shape
    (1000,)

    >>> # Example 2: Multi-channel signal
    >>> x = np.vstack([np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000)),
    ...                np.cos(2 * np.pi * 3 * np.linspace(0, 1, 1000))]).T
    >>> yupper = env_peak(x, 50)
    >>> yupper.shape
    (1000, 2)

    >>> # Example 3: Edge case with insufficient peaks
    >>> x = np.array([1, 2, 1, 2, 1])
    >>> yupper = env_peak(x, 2)
    >>> yupper
    array([2., 2., 2., 2., 2.])
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    nx, num_channels = x.shape
    yupper = np.zeros_like(x)

    # Compute upper envelope
    for chan in range(num_channels):
        if nx > n + 1:
            peaks, _ = find_peaks(x[:, chan], distance=n)
        else:
            peaks = np.array([])

        if len(peaks) < 2:
            # Include the first and last points
            i_locs = np.array([0, *peaks, nx - 1])
        else:
            i_locs = peaks

        if len(peaks) == 1:
            # If only one peak, set a constant envelope at peak value
            peak_value = x[peaks[0], chan]
            yupper[:, chan] = peak_value
        else:
            # Smoothly connect maxima via spline interpolation
            kind = 'cubic' if len(i_locs) >= 4 else 'linear'
            interp_func = interp1d(i_locs, x[i_locs, chan], kind=kind, fill_value="extrapolate")
            yupper[:, chan] = interp_func(np.arange(nx))

    return yupper.squeeze()


def envelope_hilbert(
        x: np.ndarray,
        srate: float,  # Given in seconds per sample (not Hz!)
        low_f: float = 38,
        high_f: float = 42
) -> np.ndarray:
    """
    Compute the upper envelope of the gamma-band filtered signal.

    Parameters:
    ----------
    x : np.ndarray
        Input signal (1D or 2D array, shape: samples x channels).
    srate : float
        Sampling interval in seconds per sample (not Hz!).
    low_f : float, optional
        Lower bound of the gamma bandpass filter in Hz (default: 38 Hz).
    high_f : float, optional
        Upper bound of the gamma bandpass filter in Hz (default: 42 Hz).
    n : int | None, optional
        Number of FFT points for the Hilbert transform (default: None).

    Returns:
    -------
    np.ndarray
        Upper envelope of the gamma-band filtered signal, same shape as `x`.
        If input has a single channel, output is returned as a 1D array.

    Example:
    -------
    >>> import numpy as np
    >>> srate = 0.002  # Sampling interval (500 Hz sampling rate)
    >>> t = np.arange(0, 1, srate)  # 1 second of data
    >>> x = np.sin(2 * np.pi * 40 * t)  # 40 Hz sine wave
    >>> env = envelope_hilbert(x, srate)
    >>> env.shape
    (500,)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]  # Convert to 2D for consistency

    x_mean = np.mean(x, axis=0, keepdims=True)

    nyquist = 1 / (2 * srate)  # Correct Nyquist frequency calculation

    if (low_f and high_f) is not None:
        low_band = low_f / nyquist
        high_band = high_f / nyquist

        # Design bandpass filter
        b, a = butter(N=4, Wn=[low_band, high_band], btype='bandpass')

        # Apply zero-phase filtering
        filtered = filtfilt(b, a, x, axis=0)
    else:
        filtered = x - x_mean

    # Compute envelope via Hilbert transform
    amplitude = np.abs(hilbert(filtered, axis=0))

    # Upper envelope
    upper_env = amplitude + x_mean

    return upper_env.squeeze()
