import numpy as np
import scipy.signal
from scipy.signal import decimate


def resample_signal(time, signal: np.ndarray,
                    original_sampling_time: float,
                    new_sampling_time: float,
                    approx: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample signal to desired sampling time.
    Args:
        signal:
        original_sampling_time:
        desired_sampling_rate:

    Returns:

    """

    downsampling_factor = int(new_sampling_time / original_sampling_time)
    assert downsampling_factor > 1, "New time resolution must by lower than original."
    # Downsample the signal using decimation
    if approx:
        time_start = time[0]
        time_end = time[-1]
        downsampled_signal = decimate(signal, downsampling_factor)
        downsampled_time = np.linspace(time_start, time_end, len(downsampled_signal))
    else:
        downsampled_signal = signal[:, ::downsampling_factor]
        downsampled_time = time[::downsampling_factor]
    return downsampled_signal, downsampled_time

def getPowerSpectrum(activity, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns a power spectrum using Welch's method.

    :param activity: One-dimensional timeseries
    :type activity: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """
    # convert to one-dimensional array if it is an (1xn)-D array
    if activity.shape[0] == 1 and activity.shape[1] > 1:
        activity = activity[0]
    assert len(activity.shape) == 1, "activity is not one-dimensional!"

    f, Pxx_spec = scipy.signal.welch(
        activity,
        1000 / dt,
        window="hann",
        nperseg=int(spectrum_windowsize * 1000 / dt),
        scaling="spectrum",
    )
    f = f[f < maxfr]
    Pxx_spec = Pxx_spec[0: len(f)]
    if normalize:
        Pxx_spec /= np.max(Pxx_spec)
    return f, Pxx_spec


def getMeanPowerSpectrum(activities, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns the mean power spectrum of multiple timeseries.

    :param activities: N-dimensional timeseries
    :type activities: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """

    powers = np.zeros(getPowerSpectrum(activities[0], dt, maxfr, spectrum_windowsize)[0].shape)
    ps = []
    for rate in activities:
        f, Pxx_spec = getPowerSpectrum(rate, dt, maxfr, spectrum_windowsize)
        ps.append(Pxx_spec)
        powers += Pxx_spec
    powers /= len(ps)
    if normalize:
        powers /= np.max(powers)
    return f, powers
