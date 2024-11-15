import pytest
import numpy as np
from tmfc_simulation.functions import resample_signal
from matplotlib import pyplot as plt
from scipy.signal import decimate

@pytest.fixture
def sample_neurosignal():
    TR = 2
    fMRI_T = 16
    original_dt = 0.1/1000  # in seconds
    new_dt = TR/fMRI_T
    time = np.arange(0, 1, original_dt)
    signal = np.sin(2 * np.pi * 10 * time)
    signal_3d = np.stack([signal, signal, signal])
    return time, signal, signal_3d, original_dt, new_dt

from scipy.signal import decimate

def test_resample_signal_downsample(sample_neurosignal):
    time, signal, signal_3d,original_dt, new_dt = sample_neurosignal
    downsampled_signal, downsampled_time,= resample_signal(time, signal, original_dt, new_dt, approx=False)
    downsampled_signal1, downsampled_time1, = resample_signal(time, signal, original_dt, new_dt, approx=True)

    plot = False
    if plot:
        plt.subplot(121); plt.plot(time, signal)
        plt.subplot(121); plt.plot(downsampled_time, downsampled_signal)
        plt.subplot(122); plt.plot(time, signal)
        plt.subplot(122); plt.plot(downsampled_time1, downsampled_signal1)

        plt.show()

    assert len(downsampled_signal) == len(downsampled_time)
    assert len(downsampled_time) == 8
    assert downsampled_time[-1] < time[-1] # because the new time array should cover the original timespan


def test_resample_signal_invalid_rates(sample_neurosignal):
    # Test with new_dt not being integer multiple of original_dt
    time, signal, signal_3d,original_dt, new_dt = sample_neurosignal
    new_dt = 0.05/1000  # less then old

    with pytest.raises(AssertionError):
        resample_signal(time, signal, original_dt, new_dt) # expecting error because factor is less then 1


def test_resample_3dsignal(sample_neurosignal):
    # Test with new_dt not being integer multiple of original_dt
    time, signal, signal_3d,original_dt, new_dt = sample_neurosignal
    downsampled_signal, downsampled_time,= resample_signal(time, signal_3d, original_dt, new_dt, approx=False)
    assert len(downsampled_signal[0]) == len(downsampled_time)








def test_resample_signal_no_change():
    # Test with new_sampling_time = original_sampling_time (no change)
    original_sampling_rate = 1000  # Hz
    new_sampling_rate = 1000  # Hz
    time = np.arange(0, 1, 1 / original_sampling_rate)
    signal = np.sin(2 * np.pi * 10 * time)  # 10 Hz sine wave

    downsampled_signal, downsampled_time = resample_signal(time, signal, original_sampling_rate,
                                                          new_sampling_rate)

    assert len(downsampled_signal) == len(signal)
    assert len(downsampled_time) == len(time)



