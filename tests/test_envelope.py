import pytest
from scipy import io
import numpy as np
from scipy.signal import hilbert
from er_simulator.envelope import envelope_hilbert, normalize, env_peak
from matplotlib import pyplot as plt


@pytest.fixture
def simple_sinus():
    srate = 0.002  # 500 Hz sampling rate (s per sample)
    t = np.arange(0, 1, srate)  # 1 second of data
    x = np.sin(2 * np.pi * 40 * t)  # 40 Hz sine wave
    return t, x, srate


@pytest.fixture
def additive_cos():
    """Provide reusable test signals."""
    srate = 0.001
    t = np.arange(0, 2, srate)
    part10 = np.cos(2 * np.pi * 10 * t)
    part40 = np.cos(2 * np.pi * 40 * t)
    part100 = np.cos(2 * np.pi * 100 * t)
    signal = part10 + part40 + part100

    return t, signal, srate


@pytest.fixture
def multiple_cos():
    """Provide reusable test signals."""
    srate = 0.001
    t = np.arange(0, 2, srate)
    part10 = np.cos(2 * np.pi * 10 * t)
    part40 = np.cos(2 * np.pi * 40 * t)
    part100 = np.cos(2 * np.pi * 100 * t)

    signal = (1 + 0.5 * part10 * part40) * part100 + 5
    return t, signal, srate


@pytest.fixture
def decaying_cos():
    """Provide reusable test signals."""
    srate = 0.001
    t = np.arange(0, 2, srate)
    decaying_sine = (1 + np.cos(2 * np.pi * t / 0.5)) * np.exp(-0.4 * t)
    return t, decaying_sine, srate


@pytest.fixture
def real_signal():
    """Provide reusable test signals."""
    input_data = io.loadmat('env_test.mat')
    return input_data['task_syn_act_5ms'], input_data['task_env_600'], input_data['task_env_1000']


def test_envelope_shape_1d(simple_sinus):
    """Test that the function returns a 1D array for 1D input."""
    t, x, srate = simple_sinus
    env = envelope_hilbert(x, srate, low_f=20, high_f=100)
    assert env.shape == x.shape, "Envelope shape should match input shape."


def test_envelope_shape_2d(simple_sinus):
    """Test that the function returns a 2D array for multi-channel input."""
    t, x, srate = simple_sinus
    x_multi = np.vstack([x, x]).T  # Create a 2-channel signal
    env = envelope_hilbert(x_multi, srate)
    assert env.shape == x_multi.shape, "Envelope shape should match multi-channel input shape."


def test_envelope_constant_signal():
    """Test that the envelope of a constant signal is the constant itself."""
    x = np.ones(500)  # Constant signal
    srate = 0.002
    env = envelope_hilbert(x, srate)
    np.testing.assert_allclose(env, x, atol=1e-6, err_msg="Envelope of constant signal should be constant.")


def test_envelope_random_noise():
    """Test that the function does not crash on random noise."""
    np.random.seed(42)
    x = np.random.randn(1000)  # Gaussian noise
    srate = 0.002
    env = envelope_hilbert(x, srate, low_f=20, high_f=100)
    draw_plot = True
    if draw_plot:
        plt.plot(x)
        plt.plot(env)
        plt.show()
    assert env.shape == x.shape, "Envelope shape should match input shape."


def test_invalid_srate():
    """Test that an invalid sampling rate raises an error."""
    x = np.sin(2 * np.pi * np.linspace(0, 1, 500))
    with pytest.raises(ZeroDivisionError):
        envelope_hilbert(x, srate=0)


def test_envelope_peak(simple_sinus):
    t, x, srate = simple_sinus
    env = env_peak(x, 10)
    assert env.shape == x.shape, "Envelope shape should match input shape."

def test_envelope_shape_2d(simple_sinus):
    """Test that the function returns a 2D array for multi-channel input."""
    t, x, srate = simple_sinus
    x_multi = np.vstack([x, x]).T  # Create a 2-channel signal
    env = envelope_hilbert(x_multi, srate)
    env_p = env_peak(x_multi, 10)
    assert env.shape == x_multi.shape, "Envelope shape should match multi-channel input shape."
    assert env_p.shape == x_multi.shape, "Envelope shape should match multi-channel input shape."


def test_plot_env_peack(additive_cos):
    t, x, srate = additive_cos

    upper = env_peak(x, n=10)
    plt.plot(t, x)
    plt.plot(t, upper)

    plt.show()

    assert True

def test_plot_decaying_cos(decaying_cos):
    t, x, srate = decaying_cos

    upper = env_peak(x, n=10)
    env_h = envelope_hilbert(x, srate, low_f=None, high_f=None)

    plt.plot(t, x)
    plt.plot(t, upper)
    plt.plot(t, env_h)

    plt.show()
    assert True

def test_plot_muly_cos(multiple_cos):
    t, x, srate = multiple_cos

    upper = env_peak(x, n=10)
    env_h = envelope_hilbert(x, srate, low_f=None, high_f=None)
    env_h_b = envelope_hilbert(x, srate, low_f=20, high_f=90)


    plt.plot(t, x)
    plt.plot(t, upper)
    plt.plot(t, env_h)
    plt.plot(t, env_h_b)

    plt.show()
    assert True


def test_real_signal(real_signal):
    #input_data = io.loadmat('env_test.mat')
    sig5, env600, env1000 = real_signal
    env600p = env_peak(sig5.T[:30000, 0], 600)
    env1000p = env_peak(sig5.T[:30000, 0], 1000)
    env_h = envelope_hilbert(sig5.T[:, 0], 0.005, low_f=38, high_f=42)
    env_h_b = envelope_hilbert(sig5.T[:, 0], 0.005, low_f=30, high_f=50)
    assert np.corrcoef(env600[0,:20000], env600p[:20000])[0, 1] > 0.98
    assert np.corrcoef(env1000[0,:20000], env1000p[:20000])[0, 1] > 0.98
    assert np.corrcoef(env_h[:20000], env600p[:20000])[0, 1] > 0.5

    draw_plot=True
    if draw_plot:
        plt.subplot(221); plt.plot(env600p);  plt.plot(env600[0, :30000])
        plt.subplot(222); plt.plot(env1000p);  plt.plot(env1000[0, :30000])
        plt.subplot(223);  plt.plot(env_h[:30000])
        plt.subplot(224); plt.plot(env_h_b[:30000])


    plt.show()

    assert True

