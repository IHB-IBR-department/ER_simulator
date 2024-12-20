import pytest
import numpy as np
import yaml
from er_simulator.load_wc_params import load_wc_params
from collections import namedtuple

# Create a dummy dotdict for testing
dotdict = namedtuple('dotdict', [])

# Create a temporary YAML file for testing
with open("test_wc_params.yaml", "w") as f:
    yaml.dump({
        "a": 1,
        "b": 2.0,
        "c": "string",
        "signalV": 10  # Add signalV for delay calculation testing
    }, f)


def test_load_wc_params_default():
    params = load_wc_params(config_file="test_wc_params.yaml")
    assert params.N == 1
    assert params.Cmat.shape == (1, 1)
    assert params.lengthMat.shape == (1, 1)
    assert params.exc_init.shape == (1, 1)
    assert params.inh_init.shape == (1, 1)
    assert params.exc_ou.shape == (1,)
    assert params.inh_ou.shape == (1,)
    assert params.a == 1
    assert params.b == 2.0
    assert params.c == "string"

def test_load_wc_params_with_matrices():
    N = 3
    Cmat = np.random.rand(N, N)
    np.fill_diagonal(Cmat, 0)
    Dmat = np.random.rand(N, N)
    params = load_wc_params(Cmat=Cmat, Dmat=Dmat, config_file="test_wc_params.yaml")
    assert params.N == N
    assert np.allclose(params.Cmat, Cmat)  # Check for equality after diagonal zeroing
    np.fill_diagonal(Cmat, 0) # Zero the diagonal for comparison
    assert np.array_equal(params.Cmat,Cmat)
    assert np.allclose(params.lengthMat, Dmat)
    assert params.exc_init.shape == (N, 1)
    assert params.inh_init.shape == (N, 1)
    assert params.exc_ou.shape == (N,)
    assert params.inh_ou.shape == (N,)
    assert params.a == 1
    assert params.b == 2.0
    assert params.c == "string"

def test_load_wc_params_seed():
    seed = 42
    params1 = load_wc_params(seed=seed, config_file="test_wc_params.yaml")
    params2 = load_wc_params(seed=seed, config_file="test_wc_params.yaml")
    assert np.allclose(params1.exc_init, params2.exc_init)
    assert np.allclose(params1.inh_init, params2.inh_init)

def test_load_actual_params():
    params = load_wc_params(config_file="../er_simulator/wc_params.yaml")

    assert params.dt == 0.1

