import pytest
import numpy as np
from er_simulator.load_wc_params import load_wc_params
from er_simulator.time_integration import time_integration

@pytest.fixture(scope="module")
def params():
    return load_wc_params(config_file="../er_simulator/wc_params.yaml")

def test_time_integration(params):

    params.duration = 100

    t, excs, inhs, exc_ou, inh_ou = time_integration(params)
    assert True
