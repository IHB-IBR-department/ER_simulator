import pytest
import numpy as np
from tmfc_simulation.load_wc_params import load_wc_params
from tmfc_simulation.time_integration import time_integration

@pytest.fixture(scope="module")
def params():
    return load_wc_params(config_file="../tmfc_simulation/wc_params.yaml")

def test_time_integration(params):

    params.duration = 100

    t, excs, inhs, exc_ou, inh_ou = time_integration(params)
    assert True
