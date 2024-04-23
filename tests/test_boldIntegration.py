import numpy as np
from unittest import TestCase
from tmfc_simulation.boldIntegration import simulateBOLD

class TestBold(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.N = 10
        cls.dt = 10e-3
        length = 25
        onset = 5
        duration = 2
        cls.activation = np.zeros((cls.N, int(length / cls.dt)))
        cls.activation[:, int(onset / cls.dt):int((onset + duration) / cls.dt)] = 1

    def test_simulate_bold(self):
       bold = simulateBOLD(self.activation,
                           self.dt,
                           fix=True)
       self.assertTrue(True)


