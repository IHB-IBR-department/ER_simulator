import numpy as np
from unittest import TestCase
from tmfc_simulation.boldIntegration import simulateBOLD
import matplotlib.pyplot as plt


class TestBold(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.N = 100
        cls.dt = 10e-4
        cls.length = 25
        onset = 0
        duration = 10e-4
        cls.activation = np.zeros((cls.N, int(cls.length / cls.dt)))
        cls.activation[:, int(onset / cls.dt):int((onset + duration) / cls.dt)] = 1

    def test_simulate_bold_params(self):
        normalize_max = 1.0
        self.activation = normalize_max * self.activation
        BOLD, X, F, Q, V = simulateBOLD(self.activation,
                                        self.dt,
                                        alpha = (0.32, 0.0015),
                                        rho=(0.34, 0.0024),
                                        tau=(0.98, 0.0568),
                                        gamma=(0.41, 0.002),
                                        k = (0.65, 0.015),
                                        fix=False)
        self.assertRaises(AssertionError, simulateBOLD,
                          self.activation, self.dt, rho='tt')
        self.assertRaises(AssertionError, simulateBOLD,
                          self.activation, self.dt, alpha='0.34')

    def test_simulate_bold_fix(self):

        voxel_counts = 10000 * np.ones((self.N,))
        BOLD, X, F, Q, V = simulateBOLD(self.activation,
                            self.dt,
                            fix=True)
        plt.subplot(121);
        plt.plot(self.activation[0, :])
        plt.subplot(122);
        plt.plot(BOLD.T)
        plt.show()
        self.assertTrue(True)
    def test_simulate_bold_random(self):
        normalize_max=1.0
        self.activation= normalize_max * self.activation
        BOLD, X, F, Q, V = simulateBOLD(self.activation,
                            self.dt,
                            fix=False)
        time = np.linspace(0,self.length, int(self.length/self.dt))
        time_to_peaks = [time[np.argmax(BOLD[i])] for i in range(self.N)]
        plt.figure(figsize=(12,4))
        plt.subplot(121);
        plt.plot(time, BOLD.T); plt.title('Bold response for different nodes')
        plt.subplot(122);
        plt.hist(time_to_peaks); plt.title('Time to peak distribution')
        plt.show()
        self.assertTrue(True)


    def test_simulate_bold_all_different(self):
        normalize_max = 1.0
        self.activation = normalize_max * self.activation
        N = np.shape(self.activation)[0]
        rho, var_rho = 0.34, 0.0024
        Rho = np.random.normal(rho, np.sqrt(var_rho), size=(N,))

        BOLD, X, F, Q, V = simulateBOLD(self.activation,
                                        self.dt,
                                        rho=Rho,
                                        fix=True)
        time = np.linspace(0, self.length, int(self.length / self.dt))
        time_to_peaks = [time[np.argmax(BOLD[i])] for i in range(self.N)]
        plt.figure(figsize=(12, 4))
        plt.subplot(121);
        plt.plot(time, BOLD.T);
        plt.title('Bold response for different nodes')
        plt.subplot(122);
        plt.hist(time_to_peaks);
        plt.title('Time to peak distribution')
        plt.show()
        self.assertTrue(True)

