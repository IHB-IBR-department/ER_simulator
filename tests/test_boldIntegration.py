import numpy as np
from unittest import TestCase
import pytest
from er_simulator.boldIntegration import simulateBOLD
from er_simulator.boldIntegration import BWBoldModel
import matplotlib.pyplot as plt
from er_simulator.task_utils import create_task_design_activation, create_activations_per_module


class TestBWBoldModel:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dt = 10e-04
        self.model = BWBoldModel(1, self.dt, fix=True)
        self.model_random = BWBoldModel(1, self.dt, fix=False)
        self.model100 = BWBoldModel(100, self.dt, fix=False)

    def test_init_fix(self):

        assert self.model.N == 1
        assert np.isclose(self.model.rho, 0.34, atol=1e-03)
        assert np.isclose(self.model.alpha, 0.32, atol=1e-03)
        assert np.isclose(self.model.k, 0.65, atol=1e-03)
        assert np.isclose(self.model.tau, 0.98, atol=1e-03)
        assert self.model.X_BOLD[0] == 0.
        assert self.model.V_BOLD[0] == 1.
        assert self.model.Q_BOLD[0] == 1.
        assert self.model.F_BOLD[0] == 1.

    def test_init_random(self):

        assert self.model_random.N == 1
        assert 0.34 - 4 * np.sqrt(0.0024) <= self.model_random.rho <= 0.34 + 4 * np.sqrt(0.0024)
        assert 0.32 - 4 * np.sqrt(0.0015) <= self.model_random.alpha <= 0.32 + 4 * np.sqrt(0.0015)
        assert 0.65 - 4 * np.sqrt(0.015) <= self.model_random.k <= 0.65 + 4 * np.sqrt(0.015)
        assert 0.98 - 4 * np.sqrt(0.0568) <= self.model_random.tau <= 0.98 + 4 * np.sqrt(0.0568)
        assert self.model_random.X_BOLD == 0.
        assert self.model_random.V_BOLD == 1.
        assert self.model_random.Q_BOLD == 1.
        assert self.model_random.F_BOLD == 1.

    def test_init_model100(self):

        assert self.model100.N == 100
        assert self.model100.rho.shape == (100,)
        assert self.model100.alpha.shape == (100,)
        assert self.model100.gamma.shape == (100,)
        assert self.model100.k.shape == (100,)
        assert np.isclose(self.model100.X_BOLD, 0).all()
        assert np.isclose(self.model100.V_BOLD, 1).all()
        assert np.isclose(self.model100.Q_BOLD, 1).all()
        assert np.isclose(self.model100.F_BOLD, 1).all()

    def test_run_BOLD_impulse_one_node(self):
        # Create a simple test signal
        dt = self.model.dt
        # time length (in seconds)
        length = 25
        onset = 0
        duration = 10e-4
        activation = np.zeros((self.model.N, int(length / dt)))
        activation[:, int(onset / dt):int((onset + duration) / dt)] = 1

        BOLD = self.model.run(activation)
        plot = True
        if plot:
            plt.plot(BOLD.T)
            plt.show()

        # Basic checks:
        assert BOLD.size == int(length / dt)
        assert BOLD[0][0] == 0

    def test_run_BOLD_impulse_100_node(self):
        # Create a simple test signal
        dt = self.model100.dt
        # time length (in seconds)
        length = 25
        onset = 0
        duration = 10e-4
        activation = np.zeros((self.model100.N, int(length / dt)))
        activation[:, int(onset / dt):int((onset + duration) / dt)] = 1

        BOLD = self.model100.run(activation)
        plot = True
        if plot:
            plt.plot(BOLD.T)
            plt.show()

        # Basic checks:
        assert BOLD.shape == (100, int(length / dt))
        assert (np.argmax(BOLD, axis=1) > 2000).all()
        assert (np.argmax(BOLD, axis=1) < 10000).all()

    def test_run_on_impulse(self):

        BOLD, time = self.model100.run_on_impulse()
        plot = True
        if plot:
            plt.plot(time, BOLD.T)
            plt.show()
        assert BOLD.shape == (100, int(self.model100.length / self.model100.dt))
        assert (np.argmax(BOLD, axis=1) > 2000).all()
        assert (np.argmax(BOLD, axis=1) < 10000).all()

    def test_save_parameters_npy(self):

        file_path = 'test100.npy'
        self.model100.save_imp_with_params('test100.npy', s_type='npy')
        data = np.load(file_path, allow_pickle=True)
        assert (data.item()['rho'] == self.model100.rho).all()
        assert "alpha" in data.item().keys()
        assert "tau" in data.item().keys()
        assert "gamma" in data.item().keys()

    def test_run_bold_on_coactivation(self):
        onsets_list = [[10, 40, 60],
                       [20, 50, 70]]
        duration_list = [5, 5]
        box_car_activations = create_task_design_activation(
            onsets_list,
            duration_list,
            dt=10)
        #plt.plot(box_car_activations[0])
        #plt.plot(box_car_activations[1])
        activations = [[0, 1, 1],
                       [1, 0, 1]]
        activations_by_module = create_activations_per_module(activations,
                                                              box_car_activations)
        plt.subplot(311); plt.plot(activations_by_module[0])
        plt.subplot(312); plt.plot(activations_by_module[1])
        plt.subplot(313); plt.plot(activations_by_module[2])
        plt.show()
        model = BWBoldModel(3, dt=10e-3, fix=True, normalize_constant=1)
        bold_activations = model.run(activations_by_module)
        plt.plot(np.max(bold_activations)*activations_by_module[2])
        plt.plot(bold_activations[2])
        plt.show()


        assert True

    def test_save_parameters_mat(self):

        from scipy import io
        self.model100.save_imp_with_params('test100.mat', s_type='mat')
        data = io.loadmat('test100.mat')

        assert (data['rho'] == self.model100.rho).all()
        assert "alpha" in data.keys()
        assert "tau" in data.keys()
        assert "gamma" in data.keys()
        assert True


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
                                        alpha=(0.32, 0.0015),
                                        rho=(0.34, 0.0024),
                                        tau=(0.98, 0.0568),
                                        gamma=(0.41, 0.002),
                                        k=(0.65, 0.015),
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
        normalize_max = 1.0
        self.activation = normalize_max * self.activation
        BOLD, X, F, Q, V = simulateBOLD(self.activation,
                                        self.dt,
                                        fix=False)
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
