import pytest
import cProfile
import time
import pstats
from tmfc_simulation.wc_model import WCTaskSim
from tmfc_simulation.load_wc_params import load_wc_params
import numpy as np
from matplotlib import pyplot as plt
from tmfc_simulation.functions import resample_signal
from tmfc_simulation.boldIntegration import BWBoldModel
from tmfc_simulation.synaptic_weights_matrices import normalize, generate_synaptic_weights_matrices


class TestWCTaskSimBasic:

    @pytest.fixture(autouse=True)  # autouse=True applies this fixture to all tests in the class
    def setup_params(self):
        """Sets up default parameters for WCTaskSim tests."""

        # Use load_wc_params to load default parameters or provide a test YAML
        self.wc_params = load_wc_params(config_file="../tmfc_simulation/wc_params.yaml")  # Replace with your config

        # Create a dummy Cmat and Dmat if needed
        N = 3  # Example number of nodes
        np.random.seed(26)  # Example seed value
        self.wc_params.Cmat = np.abs(np.random.rand(N, N))
        # Zero diagonal
        np.fill_diagonal(self.wc_params.Cmat, 0)
        self.wc_params.lengthMat = 250 * np.ones((N, N))
        # Zero diagonal
        np.fill_diagonal(self.wc_params.lengthMat, 0)
        self.wc_params.N = N  # set N from generated Cmat

    @pytest.fixture
    def setup_one_event(self):
        onset_time_list = [2]
        tasks_name_list = ["Task1"]
        Cmat = 2 * np.abs(np.random.rand(3, 3))
        duration_list = [3]
        return onset_time_list, tasks_name_list, duration_list, {"Task1": Cmat}

    @pytest.fixture
    def setup_solo_rest(self):
        onset_time_list = []
        tasks_name_list = []
        duration_list = []
        return onset_time_list, tasks_name_list, duration_list

    @pytest.fixture(autouse=True)
    def setup_two_events(self):
        onset_time_list = [2, 5]
        tasks_name_list = ["Task1", "Task2"]
        duration_list = [3, 2]
        return onset_time_list, tasks_name_list, duration_list

    @pytest.fixture
    def setup_neurolib_n3(self):
        seed = 26
        N = 3
        Cmat = np.array([[0., 0.51939148, 0.76829766],
                         [0.78922074, 0., 0.18792139],
                         [0.26950525, 0.49619214, 0.]])
        D = 250 * np.ones((N, N))
        wc_params = load_wc_params(config_file="../tmfc_simulation/wc_params.yaml")
        sim = WCTaskSim(wc_params,
                        C_rest=Cmat,
                        D=D,
                        output_type="exc",
                        seed=seed,
                        TR=2,
                        fMRI_T=20,
                        rest_duration=40,
                        rest_before=0,
                        rest_after=0
                        )
        sim.wc_params['exc_init'] = np.array([[0.01347526], [0.02480961], [0.03695609]])
        sim.wc_params['inh_init'] = np.array([[0.0097476], [0.00898726], [0.02694131]])
        return sim

    def test_complete_onsets_with_rest_empty_input(self):
        sim = WCTaskSim(self.wc_params, rest_before=3, rest_after=3)  # Pass wc_params to the constructor
        onsets, tasks = sim.complete_onsets_with_rest([], [], [],
                                                      1, 2, rest_duration=6)
        assert onsets == [-1, 6]
        assert tasks == ["Rest"]

        onsets, tasks = sim.complete_onsets_with_rest([], None, None,
                                                      1, 2, rest_duration=6)
        assert onsets == [-1, 6]
        assert tasks == ["Rest"]

    def test_complete_onsets_with_rest_single_task(self):
        sim = WCTaskSim(self.wc_params)
        onsets, tasks = sim.complete_onsets_with_rest([2], ["Task1"], [3], 1, 2)
        assert onsets == [-1, 2, 5, 7]
        assert tasks == ["Rest", "Task1", "Rest"]

    def test_complete_onsets_with_rest_two_task(self):
        sim = WCTaskSim(self.wc_params)
        onsets, tasks = sim.complete_onsets_with_rest([2, 5], ["Task1", "Task2"],
                                                      [3, 2], 0, 0)
        assert onsets == [0.0, 2, 5, 7]
        assert tasks == ["Rest", "Task1", "Task2"]

    def test_generate_full_single_chunk_without_update(self):
        sim = WCTaskSim(self.wc_params)
        duration = 10
        Cmat = self.wc_params.Cmat
        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(Cmat, duration)
        assert Cmat.shape[0] == excs.shape[0]
        assert len(t) == excs.shape[1]

    def test_generate_full_single_chunk_with_update(self):
        sim = WCTaskSim(self.wc_params)
        duration = 1
        Cmat = self.wc_params.Cmat
        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(Cmat,
                                                                        duration * 1000,
                                                                        update_inits=True)

        assert np.allclose(sim.inits['exc_init'], excs[:, -sim.max_delay - 1:])
        assert np.allclose(sim.inits['exc_ou'], exc_ou)
        assert np.allclose(sim.inits['inh_ou'], inh_ou)
        assert Cmat.shape[0] == excs.shape[0]
        assert len(t) == excs.shape[1]

    def test_generate_chunk_with_onsets_solo_rest(self):
        sim = WCTaskSim(self.wc_params)
        sim._init_simulation(self.wc_params.lengthMat, self.wc_params.Cmat)
        start_time = 0
        out_dict = sim._generate_chunk_with_onsets(start_time)
        assert isinstance(out_dict, dict)
        assert len(out_dict['mtime']) == out_dict['syn_act'].shape[1]

    def test_generate_chunk_with_onsets_one_event(self, setup_one_event):
        onset_time_list, tasks_name_list, duration_list, C_task_dict = setup_one_event
        C_rest = 2 * np.abs(np.random.rand(3, 3))
        sim = WCTaskSim(self.wc_params,
                        C_rest=C_rest,
                        C_task_dict=C_task_dict,
                        onset_time_list=onset_time_list,
                        task_name_list=tasks_name_list,
                        duration_list=duration_list,
                        rest_before=0,
                        rest_after=0,
                        TR=2,
                        fMRI_T=20
                        )

        start_time = 0.95
        out_dict = sim._generate_chunk_with_onsets(start_time)
        assert out_dict['mtime'][-1] - start_time <= sim.chunksize
        assert sim.inits['last_t'] == 2000.
        assert out_dict['mtime'].shape[0] == out_dict['syn_act'].shape[1]
        start_time = 4
        out_dict = sim._generate_chunk_with_onsets(start_time)
        assert out_dict['mtime'][-1] - start_time <= sim.chunksize

    def test_single_node_run(self):
        self.wc_params.N = 1
        self.wc_params.exc_ext = 2.5
        self.wc_params.sigma_ou = 0
        self.wc_params.Cmat = np.array([[0.]])
        self.wc_params.lengthMat = np.array([[0.]])

        sim = WCTaskSim(self.wc_params,
                        C_rest=self.wc_params.Cmat,
                        output_type="exc",
                        fMRI_T=2000,
                        duration_list=1,
                        chunksize=2,
                        rest_before=0,
                        rest_after=0,
                        rest_duration=1
                        )
        sim.generate_full_series(compute_bold=False)
        plot = True
        if plot:
            plt.plot(sim.output['mtime'], sim.output["exc"].T, c='k', lw=2)
            plt.show()

        assert sim.output['exc'].std() > 0.05
        assert np.allclose(sim.output['exc'][0, :500].std(), sim.output['exc'][0, 500:].std(), atol=1e-01)

    def test_single_node_run_with_bold(self):
        self.wc_params.N = 1
        self.wc_params.exc_ext = 2.5
        self.wc_params.sigma_ou = 0
        self.wc_params.Cmat = np.array([[0.]])
        self.wc_params.lengthMat = np.array([[0.]])

        bold_params = {'fix': True, 'normalize_constant': 50}

        sim = WCTaskSim(self.wc_params,
                        bold_params=bold_params,
                        C_rest=self.wc_params.Cmat,
                        D=self.wc_params.lengthMat,
                        fMRI_T=200,
                        rest_duration=60,
                        chunksize=2,
                        rest_before=0,
                        rest_after=0,
                        output_type="exc")
        sim.generate_full_series(compute_bold=True)

        bw_model = BWBoldModel(1, sim.mTime, **bold_params)
        BOLD = bw_model.run(sim.output["exc"])

        plot = True
        if plot:
            plt.subplot(311);
            plt.plot(sim.output['mtime'], sim.output["exc"].T, c='k', lw=2)
            plt.subplot(312);
            plt.plot(sim.output['mtime'][600:], sim.output["BOLD"][0, 600:], c='k', lw=2)
            plt.subplot(313);
            plt.plot(sim.output['mtime'][600:], BOLD.T[600:], c='k', lw=2)

            plt.show()

        assert (BOLD - sim.output['BOLD']).mean() < BOLD.mean() / 100

    def test_generate_full_series_solo_rest(self, setup_neurolib_n3):

        sim = setup_neurolib_n3
        sim.generate_full_series(compute_bold=False)
        res_signal, res_time = resample_signal(sim.output['mtime'], sim.output['exc'], 0.01, 0.1)

        #look at the results of downsampling, could be very controversary
        plot = True
        if plot:
            plt.subplot(121);
            plt.plot(sim.output['mtime'], sim.output['exc'][1, :])
            plt.subplot(122);
            plt.plot(res_time, res_signal[1, :])

            plt.show()
        assert "exc" in sim.output.keys()
        assert sim.output["exc"].shape[0] == 3
        assert sim.output["exc"].dtype == np.float64
        assert round(np.diff(sim.output['mtime']).max(), 4) == round(np.diff(sim.output['mtime']).min(), 5)
        assert len(sim.output["mtime"]) == sim.output["exc"].shape[1]

    def test_compare_chunk_with_neurolib_output(self, setup_neurolib_n3):

        sim = setup_neurolib_n3
        sim.wc_params['exc_init'] = np.array([[0.01347526], [0.02480961], [0.03695609]])
        sim.wc_params['inh_init'] = np.array([[0.0097476], [0.00898726], [0.02694131]])

        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(sim.C_rest,
                                                                        duration=2000,
                                                                        update_inits=True)

        plot = True
        if plot:
            plt.subplot(3, 1, 1)
            plt.plot(excs[0, ::100].T, c='k', lw=2)
            plt.subplot(3, 1, 2)
            plt.plot(excs[1, ::100].T, c='k', lw=2)
            plt.subplot(3, 1, 3)
            plt.plot(excs[2, ::100].T, c='k', lw=2)
            plt.show()
        expected_exc_ou = np.array([-0.00079995, -0.00567414, 0.00234893])
        expected_inh_ou = np.array([-0.0040344, 0.00344725, -0.00241103])
        expected_exc_last = np.array([[0.01833568], [0.01233852], [0.02929641]])

        assert np.allclose(excs[:, -1:], expected_exc_last, atol=1e-06)
        assert np.allclose(inh_ou, expected_inh_ou, atol=1e-06)
        assert np.allclose(exc_ou, expected_exc_ou, atol=1e-06)

    def test_compare_series_with_neurolib_output(self, setup_neurolib_n3):

        #check how chuncks join to each others
        sim = setup_neurolib_n3
        start_time = time.perf_counter()
        sim.generate_full_series(compute_bold=False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        plot = True
        if plot:
            plt.subplot(3, 1, 1)
            plt.plot(sim.output['mtime'], sim.output["exc"][0, :].T, c='k', lw=2)
            plt.subplot(3, 1, 2)
            plt.plot(sim.output['mtime'], sim.output["exc"][1, :].T, c='k', lw=2)
            plt.subplot(3, 1, 3)
            plt.plot(sim.output['mtime'], sim.output["exc"][2, :].T, c='k', lw=2)
            plt.show()
        mean_exc_first = sim.output['exc'][:, :200].mean(axis=1)
        mean_exc_last = sim.output['exc'][:, 200:].mean(axis=1)
        std_exc_first = sim.output['exc'][:, :200].std(axis=1)
        std_exc_last = sim.output['exc'][:, 200:].std(axis=1)

        assert np.allclose(mean_exc_first, mean_exc_last, atol=1e-01)
        assert np.allclose(std_exc_first, std_exc_last, atol=1e-01)
        assert elapsed_time < 3

    def test_compare_series_with_neurolib_output_bold(self, setup_neurolib_n3):

        sim = setup_neurolib_n3
        sim.bold_params = {'fix': True, 'normalize_constant': 50}
        start_time = time.perf_counter()
        sim.generate_full_series(compute_bold=True)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        plot = True
        if plot:
            plt.subplot(3, 1, 1)
            plt.plot(sim.output['mtime'], sim.output["BOLD"][0, :].T, c='k', lw=2)
            plt.subplot(3, 1, 2)
            plt.plot(sim.output['mtime'], sim.output["BOLD"][1, :].T, c='k', lw=2)
            plt.subplot(3, 1, 3)
            plt.plot(sim.output['mtime'], sim.output["BOLD"][2, :].T, c='k', lw=2)
            plt.show()

        assert True

    #check performance for real C with N=100


class TestWCTaskSimFull:

    @pytest.fixture(autouse=True)
    def setup(self):
        num_regions = 30
        num_modules = 3
        X = 0.9
        Z = 0.5
        rest_factors = np.array([[X, 0.1, 0.1],
                                 [0.1, X, 0.1],
                                 [0.1, 0.1, X]])
        taskA_factors = np.array([[X, Z, 0.1],
                                  [Z, X, 0.1],
                                  [0.1, 0.1, X]])
        taskB_factors = np.array([[X, 0.1, Z],
                                  [0.1, X, 0.1],
                                  [Z, 0.1, X]])

        c_rest = generate_synaptic_weights_matrices(num_regions,
                                                    num_modules,
                                                    factors=rest_factors,
                                                    sigma=0.1)
        c_task_a = generate_synaptic_weights_matrices(num_regions,
                                                      num_modules,
                                                      factors=taskA_factors,
                                                      sigma=0.1)
        c_task_b = generate_synaptic_weights_matrices(num_regions,
                                                      num_modules,
                                                      factors=taskB_factors,
                                                      sigma=0.1)
        self.D = np.ones((num_regions, num_regions)) * 250
        np.fill_diagonal(self.D, 0)
        norm_type = "cols"
        self.C_rest = normalize(c_rest, norm_type=norm_type)
        c_task_a = normalize(c_task_a, norm_type=norm_type)
        c_task_b = normalize(c_task_b, norm_type=norm_type)
        self.C_task_dict = {"task_A": c_task_a, "task_B": c_task_b}
        #self.mat_path = '../data/smallSOTs_1.5s_duration.mat'

    def test_init(self):
        wc_block = WCTaskSim(C_rest=self.C_rest,
                             C_task_dict=self.C_task_dict,
                             D=self.D,
                             duration_list=3,
                             rest_before=6,
                             rest_after=6)

        assert len(wc_block.onset_time_list) == 2
        assert len(np.unique(wc_block.task_name_list)) == 1

    def test_generate_series(self):

        TR = 2
        fMRI_T = 16
        rest_before = 6
        wc_block = WCTaskSim(C_rest=self.C_rest,
                             C_task_dict=self.C_task_dict,
                             D=self.D,
                             TR=TR,
                             fMRI_T=fMRI_T,
                             rest_before=rest_before,
                             rest_after=0,
                             rest_duration=10,
                             duration_list=3)
        wc_block.generate_full_series(compute_bold=False)

        plot = True
        if plot:
            tune = int(rest_before / (TR / fMRI_T))
            plt.plot(wc_block.output['syn_act'][0:10, tune:].T)
            plt.plot(wc_block.output['syn_act'][10:20, tune:].T)
            plt.show()

        assert wc_block.output['syn_act'].shape[0] == 30
        assert wc_block.mTime == TR / fMRI_T

    def test_generate_full_series_one_task(self):
        rest_before = 6
        wc_block = WCTaskSim(C_rest=self.C_rest,
                             C_task_dict=self.C_task_dict,
                             D=self.D,
                             rest_before=rest_before,
                             rest_after=10,
                             onset_time_list=[0],
                             duration_list=10,
                             chunksize=3,
                             task_name_list=["task_A"],
                             )

        wc_block.generate_full_series(compute_bold=False)

        plot = True
        if plot:
            tune = int(rest_before / wc_block.mTime)
            plt.subplot(121);
            plt.plot(wc_block.output['mtime'][tune:], wc_block.output['syn_act'][0:10, tune:].T)
            plt.subplot(122);
            plt.plot(wc_block.output['mtime'][tune:], wc_block.output['syn_act'][10:20, tune:].T)
            plt.show()

        assert wc_block.output['syn_act'].shape[0] == 30

    def test_generate_full_series_two_task(self):
        rest_before = 20
        wc_block = WCTaskSim(C_rest=self.C_rest,
                             C_task_dict=self.C_task_dict,
                             D=self.D,
                             rest_before=rest_before,
                             rest_after=10,
                             onset_time_list=[0.01, 3.76, 6.01, 8.13],
                             duration_list=[1, 1.5, 1, 1],
                             chunksize=3,
                             task_name_list=["task_A", "task_B", "task_A", "task_B"],
                             )

        wc_block.generate_full_series(compute_bold=True)

        plot = True
        if plot:
            tune = int(rest_before / wc_block.mTime)
            plt.subplot(121);
            plt.plot(wc_block.output['mtime'][tune:], wc_block.output['syn_act'][0:10, tune:].T)
            plt.subplot(122);
            plt.plot(wc_block.output['mtime'][tune:], wc_block.output['BOLD'][10:20, tune:].T)
            plt.show()

        assert wc_block.output['syn_act'].shape[0] == 30
