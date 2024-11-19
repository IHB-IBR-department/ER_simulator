import pytest
import cProfile
import time
import pstats
from tmfc_simulation.wc_model import WCTaskSim
from tmfc_simulation.load_wc_params import load_wc_params
import numpy as np
from matplotlib import pyplot as plt
from tmfc_simulation.functions import resample_signal




class TestWCTaskSim:

    @pytest.fixture(autouse=True)  # autouse=True applies this fixture to all tests in the class
    def setup_params(self):
        """Sets up default parameters for WCTaskSim tests."""

        # Use load_wc_params to load default parameters or provide a test YAML
        self.params = load_wc_params(config_file="../tmfc_simulation/wc_params.yaml")  # Replace with your config

        # Create a dummy Cmat and Dmat if needed
        N = 3  # Example number of nodes
        np.random.seed(26)  # Example seed value
        self.params.Cmat = np.abs(np.random.rand(N, N))
        # Zero diagonal
        np.fill_diagonal(self.params.Cmat, 0)
        self.params.lengthMat = 250*np.ones((N,N))
        # Zero diagonal
        np.fill_diagonal(self.params.lengthMat, 0)
        self.params.N = N # set N from generated Cmat

    @pytest.fixture(autouse=True)
    def setup_one_event(self):
        onset_time_list = [2]
        tasks_name_list = ["Task1"]
        Cmat = 2*np.abs(np.random.rand(3, 3))
        duration_list = [3]
        return onset_time_list, tasks_name_list, duration_list, {"Task1": Cmat}

    @pytest.fixture(autouse=True)
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



    def test_complete_onsets_with_rest_empty_input(self):
        sim = WCTaskSim(self.params)  # Pass params to the constructor
        onsets, tasks = sim.complete_onsets_with_rest([], [], [], 1, 2)
        assert onsets == [-1, 0.0, 2]
        assert tasks == ["Rest", "Rest"]

    def test_complete_onsets_with_rest_single_task(self):
        sim = WCTaskSim(self.params)
        onsets, tasks = sim.complete_onsets_with_rest([2], ["Task1"], [3], 1, 2)
        assert onsets == [-1, 0.0, 2, 5, 7]
        assert tasks == ["Rest", "Rest", "Task1", "Rest"]

    def test_complete_onsets_with_rest_two_task(self):
        sim = WCTaskSim(self.params)
        onsets, tasks = sim.complete_onsets_with_rest([2, 5], ["Task1", "Task2"],
                                                      [3, 2],0, 0)
        assert onsets == [0.0, 2, 5, 7]
        assert tasks == ["Rest", "Task1", "Task2"]


    def test_generate_full_single_chunk_without_update(self):
        sim = WCTaskSim(self.params)
        duration = 10
        Cmat = self.params.Cmat
        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(Cmat, duration)
        assert Cmat.shape[0] == excs.shape[0]
        assert len(t) == excs.shape[1]

    def test_generate_full_single_chunk_with_update(self):
        sim = WCTaskSim(self.params)
        duration = 1
        Cmat = self.params.Cmat
        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(Cmat,
                                                                        duration*1000,
                                                                        update_inits=True)

        assert np.allclose(sim.inits['exc_init'], excs[:, -sim.max_delay-1:])
        assert np.allclose(sim.inits['exc_ou'], exc_ou)
        assert np.allclose(sim.inits['inh_ou'], inh_ou)
        assert Cmat.shape[0] == excs.shape[0]
        assert len(t) == excs.shape[1]

    def test_generate_chunk_with_onsets_solo_rest(self):
        sim = WCTaskSim(self.params)
        sim._init_simulation(self.params.lengthMat, self.params.Cmat)
        start_time = 0
        out, time = sim._generate_chunk_with_onsets(start_time)
        assert len(time) == len(out[0])

    def test_generate_chunk_with_onsets_one_event(self, setup_one_event):
        onset_time_list, tasks_name_list, duration_list, C_task_dict = setup_one_event
        sim = WCTaskSim(self.params)
        sim._init_simulation(self.params.lengthMat,
                             self.params.Cmat,
                             C_task_dict,
                             onset_time_list,
                             tasks_name_list,
                             duration_list,
                             rest_before=0,
                             rest_after=0,
                             TR=2,
                             fMRI_T=20)
        start_time = 0.95
        out, time = sim._generate_chunk_with_onsets(start_time)
        assert time[-1] - start_time <= sim.chunksize
        assert sim.inits['last_t'] == 2000.
        assert len(time) == len(out[0])
        start_time = 4
        out, time = sim._generate_chunk_with_onsets(start_time)
        assert time[-1] - start_time <= sim.chunksize

    def test_single_node_run(self):
        self.params.N = 1
        self.params.exc_ext = 2.5
        self.params.sigma_ou = 0
        self.params.Cmat = np.array([[0.]])
        self.params.lengthMat = np.array([[0.]])

        sim = WCTaskSim(self.params, output_type="exc")
        sim.generate_full_series(self.params.lengthMat,
                                 self.params.Cmat,
                                 fMRI_T=2000,
                                 duration=1,
                                 chunksize=2,
                                 rest_before=0,
                                 rest_after=0)
        plot = False
        if plot:
            plt.plot(sim.output['mtime'], sim.output["exc"].T, c='k', lw=2)
            plt.show()

        assert sim.output['exc'].std() > 0.05
        assert np.allclose(sim.output['exc'][0, :500].std(), sim.output['exc'][0, 500:].std(), atol=1e-01)

    def test_generate_full_series_solo_rest(self):

        sim = WCTaskSim(self.params, output_type="all")
        sim.generate_full_series(self.params.lengthMat,
                                 self.params.Cmat,
                                 TR=2,
                                 fMRI_T=200,
                                 duration=10,
                                 rest_before=0,
                                 rest_after=3)
        res_signal, res_time = resample_signal(sim.output['mtime'], sim.output['exc'], 0.01, 0.1)

       #look at the results of downsampling, could be very controversary
        plot = True
        if plot:
            plt.subplot(121); plt.plot(sim.output['mtime'], sim.output['exc'][1, :])
            plt.subplot(122); plt.plot(res_time, res_signal[1, :])

            plt.show()
        assert "syn_act" in sim.output.keys()
        assert sim.output["syn_act"].shape[0] == 3
        assert sim.output["syn_act"].dtype == np.float64
        assert round(np.diff(sim.output['mtime']).max(), 4) == round(np.diff(sim.output['mtime']).min(), 5)
        assert len(sim.output["mtime"]) == sim.output["syn_act"].shape[1]

    def test_generate_full_series_profile(self):  # create test method
        sim = WCTaskSim(self.params)

        profiler = cProfile.Profile()
        profiler.enable()
        sim.generate_full_series(self.params.lengthMat,
                                 self.params.Cmat,
                                 duration=2,
                                 rest_before=3,
                                 rest_after=3)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')

        # Print to console (less useful in tests) or save stats to a file
        stats.dump_stats("profile_results.prof")  # save output to file

    def test_compare_chunk_with_neurolib_output(self):

        seed = 26
        N = 3
        Cmat = np.array([[0., 0.51939148, 0.76829766],
                        [0.78922074, 0., 0.18792139],
                        [0.26950525, 0.49619214, 0.]])
        np.fill_diagonal(Cmat, 0)
        sim = WCTaskSim(self.params, output_type="exc", seed=seed)
        sim.params.exc_init = np.array([[0.01347526], [0.02480961], [0.03695609]])
        sim.params.inh_init = np.array([[0.0097476 ],[0.00898726],[0.02694131]])
        sim.params.lengthMat = 250 * np.ones((N, N))
        np.fill_diagonal(sim.params.lengthMat, 0)



        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(Cmat,
                                                                         duration=2000,
                                                                         update_inits=True)

        plot = False
        if plot:
            plt.subplot(3, 1, 1)
            plt.plot( excs[0, ::100].T, c='k', lw=2)
            plt.subplot(3, 1, 2)
            plt.plot(excs[1, ::100].T, c='k', lw=2)
            plt.subplot(3, 1, 3)
            plt.plot( excs[2, ::100].T, c='k', lw=2)
            plt.show()
        expected_exc_ou = np.array([-0.00079995, -0.00567414,  0.00234893])
        expected_inh_ou = np.array([-0.0040344 ,  0.00344725, -0.00241103])
        expected_exc_last = np.array([[0.01833568], [0.01233852], [0.02929641]])

        assert np.allclose(excs[:, -1:], expected_exc_last, atol=1e-06)
        assert np.allclose(inh_ou, expected_inh_ou, atol=1e-06)
        assert np.allclose(exc_ou, expected_exc_ou, atol=1e-06)

    def test_compare_series_with_neurolib_output(self):

        #check how chuncks join to each others
        seed = 26
        N = 3
        Cmat = np.array([[0., 0.51939148, 0.76829766],
                         [0.78922074, 0., 0.18792139],
                         [0.26950525, 0.49619214, 0.]])
        np.fill_diagonal(Cmat, 0)
        sim = WCTaskSim(self.params, output_type="all", seed=seed)
        sim.params.exc_init = np.array([[0.01347526], [0.02480961], [0.03695609]])
        sim.params.inh_init = np.array([[0.0097476], [0.00898726], [0.02694131]])
        sim.params.lengthMat = 250 * np.ones((N, N))
        np.fill_diagonal(sim.params.lengthMat, 0)
        start_time = time.perf_counter()
        sim.generate_full_series(sim.params.lengthMat,
                                 Cmat,
                                 TR=2,
                                 fMRI_T=20,
                                 duration=40,
                                 rest_before=0,
                                 rest_after=0)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        plot = False
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







    def test_generate_full_series_performance(self):  # create test method
        seed = 26
        sim = WCTaskSim(self.params, output_type="exc", seed=seed)
        sim.params.exc_init = 0.05 * np.random.uniform(0, 1, (sim.params.N, 1))
        sim.params.inh_init = 0.05 * np.random.uniform(0, 1, (sim.params.N, 1))
        start_time = time.perf_counter()
        N = 3
        np.random.seed(seed)
        Cmat = 2*np.abs(np.random.rand(N, N))
        np.fill_diagonal(Cmat, 0)
        lengthMat = 250 * np.ones((N, N))
        np.fill_diagonal(lengthMat, 0)
        # Zero diagonal
        np.fill_diagonal(lengthMat, 0)
        sim.generate_full_series(lengthMat,
                                 Cmat,
                                 TR=2,
                                 fMRI_T=20,
                                 duration=10,
                                 rest_before=0,
                                 rest_after=0)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        plt.subplot(3, 1, 1)
        plt.plot(sim.output['mtime'], sim.output["exc"][0,:].T, c='k', lw=2)
        plt.subplot(3, 1, 2)
        plt.plot(sim.output['mtime'], sim.output["exc"][1,:].T, c='k', lw=2)
        plt.subplot(3, 1, 3)
        plt.plot(sim.output['mtime'], sim.output["exc"][2, :].T, c='k', lw=2)
        plt.show()
        # Optionally assert that the time is below a certain threshold
        assert elapsed_time < 3  # Example: Assert execution time is under 3s

    #check performance for real C with N=100