import pytest
import cProfile
import time
import pstats
from tmfc_simulation.wc_model import WCTaskSim
from tmfc_simulation.load_wc_params import load_wc_params
import numpy as np



class TestWCTaskSim:

    @pytest.fixture(autouse=True)  # autouse=True applies this fixture to all tests in the class
    def setup_params(self):
        """Sets up default parameters for WCTaskSim tests."""

        # Use load_wc_params to load default parameters or provide a test YAML
        self.params = load_wc_params(config_file="../tmfc_simulation/wc_params.yaml")  # Replace with your config

        # Create a dummy Cmat and Dmat if needed
        N = 3  # Example number of nodes
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
        assert len(t) == excs.shape[1]-251

    def test_generate_full_single_chunk_with_update(self):
        sim = WCTaskSim(self.params)
        duration = 10
        Cmat = self.params.Cmat
        t, excs, inhs, exc_ou, inh_ou = sim._generate_full_single_chunk(Cmat,
                                                                        duration,
                                                                        update_inits=True)
        assert Cmat.shape[0] == excs.shape[0]
        assert len(t) == excs.shape[1]-251

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
                             duration_list)
        start_time = 3
        out, time = sim._generate_chunk_with_onsets(start_time)
        assert time[-1] - start_time <= sim.chunksize
        assert sim.inits['last_t'] == 5000
        assert len(time) == len(out[0])
        start_time = 4
        out, time = sim._generate_chunk_with_onsets(start_time)
        assert time[-1] - start_time <= sim.chunksize


    def test_generate_full_series_solo_rest(self):

        sim = WCTaskSim(self.params)
        sim.generate_full_series(self.params.lengthMat,
                                 self.params.Cmat,
                                 duration=2,
                                 rest_before=3,
                                 rest_after=3)
        assert "syn_act" in sim.output.keys()
        assert sim.output["syn_act"].shape == (3, 40)
        assert sim.output["syn_act"].dtype == np.float64
        assert len(sim.output["mtime"]) == 40

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

    def test_generate_full_series_performance(self):  # create test method
        sim = WCTaskSim(self.params)
        start_time = time.perf_counter()
        N = 3
        Cmat = 2*np.abs(np.random.rand(N, N))
        np.fill_diagonal(Cmat, 0)
        lengthMat = 250 * np.ones((N, N))
        np.fill_diagonal(lengthMat, 0)
        # Zero diagonal
        np.fill_diagonal(lengthMat, 0)
        sim.generate_full_series(lengthMat,
                                 Cmat,
                                 duration=20,
                                 rest_before=5,
                                 rest_after=0)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        # Optionally assert that the time is below a certain threshold
        assert elapsed_time < 3  # Example: Assert execution time is under 3s

    #check performance for real C with N=100