import numpy as np
import yaml
from scipy import io
from typing import Optional, Union, Dict
import numpy.typing as npt
from er_simulator.load_wc_params import load_wc_params
from er_simulator.time_integration import time_integration
from er_simulator.functions import resample_signal
from er_simulator.boldIntegration import BWBoldModel
from .read_utils import generate_sw_matrices_from_mat
from .read_utils import read_onsets_from_mat
from .task_utils import (create_task_design_activation,
                         create_reg_activations,
                         create_activations_per_module)


class WCTaskSim:
    """Class for simulation block design fMRI with WC model,
       where used the own matrix synaptic matrix for each state
       with predefined onset times
       """

    description = "Wilson-Cowan model"
    init_vars = ["exc_init", "inh_init", "exc_ou", "inh_ou"]

    def __init__(self,
                 wc_params=None,
                 bold_params=None,
                 C_rest: npt.NDArray[np.float64] = None,
                 D: npt.NDArray[np.float64] = None,
                 C_task_dict: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
                 seed=None,
                 onset_time_list: Optional[list[float]] = None,
                 task_name_list: Optional[list[str]] = None,
                 duration_list: Union[list, float] = 3,
                 rest_before: float = 6,
                 rest_after: float = 6,
                 rest_duration: float = 900,
                 TR=2,
                 fMRI_T=16,
                 chunksize=2,
                 output_type="syn_act",
                 ):
        """
        Parameters
        Args:
            wc_params:
            output_type: possible values "syn_act", "exc", "inh", "all"
        """
        assert output_type in ["syn_act", "exc", "inh", "all"]
        self.output_type = output_type
        self.output = dict()
        if output_type == "all":
            self.output: Dict[str, Optional[npt.NDArray[np.float64]]] = {
                "exc": None,
                "inh": None,
                "syn_act": None,
            }
        else:
            self.output[output_type] = None

        self.output['mtime'] = None
        self.output['BOLD'] = None
        self.output['TR_time'] = None
        self.output['BOLD_TR'] = None
        if wc_params is None:
            wc_params = {}
        if bold_params is None:
            bold_params = {}
        assert isinstance(wc_params, dict)
        assert isinstance(bold_params, dict)
        wc_params['seed'] = seed
        bold_params['seed'] = seed

        self.wc_params = self._set_wc_params_dict(wc_params)
        self.bold_params = self._set_bold_dict(bold_params)
        self.seed = seed

        self.compute_bold = None

        self.inits: Dict[str, Optional[Union[float, npt.NDArray[np.float64]]]] = {}
        for init_var in self.init_vars:
            self.inits[init_var] = None
        # time in ms
        self.delay_matrix = D
        self.rest_duration = rest_duration
        self._init_simulation(D,
                              C_rest,
                              C_task_dict,
                              onset_time_list,
                              task_name_list,
                              duration_list,
                              rest_before,
                              rest_after,
                              rest_duration,
                              TR,
                              fMRI_T,
                              chunksize)
        self.boldModel = BWBoldModel(self.num_regions, self.wc_params['dt'] * 1e-03, **self.bold_params)
        self.config_file = None

    @classmethod
    def from_config(cls, config_file):
        """
        Creates a WCTaskSim instance from a configuration file.

        Args:
            config_file (str or Path): Path to the YAML configuration file.

        Returns:
            WCTaskSim: An instance of WCTaskSim initialized with parameters from the config file.
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        wc_params = config.get("wc_params", None)
        wc_params = cls._set_wc_params_dict(wc_params)
        bold_params = config.get("bold_params", None)
        bold_params = cls._set_bold_dict(bold_params)
        seed = config.get("seed", None)
        bold_params["seed"] = seed
        wc_params['seed'] = seed

        rest_before = config['sym_parameters'].get("rest_before", 6)
        rest_after = config['sym_parameters'].get("rest_after", 6)
        TR = config['sym_parameters'].get("TR", 2)
        fMRI_T = config['sym_parameters'].get("fMRI_T", 16)
        chunksize = config['sym_parameters'].get("chunksize", 2)
        output_type = config['sym_parameters'].get("output_type", "syn_act")
        rest_duration = config['sym_parameters'].get("rest_duration", 900)
        delay = config['sym_parameters'].get("delay", 250)

        mat_path = config['weight_matrix'].get("mat_path", None)
        #npy_path = config['weight_matrix'].get("npy_path", None)
        num_regions = config['weight_matrix'].get("num_regions", 30)
        sigma = config['weight_matrix'].get("sigma", 0.1)
        num_modules = config['weight_matrix'].get("num_modules", 3)
        num_regions_per_modules = config['weight_matrix'].get("num_regions_per_modules", None)
        gen_type = config['weight_matrix'].get("gen_type", 'simple_prod')
        norm_type = config['weight_matrix'].get("norm_type", 'cols')

        D = np.ones((num_regions, num_regions)) * delay

        if mat_path:
            C_rest, C_task_dict = generate_sw_matrices_from_mat(mat_path,
                                                                num_regions,
                                                                num_modules,
                                                                num_regions_per_modules,
                                                                sigma=sigma,
                                                                gen_type=gen_type,
                                                                norm_type=norm_type)
            onset_time_list, task_name_list, duration_list = read_onsets_from_mat(mat_path)

        else:
            C_rest, C_task_dict = None, None
            onset_time_list, task_name_list, duration_list = None, None, None

        return cls(wc_params=wc_params,
                   bold_params=bold_params,
                   seed=seed,
                   D=D,
                   C_rest=C_rest,
                   C_task_dict=C_task_dict,
                   onset_time_list=onset_time_list,
                   task_name_list=task_name_list,
                   duration_list=duration_list,
                   rest_before=rest_before,
                   rest_after=rest_after,
                   rest_duration=rest_duration,
                   TR=TR,
                   fMRI_T=fMRI_T,
                   chunksize=chunksize,
                   output_type=output_type)

    @property
    def num_regions(self):
        if self.C_rest is None:
            return 1
        else:
            return len(self.C_rest)

    @property
    def delay_matrix(self):
        return self._D

    @delay_matrix.setter
    def delay_matrix(self, D):
        self._D = D
        self.wc_params['lengthMat'] = D
        self.max_delay = self.getMaxDelay(D)

    @property
    def task_params(self):
        task_params = {}
        task_params["onset_time_list"] = self.onset_time_list
        task_params["C_rest"] = self.C_rest
        task_params["C_task_dict"] = self.C_task_dict
        task_params["D"] = self.delay_matrix
        task_params["task_name_list"] = self.task_name_list
        task_params["duration_list"] = self.duration_list
        task_params["rest_before"] = self.rest_before
        task_params["rest_after"] = self.rest_after
        return task_params

    @task_params.setter
    def task_params(self,
                   task_params):
        task_params_keys = ["C_rest", "C_task_dict", "D", "onset_time_list", "task_name_list", "duration_list",
                            "rest_before", "rest_duration", "rest_after", "TR", "fMRI_T", "chunksize"]

        for key in task_params_keys:
            if key not in task_params:
                if key == "D":
                    task_params[key] = self.delay_matrix
                else:
                    task_params[key] = getattr(self, key)

        upd_task_params = {k: v for k, v in task_params.items() if k in task_params_keys}
        if "rest_completed" in task_params.keys():
            upd_task_params["rest_completed"] = task_params["rest_completed"]


        self._init_simulation(**task_params)

    def generate_rest_series(self, keep_task_params=True, rest_duration=None, compute_bold=True):
        """Generate only rest series and save output to dict, rewrite all outputs

        :param keep_task_params:
        :param rest_duration:
        :param compute_bold:
        :return:
        """

        if keep_task_params:
            task_params = self.task_params
            task_params["rest_completed"] = True
        if rest_duration is None:
            rest_duration = self.rest_duration
        rest_params = {"onset_time_list": None, "task_name_list": None, "rest_duration": rest_duration}
        self.task_params = rest_params
        self.generate_full_series(compute_bold=compute_bold)
        if keep_task_params:
            output = self.output.copy()
            self.task_params = task_params
            return output
        else:
            return self.output





    def _init_simulation(self,
                         D,
                         C_rest: npt.NDArray[np.float64],
                         C_task_dict: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
                         onset_time_list: Optional[list[float]] = None,
                         task_name_list: Optional[list[str]] = None,
                         duration_list: Union[list, float] = 3,
                         rest_before: float = 6,
                         rest_after: float = 6,
                         rest_duration: float = 900,
                         TR=2,
                         fMRI_T=16,
                         chunksize=2,
                         rest_completed=False):
        """
        Initialize task dependent simulation with Wilson-Cowan model. So,
        in this case coupling matrix changes with the time. If you need to simulate only rest,
        leave parameters onset_time_list and task_name_list as None
        Args:
            D:
            C_rest:
            C_task_dict:
            onset_time_list:
            task_name_list:
            duration_list:
            rest_before:
            TR:
            fMRI_T:
            chunksize:

        Returns:

        """

        self.delay_matrix = D
        self.C_rest = C_rest
        self.rest_before = rest_before
        self.rest_after = rest_after
        self.rest_duration = rest_duration
        self.TR = TR
        self.fMRI_T = fMRI_T
        self.mTime = TR / fMRI_T  #microtimebin in seconds
        self.chunksize = chunksize

        #check only rest option
        if C_task_dict is None:
            assert onset_time_list is None, "Only rest will be generated, onset list should be None"
            assert task_name_list is None, "No task matrix, task name list should be None"
            assert isinstance(duration_list, (float, int)), "duration_list should be a float for only rest generation"
        else:
            unique_tasks = list(set(np.unique(task_name_list))-{'Rest'})
            assert all(task in C_task_dict for task in unique_tasks) or (task_name_list is None), \
                "All task names must be keys in C_task_dict"

        if isinstance(duration_list, (float, int)):
            if task_name_list is not None:
                duration_list = [duration_list] * len(task_name_list)

        if not rest_completed:
            onset_time_list, task_name_list = self.complete_onsets_with_rest(onset_time_list,
                                                                         task_name_list,
                                                                         duration_list,
                                                                         rest_before=self.rest_before,
                                                                         rest_after=self.rest_after,
                                                                         rest_duration=self.rest_duration
                                                                         )
        self.onset_time_list = onset_time_list
        self.task_name_list = task_name_list
        self.duration_list = duration_list
        self.C_task_dict = C_task_dict

        for init_var in self.init_vars:
            self.inits[init_var] = None
        for key in self.output.keys():
            self.output[key] = None
        self.inits["last_t"] = self.onset_time_list[0] * 1000

    def generate_full_series(self,
                             compute_bold: bool = True):
        """

        :param compute_bold:
        :return:
        """
        #clear all outputs and inits
        for init_var in self.init_vars:
            self.inits[init_var] = None
        for key in self.output.keys():
            self.output[key] = None

        lastT = self.onset_time_list[0]

        self.compute_bold = compute_bold

        while self.onset_time_list[-1] >= lastT + 1e-6:
            out_dict = self._generate_chunk_with_onsets(start_time=lastT)
            for key in out_dict.keys():
                self.output[key] = np.hstack(
                    [self.output[key], out_dict[key]]) if self.output[key] is not None else out_dict[key]

            lastT = self.inits["last_t"] / 1000
        if compute_bold:
            self.output["BOLD_TR"], self.output["TR_time"] = resample_signal(self.output['mtime'],
                                                                             self.output['BOLD'],
                                                                             self.mTime,
                                                                             self.TR)
            self.boldModel.reset_state()

    def get_task_block(self, task_name, out_tname, out_sname, skip_time=6):
        indexes = [i for i, value in enumerate(self.task_name_list) if value == task_name]
        times = [(self.onset_time_list[idx], self.onset_time_list[idx + 1]) for idx in indexes]
        time_idxs = [
            (np.searchsorted(self.output[out_tname], time[0], side='right'),
             np.searchsorted(self.output[out_tname], time[1], side='right') - 1)
            for time in times]
        skip_idx = np.searchsorted(self.output[out_tname],
                                   skip_time, side='right') - np.searchsorted(self.output[out_tname],
                                                                              0, side='right')
        series_list = []
        for idxs in time_idxs:
            series_list.append(self.output[out_sname][:, idxs[0] + skip_idx: idxs[1]])
        concat_series = np.concatenate(series_list, axis=1)
        return concat_series

    def _generate_chunk_with_onsets(self,
                                    start_time):
        """
        Generate chunk on basis of provided onset time list and task_name list and inits
        Args:
            start_time: in seconds
            inits:

        Returns:

        """
        # update start time in init in ms
        self.inits["last_t"] = start_time * 1000
        start_time_idx = max(np.searchsorted(self.onset_time_list, start_time, side='right') - 1, 0)
        task_name = self.task_name_list[start_time_idx]
        if task_name == "Rest":
            Cmat = self.C_rest
        else:
            Cmat = self.C_task_dict[task_name]
        end_time = self.onset_time_list[start_time_idx + 1]
        # duration_list in ms, but input parameters in seconds
        duration = 1000 * min(end_time - start_time, self.chunksize)
        last_t = self.inits["last_t"]
        self.inits["last_t"] = last_t + duration
        t, excs, inhs, exc_ou, inh_ou = self._generate_full_single_chunk(Cmat, duration, update_inits=True)

        out_dict = {}
        if self.output_type in ["all", "syn_act"]:
            syn_act = self.generate_neuronal_oscill(excs, inhs, Cmat)
        if self.output_type == "syn_act":
            out_dict["syn_act"], time = resample_signal(t, syn_act, self.wc_params['dt'] / 1000, self.mTime,
                                                        last_time=last_t / 1000)
        elif self.output_type == "exc":
            out_dict["exc"], time = resample_signal(t, excs, self.wc_params['dt'] / 1000, self.mTime,
                                                    last_time=last_t / 1000)
        elif self.output_type == "all":
            out_dict["exc"], _ = resample_signal(t, excs, self.wc_params['dt'] / 1000, self.mTime,
                                                 last_time=last_t / 1000)
            out_dict["inh"], _ = resample_signal(t, inhs, self.wc_params['dt'] / 1000, self.mTime,
                                                 last_time=last_t / 1000)
            out_dict["syn_act"], time = resample_signal(t, syn_act, self.wc_params['dt'] / 1000, self.mTime,
                                                        last_time=last_t / 1000)

        if self.compute_bold:

            if self.output_type == "exc":
                activation = excs
            elif self.output_type == "inh":
                activation = inhs
            else:
                activation = syn_act
            BOLD = self.boldModel.run(activation)
            out_dict["BOLD"], _ = resample_signal(t, BOLD, self.wc_params['dt'] / 1000, self.mTime,
                                                  last_time=last_t / 1000)
        time = time + last_t
        out_dict["mtime"] = time / 1000
        return out_dict

    def _generate_full_single_chunk(self,
                                    Cmat: npt.NDArray[np.float64],
                                    duration: float,
                                    update_inits: bool = False):
        np.fill_diagonal(Cmat, 0)
        self.wc_params['Cmat'] = Cmat
        self.wc_params['duration'] = duration
        for init_var in self.init_vars:
            if self.inits[init_var] is not None:
                self.wc_params[init_var] = self.inits[init_var].copy()
        t, excs, inhs, exc_ou, inh_ou = time_integration(params=self.wc_params)
        startind = int(self.max_delay + 1)

        if update_inits:
            self.inits["exc_init"] = excs[:, -startind:]
            self.inits["inh_init"] = inhs[:, -startind:]
            self.inits["exc_ou"] = exc_ou
            self.inits["inh_ou"] = inh_ou
        return t, excs[:, startind:], inhs[:, startind:], exc_ou, inh_ou

    def getMaxDelay(self, Dmat: Optional[npt.NDArray[np.float64]] = None) -> int:
        """Computes the maximum delay of the model. This function should be overloaded
        if the model has internal delays (additional to delay between nodes defined by Dmat)
        such as the delay between an excitatory and inhibitory population within each brain area.
        If this function is not overloaded, the maximum delay is assumed to be defined from the
        global delay matrix `Dmat`.

        Note: Maxmimum delay is given in units of dt.

        :return: maxmimum delay of the model in units of dt
        :rtype: int
        """
        dt = self.wc_params.get("dt")

        if Dmat is not None:
            # divide Dmat by signalV
            signalV = self.wc_params.get("signalV") or 0
            if signalV > 0:
                Dmat = Dmat / signalV
            else:
                # if signalV is 0, eliminate delays
                Dmat = Dmat * 0.0

        # only if Dmat and dt exist, a global max delay can be computed
        if Dmat is not None and dt is not None:
            Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
            max_global_delay = int(np.amax(Dmat_ndt))
        else:
            max_global_delay = 0
        return max_global_delay

    @staticmethod
    def complete_onsets_with_rest(onset_time_list: Optional[list[float]],
                                  task_name_list: Optional[list[str]],
                                  duration_list: Optional[list[float]],
                                  rest_before: Optional[float],
                                  rest_after: Optional[float],
                                  rest_duration: Optional[float] = 6):
        """ Completes a list of task onsets with rest periods or only rest series if no tasks

        Args:
        onset_time_list: List of task onset times in seconds.
        task_name_list: List of task names corresponding to the onsets.
        duration_list: List of task durations in seconds.
        rest_before: Duration of rest period before the first task, in seconds.
        rest_after: Duration of rest period after the last task, in seconds.

        Returns:
        Tuple containing the updated onset times and task names, including rest periods.
        """

        def _check_solo_rest(task_name_list):
            if task_name_list is None:
                return True
            elif len(task_name_list) == 0:
                return True
            elif (len(task_name_list) == 1) and (task_name_list[0] == 'Rest'):
                return True
            else:
                return False

        def _check_tasks_lists(onset_time_list, task_name_list, duration_list):
            assert len(onset_time_list) == len(task_name_list) == len(duration_list), \
                "onset_time_list and task_name_list and duration_list must have same length"
            if len(set(task_name_list).difference('Rest')) >= 1:
                return True

        def _complete_solo_rest(rest_before, rest_duration):
            onset_time_list = [-rest_before, rest_duration]
            task_name_list = ['Rest']
            return onset_time_list, task_name_list

        def _complete_tasks_lists(onset_time_list, task_name_list, duration_list,
                                  rest_before, rest_after):

            if rest_before > 0:
                updated_onset_time_list = [-rest_before]
                updated_task_name_list = ["Rest"]
            elif onset_time_list[0] > 0:
                updated_onset_time_list = [0]
                updated_task_name_list = ["Rest"]
            else:
                updated_onset_time_list = []
                updated_task_name_list = []

            #current_time = onset_time_list[0]
            for i in range(len(onset_time_list) - 1):
                current_onset = onset_time_list[i]
                next_onset = onset_time_list[i + 1]
                if next_onset - current_onset > duration_list[i]:
                    updated_onset_time_list.extend([current_onset,
                                                    current_onset + duration_list[i]])
                    updated_task_name_list.extend([task_name_list[i], 'Rest'])
                else:
                    updated_onset_time_list.append(current_onset)
                    updated_task_name_list.append(task_name_list[i])
            updated_onset_time_list.append(onset_time_list[-1])
            updated_task_name_list.append(task_name_list[-1])
            if rest_after > 0:
                updated_onset_time_list.extend([onset_time_list[-1] + duration_list[-1],
                                                onset_time_list[-1] + duration_list[-1] + rest_after])
                updated_task_name_list.append('Rest')
            else:
                updated_onset_time_list.append(onset_time_list[-1] + duration_list[-1])
            return updated_onset_time_list, updated_task_name_list

        if _check_solo_rest(task_name_list):
            return _complete_solo_rest(rest_before, rest_duration)
        elif _check_tasks_lists(onset_time_list, task_name_list, duration_list):
            return _complete_tasks_lists(onset_time_list, task_name_list, duration_list,
                                         rest_before, rest_after)
        else:
            raise NotImplementedError("Check your onsets")

    def generate_neuronal_oscill(self,
                                 exc: npt.NDArray[np.float64],
                                 inh: npt.NDArray[np.float64],
                                 Cmat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Computes the integrated synaptic activity (ISA) without temporal integration (e.g., over 50ms).

        This implementation follows the equation described in Horwitz and Tagamets (1999):
        ISA = c_excexc * E + c_excinh * E + c_inhexc * I + c_inhinh * I + Cmat @ E

        Where:
            - E represents the excitatory activity.
            - I represents the inhibitory activity.
            - c_excexc, c_excinh, c_inhexc, c_inhinh are coupling parameters within a neural population.
            - Cmat is the connectivity matrix representing the influence of excitatory activity
              from other brain regions.
            - @ denotes matrix multiplication.

        The function calculates the instantaneous synaptic activity based on the provided excitatory and
        inhibitory activity, along with the connectivity matrix and intra-population coupling parameters.
        Args:
            exc: Excitatory neuronal activity (time x regions).
            inh: Inhibitory neuronal activity (time x regions).
            Cmat: Connectivity matrix representing inter-regional influences (regions x regions).
        Returns:
            Integrated synaptic activity (ISA) of the same shape as the input activity (time x regions).
        """
        ee = self.wc_params['c_excexc']
        ei = self.wc_params['c_excinh']
        ie = self.wc_params['c_inhexc']
        ii = self.wc_params['c_inhinh']

        syn_act = ee * exc + ei * exc + ie * inh + ii * inh + Cmat @ exc
        return syn_act

    def generate_coactivation_by_mat(self,
                                     mat_path,
                                     dt=50,
                                     normalize_constant=None,
                                     num_regions_per_modules=None):
        """
        Generate outer activation for each node defined with a tasks and activation info, where
        written which model sensitive to which task. All information defined in mat file
        with the description of  tasks, onsets, durations and activation info for each modules

        Args:
            mat_path (str): path to matfile
            act_scaling (float): scaling factor for hrf function
            **kwargs : kwargs for HRF class

        Returns:

        """

        num_regions = self.num_regions
        rest_before = self.rest_before
        rest_after = self.rest_after
        if dt is None:
            dt = self.mTime
        input_data = io.loadmat(mat_path)
        num_tasks = input_data['onsets'].shape[1]
        onsets_list = []
        activations = []
        durations_list = []
        for i in range(num_tasks):
            onsets_list.append(list(input_data['onsets'][0, i].squeeze()))
            activations.append(input_data['activations'][0, i].squeeze())
            durations_list.append(input_data["durations"][0, i].squeeze())

        box_car_activations = create_task_design_activation(onsets_list,
                                                            durations_list,
                                                            dt=1000 * dt,
                                                            rest_before=rest_before,
                                                            rest_after=rest_after)
        activations_by_module = create_activations_per_module(activations,
                                                              box_car_activations)
        activations_by_regions = create_reg_activations(activations_by_module,
                                                        num_regions,
                                                        num_regions_per_modules)
        assert isinstance(self.boldModel, BWBoldModel), "First you need to init common BW model "
        boldModel = self.boldModel
        boldModel.reset_state(dt)
        BOLD = boldModel.run(activations_by_regions, normalize_constant=normalize_constant)
        time = np.arange(self.onset_time_list[0], self.onset_time_list[-1], dt)
        #res_BOLD, res_time = resample_signal(time, BOLD, dt, self.TR)
        #res_activations = resample_signal(time, activations_by_regions, dt, self.TR)
        return time, activations_by_regions, BOLD

    @staticmethod
    def _set_bold_dict(bold_params: Optional[dict]) -> dict:

        bw_keys = ["normalize_constant", "rho", "alpha", "gamma", "tau", "k", "fix"]
        bold_params_default = {"normalize_constant": 2,
                               "fix": True,
                               "rho": 0.34,
                               "alpha": 0.32,
                               "gamma": 0.41,
                               "tau": 2.5,
                               "k": 0.65}

        if bold_params is None:
            bold_params = {}

        for key in bw_keys:
            if key not in bold_params:
                bold_params[key] = bold_params_default[key]

        assert isinstance(bold_params["normalize_constant"], (float, int)), "Normalizing constant should be a number"
        assert isinstance(bold_params["fix"], bool), "fix constant should be a boolean"
        return {k: v for k, v in bold_params.items() if k in bw_keys}

    @staticmethod
    def _set_wc_params_dict(wc_params: dict) -> dict:
        wc_keys = ["dt", "seed", "tau_exc",
                   "tau_inh", "c_excexc", "c_excinh", "c_inhexc",
                   "c_inhinh", "a_exc", "a_inh", "mu_exc", "mu_inh", "tau_ou", "sigma_ou", "exc_ou_mean",
                   "inh_ou_mean", "exc_ou", "inh_ou", "exc_ext_baseline", "inh_ext_baseline",
                   "exc_ext", "inh_ext", "exc_init", "inh_init", "K_gl", "signalV",
                   "lengthMat", "Cmat"]
        wc_params_default = {"dt": 0.1,
                             "seed": None,
                             "tau_exc": 2.5,
                             "tau_inh": 3.75,
                             "c_excexc": 16,
                             "c_excinh": 15,
                             "c_inhexc": 12,
                             "c_inhinh": 3,
                             "a_exc": 1.5,
                             "a_inh": 1.5,
                             "mu_exc": 3,
                             "mu_inh": 3,
                             "tau_ou": 5.0,
                             "sigma_ou": 0.005,
                             "exc_ou_mean": 0.,
                             "exc_ext_baseline": 0,
                             "inh_ext_baseline": 0,
                             "inh_ext": 0,
                             "exc_ext": 0.75,
                             "inh_ou_mean": 0.,
                             "exc_ou": 0,
                             "inh_ou": 0,
                             "exc_init": None,
                             "inh_init": None,
                             "K_gl": 2.85,
                             "signalV": 10,
                             "lengthMat": None,
                             "Cmat": None}

        if wc_params is None:
            wc_params = {}

        for key in wc_keys:
            if key not in wc_params:
                wc_params[key] = wc_params_default[key]
        return {k: v for k, v in wc_params.items() if k in wc_keys}
