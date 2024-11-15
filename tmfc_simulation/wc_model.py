import numpy as np
from typing import Optional, Union, Dict
import numpy.typing as npt
from tmfc_simulation.load_wc_params import load_wc_params
from tmfc_simulation.time_integration import time_integration
from tmfc_simulation.functions import resample_signal


class WCTaskSim:
    """Class for simulation block design fMRI with WC model,
       where used the own matrix synaptic matrix for each state
       with predefined onset times
       """

    description = "Wilson-Cowan model"
    init_vars = ["exc_init", "inh_init", "exc_ou", "inh_ou"]

    def __init__(self, params=None, output_type="syn_act", seed=None):
        """
        Parameters
        Args:
            params:
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
        self.seed = seed
        self.output['mtime'] = None
        if params is None:
            #if params is None, they will load from yaml file
            params = load_wc_params(Cmat=None)

        self.params = params

        self.inits = dict()
        for init_var in self.init_vars:
            self.inits[init_var] = None
        # time in ms
        self.inits["last_t"] = 0
        self.delay_matrix = params["lengthMat"]

    @property
    def delay_matrix(self):
        return self._D

    @delay_matrix.setter
    def delay_matrix(self, D):
        self._D = D
        self.params['lengthMat'] = D
        self.max_delay = self.getMaxDelay(D)

    def _init_simulation(self,
                         D,
                         C_rest: npt.NDArray[np.float64],
                         C_task_dict: Optional[Dict[str, npt.NDArray[np.float64]]]=None,
                         onset_time_list: Optional[list[float]] = None,
                         task_name_list: Optional[list[str]] = None,
                         duration_list: Union[list, float] = 3,
                         rest_before: float = 6,
                         rest_after: float = 6,
                         TR=2,
                         fMRI_T=16,
                         chunksize=2):
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
        self.burn_in_time = rest_before
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
            unique_tasks = np.unique(task_name_list)
            assert all(task in C_task_dict for task in unique_tasks), \
                "All task names must be keys in C_task_dict"

        if onset_time_list is None:
            self.onset_time_list = [-rest_before, duration_list]
            self.task_name_list = ["Rest"]
        else:
            assert len(onset_time_list) == len(task_name_list), \
                "You need to provide onset time for each task in list of task names"
            if isinstance(duration_list, float):
                duration_list = [duration_list] * len(task_name_list)
            assert len(duration_list) == len(task_name_list), \
                "You need to provide duration_list for each task in list"
            onset_time_list, task_name_list = self.complete_onsets_with_rest(onset_time_list,
                                                                             task_name_list,
                                                                             duration_list,
                                                                             rest_before=rest_before,
                                                                             rest_after=rest_after)
            self.onset_time_list = onset_time_list
            self.task_name_list = task_name_list
            self.C_task_dict = C_task_dict

    def generate_full_series(self,
                             D,
                             C_rest: npt.NDArray[np.float64],
                             C_task_dict: Optional[Dict[str, npt.NDArray[np.float64]]]=None,
                             onset_time_list: Optional[list[float]] = None,
                             task_name_list: Optional[list[str]] = None,
                             duration: Union[list, float] = 3,
                             rest_before: float = 6,
                             rest_after: float = 6,
                             TR=2,
                             fMRI_T=16,
                             chunksize=2):
        self._init_simulation(D, C_rest, C_task_dict, onset_time_list, task_name_list,
                              duration, rest_before, rest_after, TR, fMRI_T, chunksize)
        self.inits["last_t"] = self.onset_time_list[0]*1000

        lastT = self.onset_time_list[0]


        while self.onset_time_list[-1] >= lastT + 1e-6:
            out, mtime = self._generate_chunk_with_onsets(start_time=lastT)
            if self.output_type != "all":
                self.output[self.output_type] =  np.hstack(
                    [self.output[self.output_type], out]) if self.output[self.output_type] is not None else out
            else:
                self.output["syn_act"] = np.hstack([self.output["syn_act"],
                                                    out[0]]) if self.output[self.output_type] is not None else out[0]
                self.output["exc"] = np.hstack([self.output["exc"],
                                                out[1]]) if self.output[self.output_type] is not None else out[1]
                self.output["inh"] = np.hstack([self.output["inh"],
                                                out[2]]) if self.output[self.output_type] is not None else out[2]

            self.output["mtime"] = np.hstack([self.output["mtime"],
                                              mtime]) if self.output['mtime'] is not None else mtime
            lastT = self.inits["last_t"]/1000

    def _generate_chunk_with_onsets(self,
                                    start_time):
        """
        Generate chunk on basis of provided onset time list and task_name list and inits
        Args:
            start_time: in seconds
            inits:

        Returns:

        """
        self.inits["last_t"] = start_time*1000 #update start time in init in ms
        start_time_idx = max(np.searchsorted(self.onset_time_list, start_time, side='left') - 1, 0)
        task_name = self.task_name_list[start_time_idx]
        if task_name == "Rest":
            Cmat = self.C_rest
        else:
            Cmat = self.C_task_dict[task_name]
        end_time = self.onset_time_list[start_time_idx + 1]
        # duration_list in ms, but input parameters in seconds
        duration = 1000 * min(end_time - start_time, self.chunksize)
        last_t = self.inits["last_t"]
        self.inits["last_t"] = last_t+duration
        t, excs, inhs, exc_ou, inh_ou = self._generate_full_single_chunk(Cmat, duration, update_inits=True)
        excs = excs[:, self.max_delay+1:]
        inhs = inhs[:, self.max_delay+1:]
        #compute syn_act if need
        if self.output_type in ["all", "syn_act"]:
            syn_act = self.generate_neuronal_oscill(excs, inhs, Cmat)
        if self.output_type == "syn_act":
            out, time = resample_signal(t, syn_act, self.params['dt'] / 1000, self.mTime)
        elif self.output_type == "exc":
            out, time = resample_signal(t, excs, self.params['dt'] / 1000, self.mTime)
        elif self.output_type == "all":
            out_excs,_ = resample_signal(t, excs, self.params['dt'] / 1000, self.mTime)
            out_syn,_ = resample_signal(t, syn_act, self.params['dt'] / 1000, self.mTime)
            out_inh, time = resample_signal(t, inhs, self.params['dt'] / 1000, self.mTime)

            out = (out_syn, out_excs, out_inh), time

        time = time + last_t
        return out, time/1000

    def _generate_full_single_chunk(self,
                                    Cmat: npt.NDArray[np.float64],
                                    duration: float,
                                    update_inits: bool = False):
        self.params['Cmat'] = Cmat
        self.params['duration_list'] = duration
        for init_var in self.init_vars:
            if self.inits[init_var] is not None:
                self.params[init_var] = self.inits[init_var].copy()
        t, excs, inhs, exc_ou, inh_ou = time_integration(params=self.params)
        if update_inits:
            startind = int(self.max_delay + 1)
            self.inits["exc_init"] = excs[:, -startind:]
            self.inits["inh_init"] = inhs[:, -startind:]
            self.inits["exc_ou"] = exc_ou
            self.inits["inh_ou"] = inh_ou
        return t, excs, inhs, exc_ou, inh_ou

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
        dt = self.params.get("dt")

        if Dmat is not None:
            # divide Dmat by signalV
            signalV = self.params.get("signalV") or 0
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

    def complete_onsets_with_rest(self,
                                  onset_time_list: list[float],
                                  task_name_list: list[str],
                                  duration_list: list[float],
                                  rest_before: float,
                                  rest_after: float):
        """ Completes a list of task onsets with rest periods.

        Args:
        onset_time_list: List of task onset times in seconds.
        task_name_list: List of task names corresponding to the onsets.
        duration_list: List of task durations in seconds.
        rest_before: Duration of rest period before the first task, in seconds.
        rest_after: Duration of rest period after the last task, in seconds.

        Returns:
        Tuple containing the updated onset times and task names, including rest periods.
        """

        if rest_before > 0:
            updated_onset_time_list = [-rest_before]
            updated_task_name_list = ["Rest"]
        else:
            updated_onset_time_list = []
            updated_task_name_list = []

        current_time = 0.0

        for onset, task_name, duration in zip(onset_time_list, task_name_list, duration_list):
            if onset > current_time:
                updated_onset_time_list.append(current_time)
                updated_task_name_list.append("Rest")
                updated_onset_time_list.append(onset)
                updated_task_name_list.append(task_name)
                current_time = onset + duration
            else:
                updated_onset_time_list.append(onset)
                updated_task_name_list.append(task_name)
                current_time = onset + duration

        updated_onset_time_list.append(current_time)
        if rest_after > 0:
            updated_task_name_list.append("Rest")
            updated_onset_time_list.append(current_time + rest_after)

        return updated_onset_time_list, updated_task_name_list

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
        ee = self.params['c_excexc']
        ei = self.params['c_excinh']
        ie = self.params['c_inhexc']
        ii = self.params['c_inhinh']

        syn_act = ee * exc + ei * exc + ie * inh + ii * inh + Cmat @ exc
        return syn_act

