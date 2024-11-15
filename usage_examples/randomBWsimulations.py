from tmfc_simulation.read_utils import generate_sw_matrices_from_mat
from tmfc_simulation.boldIntegration import simulateBOLD
from scipy.signal import decimate
from tmfc_simulation.wilson_cowan_task_simulation import WCTaskSim
import yaml
import numpy as np
import os
from scipy import io
import matplotlib.pyplot as plt



def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config_02_EVENT.yaml')
mat_path = config['weigth_matrix']['mat_path']
N_ROIs = config['weigth_matrix']['N_ROIs'] # number of brain regions
SIGMA = config['weigth_matrix']['sigma'] # standard deviation of Gaussian distribution for synaptic weights generation
NORM_TYPE = config['weigth_matrix']['norm_type'] # normalize by columns (all inputs to each region summed to one)
NUM_MODULES = config['weigth_matrix']['num_modules'] # number of functional modules
GEN_TYPE = config['weigth_matrix']['gen_type']

TR = config['time_resolution']['TR']
fMRI_T = config['time_resolution']['fMRI_T']
micro_dt = TR/fMRI_T

# Synaptic weight matrix block generation
from tmfc_simulation.read_utils import generate_sw_matrices_from_mat

Wij_rest, Wij_task_dict = generate_sw_matrices_from_mat(mat_path, N_ROIs, num_modules=NUM_MODULES,
                                 sigma=SIGMA, norm_type=NORM_TYPE, gen_type = GEN_TYPE)
Wij_task_list = list(Wij_task_dict.values())
Wij_list = [Wij_rest]+Wij_task_list

Wij_task_dict["Rest"] = Wij_rest

# title_list =['Rest']+ list(Wij_task_dict.keys())
# fig, axs = plt.subplots(1, 3, figsize = (15,4))
#
# #list of possible colormaps here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
# for i in [0,1,2]:
#     im = axs[i].imshow(Wij_list[i], cmap='hot', vmin = 0, vmax=0.05); axs[i].set_title(title_list[i]);
#     fig.colorbar(im, ax = axs[i], fraction=0.046, pad=0.04);
# fig.tight_layout()
# plt.show()

# Balloon-Windkessel haemodynamic model


k = config['bw_params']['k']['value']
var_k = config['bw_params']['k']['variance']

gamma = config['bw_params']['gamma']['value']
var_gamma = config['bw_params']['gamma']['variance']

tau = config['bw_params']['tau']['value']
var_tau = config['bw_params']['tau']['variance']

alpha = config['bw_params']['alpha']['value']
var_alpha = config['bw_params']['alpha']['variance']

rho = config['bw_params']['rho']['value']
var_rho = config['bw_params']['rho']['variance']

K = np.random.normal(k, np.sqrt(var_k), size=(N_ROIs,))
Gamma = np.random.normal(gamma, np.sqrt(var_gamma), size=(N_ROIs,))
Tau = np.random.normal(tau, np.sqrt(var_tau), size=(N_ROIs,))
Alpha = np.random.normal(alpha, np.sqrt(var_alpha), size=(N_ROIs,))
Rho = np.random.normal(rho, np.sqrt(var_rho), size=(N_ROIs,))
bw_params = {"k":K, "gamma":Gamma, "tau":Tau, "alpha": Alpha, "rho": Rho}

K = np.random.normal(k, np.sqrt(var_k), size=(N_ROIs,))
Gamma = np.random.normal(gamma, np.sqrt(var_gamma), size=(N_ROIs,))
Tau = np.random.normal(tau, np.sqrt(var_tau), size=(N_ROIs,))
Alpha = np.random.normal(alpha, np.sqrt(var_alpha), size=(N_ROIs,))
Rho = np.random.normal(rho, np.sqrt(var_rho), size=(N_ROIs,))
bw_params = {"k":K, "gamma":Gamma, "tau":Tau, "alpha": Alpha, "rho": Rho}

#Generate activation as a delta function at first moment in each node


#time length (in seconds)
length = config['bw_params']['length']
onset = 0
dt = 10e-4
duration = config['bw_params']['duration_list']
bw_normalize_max = config['bw_params']['duration_list']
activation = np.zeros((N_ROIs, int(length / dt)))
activation[:, int(onset / dt):int((onset + duration) / dt)] = 1

activation = bw_normalize_max * activation
HRF, X, F, Q, V = simulateBOLD(activation,
                                dt,
                                alpha=Alpha,
                                rho=Rho,
                                tau=Tau,
                                k=K,
                                gamma=Gamma,
                                fix=True)
time = np.linspace(0, length, int(length/dt))
downsampling_factor = int(micro_dt / dt)
micro_time = time[::downsampling_factor]
micro_HRF = HRF[:, ::downsampling_factor]

HRF_dict = {**bw_params, "bw_normalize_max": bw_normalize_max,
             "micro_dt": micro_dt, "micro_time": micro_time, "micro_HRF": micro_HRF, "TR": TR }

# plt.plot(micro_time, micro_BOLD.T)
# plt.title('Bold response for different nodes')
#
# plt.show()

## Task simulation part

#could be moved to yaml
wc_params = {'exc_ext': 0.758, # excitatory background drive
             'K_gl': 2.63, # global coupling parameter
             'sigma_ou': 0.0035, # std of the Ornstein-Uhlenbeck noise
             'inh_ext': 0, # inhibitory background drive
             'tau_ou': 5, # [ms] timescale of the Ornstein-Uhlenbeck noise
             'a_exc': 1.5, # slope of excitatory response function
             'a_inh': 1.5, # slope of inhibitory response function
             'c_excexc': 16, # E-to-E synaptic weight
             'c_excinh': 15, # E-to-I synaptic weight
             'c_inhexc': 12, # I-to-E synaptic weight
             'c_inhinh': 3, # I-to-I synaptic weight
             'mu_exc': 3,  # position of maximum slope of excitatory response function
             'mu_inh': 3,  # Position of maximum slope of inhibitory response function
             'tau_exc': 2.5, # excitatory time constant
             'tau_inh': 3.75, # inhibitory time constant
             'signalV': 10 # signal transmission speed between areas [m/s]
            }
sim_parameters = {"delay": config['sym_parameters']['delay'],
                  "rest_before": config['sym_parameters']['rest_before'],
                  "first_duration": config['sym_parameters']['first_duration'],
                  "last_duration": config['sym_parameters']['last_duration']}
activity = True


wc_sim = WCTaskSim.from_matlab_structure(mat_path, sigma=SIGMA,
                                         norm_type=NORM_TYPE,
                                         num_modules=NUM_MODULES,
                                         num_regions= N_ROIs,
                                         **wc_params, **sim_parameters,
                                         gen_type=GEN_TYPE)
wc_sim.generate_full_series(TR=TR, activity=True, a_s_rate=micro_dt, clear_raw=True,
                            normalize_max=config['sym_parameters']['normalize_max'],
                            output_activation='syn_act',
                            **bw_params)

task_BOLD = wc_sim.BOLD.transpose()
task_BOLD_time = wc_sim.t_BOLD
syn_act = wc_sim.activity['sa_series']
t_syn_act = wc_sim.activity['t']

task_dict = {"task_BOLD": task_BOLD,
             "task_BOLD_time": task_BOLD_time,
             "syn_act": syn_act,
             "t_syn_act": t_syn_act}

io.savemat("../results/test.mat", {**task_dict, **Wij_task_dict, **HRF_dict})
