import yaml
import numpy as np


class dotdict(dict):
    """Dictionary with dot notation access to attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_wc_params(Cmat=None, Dmat=None, seed=None, config_file="wc_params.yaml"):
    """
    Load parameters for the Wilson-Cowan model from a YAML configuration file.

    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths,
                 will be normalized to 1. If not given, then a single node simulation
                 will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together
                 with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional
    :param config_file: Path to the YAML configuration file, defaults to "wc_params.yaml"
    :type config_file: str, optional

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    # Load parameters from the YAML file
    with open(config_file, "r") as file:
        yaml_params = yaml.safe_load(file)

    params = dotdict(yaml_params)

    # Runtime parameters
    params.seed = seed
    np.random.seed(seed)

    # Set up Cmat, lengthMat, and N based on provided matrices
    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))
    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self-connections
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat

    # Initialize random initial states for excitatory and inhibitory variables
    params.exc_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.inh_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))

    # Initialize Ornstein-Uhlenbeck noise state variables
    params.exc_ou = np.zeros((params.N,))
    params.inh_ou = np.zeros((params.N,))

    return params
