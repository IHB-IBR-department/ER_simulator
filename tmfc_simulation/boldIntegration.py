import numpy as np
import numba
from typing import Optional, Union
import numpy.typing as npt
from tmfc_simulation.functions import resample_signal



class BWBoldModel(object):
    """ Balloon-Windkessel BOLD simulator class.
    BOLD simulation results are saved in t_BOLD, BOLD instance attributes.
    Only bold signal generation, no downsampling inside the class
    """

    def __init__(self,
                 N: int,
                 dt: float,
                 normalize_constant: float = 1,
                 rho: Optional[Union[float, tuple[float, float], list[float, float], npt.NDArray[np.float64]]] = None,
                 alpha: Optional[Union[float, tuple[float, float], list[float, float], npt.NDArray[np.float64]]] = None,
                 gamma: Optional[Union[float, tuple[float, float], list[float, float], npt.NDArray[np.float64]]] = None,
                 k: Optional[Union[float, tuple[float, float], list[float, float], npt.NDArray[np.float64]]] = None,
                 tau: Optional[Union[float, tuple[float, float], list[float, float], npt.NDArray[np.float64]]] = None,
                 seed: Optional[Union[int, np.random.RandomState]] = None,
                 fix: bool = False,
                 length: int = 32):
        """
        Initialize the BWBoldModel.

        :param N: Number of regions/nodes.
        :type N: int
        :param dt: Time step (s).
        :type dt: float
        :param normalize_constant: Normalization constant, defaults to 1.
        :type normalize_constant: float, optional
        :param rho: Resting net oxygen extraction. Can be a single float, a tuple/list of two floats (mean and variance), or a numpy array of shape (N,) representing values for each region. Defaults to None, which uses default values with variance.
        :type rho: Optional[Union[float, tuple[float, float], list[float, float], npt.NDArray[np.float64]]], optional
        :param alpha: Grubb's vessel stiffness exponent. Can be a single float, a tuple/list of two floats (mean and variance), or a numpy array of shape (N,) representing values for each region. Defaults to None, which uses default values with variance.
        :type alpha: Optional[Union[float, tuple[float, float], list[float, float],  npt.NDArray[np.float64]]]], optional
        :param gamma: Rate constant for autoregulatory feedback by blood flow. Can be a single float, a tuple/list of two floats (mean and variance), or a numpy array of shape (N,) representing values for each region. Defaults to None, which uses default values with variance.
        :type gamma: Optional[Union[float, tuple[float, float], list[float, float],  npt.NDArray[np.float64]]]], optional
        :param k: Rate of signal decay. Can be a single float, a tuple/list of two floats (mean and variance), or a numpy array of shape (N,) representing values for each region. Defaults to None, which uses default values with variance.
        :type k: Optional[Union[float, tuple[float, float], list[float, float],  npt.NDArray[np.float64]]]], optional
        :param tau: Transit time. Can be a single float, a tuple/list of two floats (mean and variance), or a numpy array of shape (N,) representing values for each region. Defaults to None, which uses default values with variance.
        :type tau: Optional[Union[float, tuple[float, float], list[float, float],  npt.NDArray[np.float64]]]], optional
        """
        self.N = N
        self.length = length
        self.dt = dt
        self.normalize_constant = normalize_constant
        self.seed = seed
        self._init_parameters(rho, alpha, gamma, k, tau, fix=fix)

        # initialize BOLD model variables
        self.X_BOLD = np.zeros((N,))
        # Vasso dilatory signal
        self.F_BOLD = np.ones((N,))
        # Blood flow
        self.Q_BOLD = np.ones((N,))
        # Deoxyhemoglobin
        self.V_BOLD = np.ones((N,))
        # Blood volume

    def run(self, activity):
        # Compute the BOLD signal for the chunk
        activity = activity*self.normalize_constant
        BOLD, self.X_BOLD, self.F_BOLD, self.Q_BOLD, self.V_BOLD = simulateBOLD(
            activity,
            self.dt,
            rho=self.rho,
            alpha=self.alpha,
            gamma=self.gamma,
            k=self.k,
            tau=self.tau,
            X=self.X_BOLD,
            F=self.F_BOLD,
            Q=self.Q_BOLD,
            V=self.V_BOLD,
        )
        return BOLD

    def run_on_impulse(self,
                       down_dt=0.125):
        # Create a simple test signal
        dt = self.dt
        # time length (in seconds)
        length = self.length
        onset = 0
        duration = 10e-4
        activation = np.zeros((self.N, int(length / dt)))
        time = np.linspace(0, length, int(length / dt))
        activation[:, int(onset / dt):int((onset + duration) / dt)] = 1
        BOLD = self.run(activation)
        resampled_BOLD, time = resample_signal(time, BOLD, dt, down_dt)

        return resampled_BOLD, time

    def save_parameters(self, filepath, s_type='mat'):
        """Saves the model parameters rho, alpha, gamma, k, tau.

        :param filepath: The path to the output file.
        :type filepath: str
        :param s_type: The desired file format. Can be 'mat', 'npy', or 'rp5'.
        :type s_type: str, optional
        :raises ValueError: if ``format`` is not 'mat', 'npy', or 'rp5'
        """

        params = {
            'rho': self.rho,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'k': self.k,
            'tau': self.tau,
        }

        if s_type == 'mat':
            import scipy.io as sio
            sio.savemat(filepath, params)
        elif s_type == 'npy':
            np.save(filepath, params)
        elif s_type == 'hdf5':
            try:
                import h5py
                with h5py.File(filepath, 'w') as hf:
                    for key, value in params.items():
                        hf.create_dataset(key, data=value)
            except ImportError:
                print("h5py not installed, skipping saving parameters in HDF5 format.")

        else:
            raise ValueError("Unsupported s_type. Choose from 'mat', 'npy', or 'hdf5'.")

    def _init_parameters(self, rho, alpha, gamma, k, tau, fix=True):

        np.random.seed(self.seed)
        def process_param(param, default_mean, default_var, param_name):
            """Helper function to process parameters."""
            voxelCounts = np.ones((self.N,))
            voxelCountsSqrtInv = 1 / np.sqrt(voxelCounts)

            if param is not None:
                assert isinstance(param, (float, list, tuple, np.ndarray)), \
                    f"{param_name} must be float or list or tuple of two floats or N floats"

            if param is None:
                mean, var = default_mean, default_var
            elif isinstance(param, float):
                mean, var = param, default_var
            elif len(param) == 2:
                mean, var = param[0], param[1]
            elif len(param) == self.N:
                mean, var = np.array(param), None
            else:
                raise ValueError(f"{param_name} must be float or of two or N floats")

            return mean * np.ones((self.N,)) if var is None or fix \
                else np.random.normal(mean, np.sqrt(var) * voxelCountsSqrtInv, size=(self.N,))

        # Process each parameter using the helper function
        self.rho = process_param(rho, 0.34, 0.0024, "rho")
        self.alpha = process_param(alpha, 0.32, 0.0015, "alpha")
        self.gamma = process_param(gamma, 0.41, 0.002, "gamma")
        self.k = process_param(k, 0.65, 0.015, "k")
        self.tau = process_param(tau, 0.98, 0.0568, "tau")


def simulateBOLD(Z,
                 dt,
                 rho,
                 alpha,
                 gamma,
                 k,
                 tau,
                 X=None,
                 F=None,
                 Q=None,
                 V=None,
                 V0=None,
                 k1_mul=None,
                 k2=None,
                 k3_mul=None):
    """ Adopted function from neurolib, added parameters for the shape of bold activation to the argument,
    the only difference
    https://github.com/neurolib-dev/neurolib/blob/master/neurolib/models/bold/timeIntegration.py

    Simulate BOLD activity using the Balloon-Windkessel model.
    See Friston 2000, Friston 2003 and Deco 2013 for reference on how the BOLD signal is simulated.
    The returned BOLD signal should be downsampled to be comparable to a recorded fMRI signal.

    :param Z: Synaptic activity or other signal to convolve
    :type Z: numpy.ndarray
    :param dt: dt of input activity in s
    :type dt: float
    :param voxelCounts: Number of voxels in each region (not used yet!)
    :type voxelCounts: numpy.ndarray
    :param X: Initial values of Vasodilatory signal, defaults to None
    :type X: numpy.ndarray, optional
    :param F: Initial values of Blood flow, defaults to None
    :type F: numpy.ndarray, optional
    :param Q: Initial values of Deoxyhemoglobin, defaults to None
    :type Q: numpy.ndarray, optional
    :param V: Initial values of Blood volume, defaults to None
    :type V: numpy.ndarray, optional

    :return: BOLD, X, F, Q, V
    :rtype: (numpy.ndarray,)


    """

    N = np.shape(Z)[0]

    # Balloon-Windkessel model parameters (from Friston 2003):
    # Friston paper: Nonlinear responses in fMRI: The balloon model, Volterra kernels, and other hemodynamics
    # Note: the distribution of each Balloon-Windkessel models parameters are given per voxel
    # Since we usually average the empirical fMRI of each voxel for a given area, the standard
    # deviation of the gaussian distribution should be divided by the number of voxels in each area
    # voxelCountsSqrtInv = 1 / np.sqrt(voxelCounts)
    #
    # See Friston 2003, Table 1 mean values and variances:
    # rho     = np.random.normal(0.34, np.sqrt(0.0024) / np.sqrt( sum(voxelCounts) ) )    # Capillary resting net oxygen extraction
    # alpha   = np.random.normal(0.32, np.sqrt(0.0015) / np.sqrt( sum(voxelCounts) ) )    # Grubb's vessel stiffness exponent
    # V0      = 0.02
    # k1      = 7 * rho
    # k2      = 2.0
    # k3      = 2 * rho - 0.2
    # Gamma   = np.random.normal(0.41 * np.ones(N), np.sqrt(0.002) * voxelCountsSqrtInv)   # Rate constant for autoregulatory feedback by blood flow
    # K       = np.random.normal(0.65 * np.ones(N), np.sqrt(0.015) * voxelCountsSqrtInv)   # Vasodilatory signal decay
    # Tau     = np.random.normal(0.98 * np.ones(N), np.sqrt(0.0568) * voxelCountsSqrtInv)   # Transit time
    #
    # If no voxel counts are given, we can use scalar values for each region's parameter:

    # Capillary resting net oxygen extraction (dimensionless), E_0 in Friston2000

    # Resting blood volume fraction (dimensionless)
    V0 = 0.02 if V0 is None else V0

    K1 = 7 * rho if k1_mul is None else k1_mul * rho  # (dimensionless)
    k2 = 2.0 if k2 is None else k2  # (dimensionless)
    K3 = 2 * rho - 0.2 if k3_mul is None else k3_mul * rho - 0.2  # (dimensionless)

    # rate of flow dependent elimination

    # initialize state variables
    # NOTE: We need to use np.copy() because these variables
    # will be overwritten later and numba doesn't like to do that
    # with anything that was defined outside the scope of the @njit'ed function
    X = np.zeros((N,)) if X is None else np.copy(X)  # Vasso dilatory signal
    F = np.ones((N,)) if F is None else np.copy(F)  # Blood flow
    Q = np.ones((N,)) if Q is None else np.copy(Q)  # Deoxyhemoglobin
    V = np.ones((N,)) if V is None else np.copy(V)  # Blood volume

    BOLD = np.zeros(np.shape(Z))
    BOLD, X, F, Q, V = integrateBOLD_numba(BOLD, X, Q, F, V, Z, dt, N,
                                           rho, alpha, V0, K1, k2, K3, gamma, k, tau)
    return BOLD, X, F, Q, V


@numba.njit
def integrateBOLD_numba(BOLD, X, Q, F, V, Z, dt, N, Rho, Alpha, V0, K1, k2, K3, Gamma, K, Tau):
    """Integrate the Balloon-Windkessel model.

    Reference:

    Friston et al. (2000), Nonlinear responses in fMRI: The balloon model, Volterra kernels, and other hemodynamics.
    Friston et al. (2003), Dynamic causal modeling

    Variable names in Friston2000:
    X = x1, Q = x4, V = x3, F = x2

    Friston2003: see Equation (3)

    NOTE: A very small constant EPS is added to F to avoid F become too small / negative
    and cause a floating point error in EQ. Q due to the exponent **(1 / F[j])

    """

    EPS = 1e-120  # epsilon for softening

    for i in range(len(Z[0, :])):  # loop over all timesteps
        # component-wise loop for compatibilty with numba
        for j in range(N):  # loop over all areas
            X[j] = X[j] + dt * (Z[j, i] - K[j] * X[j] - Gamma[j] * (F[j] - 1))
            Q[j] = Q[j] + dt / Tau[j] * (
                    F[j] / Rho[j] * (1 - (1 - Rho[j]) ** (1 / F[j])) - Q[j] * V[j] ** (1 / Alpha[j] - 1))
            V[j] = V[j] + dt / Tau[j] * (F[j] - V[j] ** (1 / Alpha[j]))
            F[j] = F[j] + dt * X[j]

            F[j] = max(F[j], EPS)

            BOLD[j, i] = V0 * (K1[j] * (1 - Q[j]) + k2 * (1 - Q[j] / V[j]) + K3[j] * (1 - V[j]))
    return BOLD, X, F, Q, V
