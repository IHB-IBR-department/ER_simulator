import numpy as np
import numba
import numpy.typing as npt
from typing import Dict, Tuple
from tmfc_simulation import model_utils as mu


# Adapted from https://github.com/neurolib-dev/neurolib

def time_integration(params: Dict) -> Tuple[npt.NDArray[np.float64], ...]:  # Specify Tuple of NumPy arrays for clarity
    """Performs time integration of a neural mass model.

    This function integrates the dynamic equations of a neural mass model over time using the Euler method.
    It simulates the interactions between excitatory and inhibitory populations in a network of brain regions,
    incorporating noise, external input, and connectivity delays.
    For integration, we use an ODE system where the output_type depends on the initial conditions, which in turn are influenced by the delay matrix.
    To ensure reproducible time series, the initial conditions exc_init and inh_init must remain consistent across the maximum delay horizon (in terms of integration steps).

    Args:
        params: A dictionary containing the model parameters.
               Required keys include: 'dt', 'duration_list', 'seed', 'tau_exc', 'tau_inh',
               'c_excexc', 'c_excinh', 'c_inhexc', 'c_inhinh', 'a_exc', 'a_inh', 'mu_exc', 'mu_inh',
               'tau_ou', 'sigma_ou', 'exc_ou_mean', 'inh_ou_mean', 'Cmat', 'K_gl', 'lengthMat',
               'signalV', 'exc_ou', 'inh_ou', 'exc_ext_baseline', 'inh_ext_baseline', 'exc_ext',
               'inh_ext', 'exc_init', 'inh_init'.

    Returns:
        A tuple containing the integrated excitatory and inhibitory activity (excs, inhs) over time.
        Each element is a NumPy array of shape (N, simulation_steps), where N is the number of nodes and
        simulation_steps is the total number of time steps in the simulation.

    Raises:
        KeyError: If any required parameter is missing from the 'params' dictionary.

    Notes:
      This function uses the Euler method for numerical integration. More advanced integration methods could be implemented for greater accuracy.
      The function assumes that the connectivity matrix ('Cmat') and length matrix ('lengthMat') are provided as NumPy arrays.

    """

    # --- Parameter Extraction and Validation ---
    try:  # Use a try-except block to catch missing parameters

        dt = params["dt"]  # Time step for the Euler intergration (ms)
        duration = params["duration_list"]  # simulation duration_list (ms)
        RNGseed = params["seed"]  # seed for RNG

        # ------------------------------------------------------------------------
        # local parameters
        # See Papadopoulos et al., Relations between large-scale brain connectivity and effects of regional stimulation
        # depend on collective dynamical state, arXiv, 2020
        tau_exc = params["tau_exc"]  #
        tau_inh = params["tau_inh"]  #
        c_excexc = params["c_excexc"]  #
        c_excinh = params["c_excinh"]  #
        c_inhexc = params["c_inhexc"]  #
        c_inhinh = params["c_inhinh"]  #
        a_exc = params["a_exc"]  #
        a_inh = params["a_inh"]  #
        mu_exc = params["mu_exc"]  #
        mu_inh = params["mu_inh"]  #

        # external input parameters:
        # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
        tau_ou = params["tau_ou"]
        # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
        sigma_ou = params["sigma_ou"]
        # Mean external excitatory input (OU process)
        exc_ou_mean = params["exc_ou_mean"]
        # Mean external inhibitory input (OU process)
        inh_ou_mean = params["inh_ou_mean"]

        # ------------------------------------------------------------------------
        # global coupling parameters

        # Connectivity matrix
        # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connection from jth to ith
        Cmat = params["Cmat"]
        N = len(Cmat)  # Number of nodes
        K_gl = params["K_gl"]  # global coupling strength
        # Interareal connection delay
        lengthMat = params["lengthMat"]
        signalV = params["signalV"]
    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e}")

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
    # ------------------------------------------------------------------------
    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    # noise variable
    exc_ou = params["exc_ou"].copy()+np.zeros((N,))
    inh_ou = params["inh_ou"].copy()+np.zeros((N,))

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    excs = np.zeros((N, startind + len(t)))
    inhs = np.zeros((N, startind + len(t)))

    exc_ext_baseline = params["exc_ext_baseline"]
    inh_ext_baseline = params["inh_ext_baseline"]

    exc_ext = mu.adjustArrayShape(params["exc_ext"], excs)
    inh_ext = mu.adjustArrayShape(params["inh_ext"], inhs)

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["exc_init"])[1] == 1:
        exc_init = np.dot(params["exc_init"], np.ones((1, startind)))
        inh_init = np.dot(params["inh_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        exc_init = params["exc_init"][:, -startind:]
        inh_init = params["inh_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    exc_input_d = np.zeros(N)  # delayed input to exc
    inh_input_d = np.zeros(N)  # delayed input to inh (note used)

    np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    excs[:, startind:] = np.random.standard_normal((N, len(t)))
    inhs[:, startind:] = np.random.standard_normal((N, len(t)))

    excs[:, :startind] = exc_init
    inhs[:, :startind] = inh_init

    noise_exc = np.zeros((N,))
    noise_inh = np.zeros((N,))

    # ------------------------------------------------------------------------
    integr_params = [startind,t, dt, sqrt_dt, N, Cmat, K_gl, Dmat_ndt,
                     excs, inhs, exc_input_d, inh_input_d, exc_ext_baseline, inh_ext_baseline,
                     exc_ext, inh_ext, tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh,
                     c_excexc, c_excinh, c_inhexc, c_inhinh, noise_exc, noise_inh, exc_ou, inh_ou,
                     exc_ou_mean, inh_ou_mean, tau_ou, sigma_ou]

    t, excs, inhs, exc_ou, inh_ou = timeIntegration_njit_elementwise(*integr_params)

    return t, excs, inhs, exc_ou, inh_ou


@numba.njit
def timeIntegration_njit_elementwise_v2(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        excs,
        inhs,
        exc_input_d,
        inh_input_d,
        exc_ext_baseline,
        inh_ext_baseline,
        exc_ext,
        inh_ext,
        tau_exc,
        tau_inh,
        a_exc,
        a_inh,
        mu_exc,
        mu_inh,
        c_excexc,
        c_excinh,
        c_inhexc,
        c_inhinh,
        noise_exc,
        noise_inh,
        exc_ou,
        inh_ou,
        exc_ou_mean,
        inh_ou_mean,
        tau_ou,
        sigma_ou,
):
    # Precompute constant terms to avoid redundant calculations in the loop
    tau_exc_inv = 1.0 / tau_exc
    tau_inh_inv = 1.0 / tau_inh
    tau_ou_inv = 1.0 / tau_ou

    for i in range(startind, startind + len(t)):
        for no in range(N):
            # Retrieve the delayed input for the current node
            exc_input_d_no = 0.0
            for l in range(N):
                delay_idx = i - Dmat_ndt[no, l] - 1
                if delay_idx >= 0:  # Ensure valid index
                    exc_input_d_no += K_gl * Cmat[no, l] * excs[l, delay_idx]
            exc_input_d[no] = exc_input_d_no

            # Precompute sigmoid arguments
            S_E_arg = (
                c_excexc * excs[no, i - 1]
                - c_inhexc * inhs[no, i - 1]
                + exc_input_d_no
                + exc_ext_baseline
                + exc_ext[no, i - 1]
            )
            S_I_arg = (
                c_excinh * excs[no, i - 1]
                - c_inhinh * inhs[no, i - 1]
                + inh_ext_baseline
                + inh_ext[no, i - 1]
            )

            # Compute the excitatory and inhibitory right-hand sides
            exc_rhs = tau_exc_inv * (
                -excs[no, i - 1]
                + (1.0 - excs[no, i - 1]) / (1.0 + np.exp(-a_exc * (S_E_arg - mu_exc)))
                + exc_ou[no]
            )
            inh_rhs = tau_inh_inv * (
                -inhs[no, i - 1]
                + (1.0 - inhs[no, i - 1]) / (1.0 + np.exp(-a_inh * (S_I_arg - mu_inh)))
                + inh_ou[no]
            )

            # Update excitatory and inhibitory states with Euler integration
            excs[no, i] = excs[no, i - 1] + dt * exc_rhs
            inhs[no, i] = inhs[no, i - 1] + dt * inh_rhs

            # Clip values to ensure they remain in [0, 1]
            excs[no, i] = min(max(excs[no, i], 0.0), 1.0)
            inhs[no, i] = min(max(inhs[no, i], 0.0), 1.0)

            # Update Ornstein-Uhlenbeck noise processes
            exc_ou[no] += tau_ou_inv * (exc_ou_mean - exc_ou[no]) * dt + sigma_ou * sqrt_dt * noise_exc[no]
            inh_ou[no] += tau_ou_inv * (inh_ou_mean - inh_ou[no]) * dt + sigma_ou * sqrt_dt * noise_inh[no]

    return t, excs, inhs, exc_ou, inh_ou

@numba.njit
def timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        excs,
        inhs,
        exc_input_d,
        inh_input_d,
        exc_ext_baseline,
        inh_ext_baseline,
        exc_ext,
        inh_ext,
        tau_exc,
        tau_inh,
        a_exc,
        a_inh,
        mu_exc,
        mu_inh,
        c_excexc,
        c_excinh,
        c_inhexc,
        c_inhinh,
        noise_exc,
        noise_inh,
        exc_ou,
        inh_ou,
        exc_ou_mean,
        inh_ou_mean,
        tau_ou,
        sigma_ou,
):
    ### integrate ODE system:

    def S_E(x):
        return 1.0 / (1.0 + np.exp(-a_exc * (x - mu_exc)))

    def S_I(x):
        return 1.0 / (1.0 + np.exp(-a_inh * (x - mu_inh)))

    for i in range(startind, startind + len(t)):
        # loop through all the nodes
        for no in range(N):
            # To save memory, noise is saved in the activity array
            noise_exc[no] = excs[no, i]
            noise_inh[no] = inhs[no, i]

            # delayed input to each node
            exc_input_d[no] = 0

            for l in range(N):
                exc_input_d[no] += K_gl * Cmat[no, l] * (excs[l, i - Dmat_ndt[no, l] - 1])

            # Wilson-Cowan model
            exc_rhs = (
                    1
                    / tau_exc
                    * (
                            -excs[no, i - 1]
                            + (1 - excs[no, i - 1])
                            * S_E(
                        c_excexc * excs[no, i - 1]  # input from within the excitatory population
                        - c_inhexc * inhs[no, i - 1]  # input from the inhibitory population
                        + exc_input_d[no]  # input from other nodes
                        + exc_ext_baseline  # baseline external input (static)
                        + exc_ext[no, i - 1]  # time-dependent external input
                    )
                            + exc_ou[no]  # ou noise
                    )
            )
            inh_rhs = (
                    1
                    / tau_inh
                    * (
                            -inhs[no, i - 1]
                            + (1 - inhs[no, i - 1])
                            * S_I(
                        c_excinh * excs[no, i - 1]  # input from the excitatory population
                        - c_inhinh * inhs[no, i - 1]  # input from within the inhibitory population
                        + inh_ext_baseline  # baseline external input (static)
                        + inh_ext[no, i - 1]  # time-dependent external input
                    )
                            + inh_ou[no]  # ou noise
                    )
            )

            # Euler integration
            excs[no, i] = excs[no, i - 1] + dt * exc_rhs
            inhs[no, i] = inhs[no, i - 1] + dt * inh_rhs

            # make sure e and i variables do not exceed 1 (can only happen with noise)
            if excs[no, i] > 1.0:
                excs[no, i] = 1.0
            if excs[no, i] < 0.0:
                excs[no, i] = 0.0

            if inhs[no, i] > 1.0:
                inhs[no, i] = 1.0
            if inhs[no, i] < 0.0:
                inhs[no, i] = 0.0

            # Ornstein-Uhlenbeck process
            exc_ou[no] = (
                    exc_ou[no] + (exc_ou_mean - exc_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[no]
            )  # mV/ms
            inh_ou[no] = (
                    inh_ou[no] + (inh_ou_mean - inh_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[no]
            )  # mV/ms

    return t, excs, inhs, exc_ou, inh_ou

