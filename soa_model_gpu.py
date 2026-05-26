import math
from scipy.constants import elementary_charge as q  # carga del electron
from scipy.constants import h
from scipy.constants import c as c0
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
from optimization_gpu import cost, count_nan


WAVELENGTH = 1.55e-6

# SOA PARAMETERS:

#     # Carrier dynamics
#     tau_sp: float         # Spontaneous lifetime [s]
#     n0: float             # Transparency carrier density [m^-3]

#     # Optical
#     Gamma: float          # Confinement factor
#     a: float              # Differential gain [m^2]
#     alpha_int: float      # Internal loss [m^-1]
#     vg: float             # Group velocity [m/s]

#     # Geometry
#     W: float              # Width [m]
#     d: float              # Active layer thickness [m]
#     L: float              # Length [m]

@cuda.jit(device=True)
def C1(I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    return (
        I * tau_sp * Gamma * a
        / (q * W * d * L)
        - n0 * Gamma * a
        - alpha_int
    )

@cuda.jit(device=True)
def C2(tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    return (
        tau_sp
        * Gamma
        * vg
        * a
    )

@cuda.jit(device=True)
def G_small_signal(I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    return math.exp(C1(I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L) * L)

@cuda.jit(device=True)
def G_inflection(S0, I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    return C1(I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)/C2(tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)/S0/alpha_int

def get_S0_from_P(P, lamda0, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    nu = c0/lamda0
    return P/(W * d * h * nu * vg)

# Transcendental equation find root of
@cuda.jit(device=True)
def f(G:float, S0:float, I:float, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):

    c1 = C1(I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)
    c2 = C2(tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)

    numerator = (
        c1
        - alpha_int * c2 * S0
    )

    denominator = (
        c1
        - alpha_int * c2 * G * S0
    )

    # Invalid logarithm region
    if abs(denominator) <= 1e-12:
        return math.nan

    return (
        c1 * L
        - math.log(abs(G))
        - (
            (alpha_int + c1)
            / alpha_int
        )
        * math.log(abs(numerator / denominator))
    )

@cuda.jit(device=True)
def solve_gain(
    S0,
    I,

    tau_sp,
    n0,
    Gamma,
    a,
    alpha_int,
    vg,
    W,
    d,
    L,
):

    g_inflection = G_inflection(
        S0,
        I,
        tau_sp,
        n0,
        Gamma,
        a,
        alpha_int,
        vg,
        W,
        d,
        L,
    )

    x1 = g_inflection if g_inflection > 0 else 500

    x0 = x1/10000

    f0 = f(
        x0,
        S0,
        I,
        tau_sp,
        n0,
        Gamma,
        a,
        alpha_int,
        vg,
        W,
        d,
        L,
    )

    G_MAX = 1000
    f1 = f(
        x1,
        S0,
        I,
        tau_sp,
        n0,
        Gamma,
        a,
        alpha_int,
        vg,
        W,
        d,
        L,
    )

    if math.isnan(f0) or math.isnan(f1):
        return math.nan

    for _ in range(64):

        denom = (f1 - f0)
        numer = (x1 - x0)
        # avoid division by zero
        if abs(denom) < 1e-14:
            return math.nan
        step_magnitude = math.exp(math.log(abs(f1)) + math.log(abs(numer)) - math.log(abs(denom)))
        step_sign =  math.copysign(1.0, f1) * math.copysign(1.0, x1 - x0) * math.copysign(1.0, denom)
        x2 = x1 - step_sign*step_magnitude
        f2 = f(
            x2,
            S0,
            I,
            tau_sp,
            n0,
            Gamma,
            a,
            alpha_int,
            vg,
            W,
            d,
            L,
        )
        if math.isnan(f2):
            return math.nan
        # convergence test
        if abs(f2) < 1e-10:
            return x2
        # shift iteration
        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f2
    return x1 if x1 < G_MAX else math.nan

@cuda.jit
def calc_gain_matrix_kernel(
    S0s,
    Is,
    G_out,

    tau_sp,
    n0,
    Gamma,
    a,
    alpha_int,
    vg,
    W,
    d,
    L,
):

    i, j = cuda.grid(2)

    if i >= Is.size or j >= S0s.size:
        return

    S0 = S0s[j]
    I  = Is[i]

    G = solve_gain(
        S0,
        I,

        tau_sp,
        n0,
        Gamma,
        a,
        alpha_int,
        vg,
        W,
        d,
        L,
    )

    G_out[i, j] = G if G < 1e4 else math.nan

def calc_gain_matrix(
    S0s,
    Is,

    tau_sp,
    n0,
    Gamma,
    a,
    alpha_int,
    vg,
    W,
    d,
    L,
):
    G_out = np.empty(
        (Is.size, S0s.size),
        dtype=np.float64
    )
    d_S0s = cuda.to_device(S0s)
    d_Is  = cuda.to_device(Is)

    d_G_out = cuda.device_array(
        G_out.shape,
        dtype=np.float64
    )
    threads_per_block = (16, 16)

    blocks_x = (Is.size + 15) // 16
    blocks_y = (S0s.size + 15) // 16

    blocks_per_grid = (
        blocks_x,
        blocks_y,
    )

    calc_gain_matrix_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        d_S0s,
        d_Is,
        d_G_out,

        tau_sp,
        n0,
        Gamma,
        a,
        alpha_int,
        vg,
        W,
        d,
        L,
    )

    cuda.synchronize()

    return d_G_out.copy_to_host()

@cuda.jit
def calc_gain_matrix_param_sweep_kernel(
    S0s,
    Is,
    tau_sp,
    n0,
    Gamma,  #array
    a,      #array
    alpha_int,
    vg,
    W,  #array
    d,  #array
    L,

    # outputs
    G_out,
    nan_mask,
):

    p_idx, i_idx, s_idx = cuda.grid(3)

    if (
        p_idx >= G_out.shape[0]
        or i_idx >= G_out.shape[1]
        or s_idx >= G_out.shape[2]
    ):
        return

    S0 = S0s[s_idx]
    I  = Is[i_idx]

    G = solve_gain(
        S0,
        I,

        tau_sp,
        n0,
        Gamma[p_idx],
        a[p_idx],
        alpha_int,
        vg,
        W[p_idx],
        d,
        L[p_idx],
    )

    G_out[p_idx, i_idx, s_idx] = G

    nan_mask[p_idx, i_idx, s_idx] = math.isnan(G)

@cuda.jit
def calc_cost_from_gain_matrix_param_sweep_kernel(
    calc_G,
    real_G,
    costs,
    nan_counts,
):

    p_idx = cuda.grid(1)

    if p_idx >= costs.size:
        return

    costs[p_idx] = cost(real_G, calc_G[p_idx])
    nan_counts[p_idx] = count_nan(calc_G[p_idx])

    #optional: penalize cost with nan_count


def calc_cost_param_sweep(
    S0s,
    Is,
    tau_sp,
    n0,
    Gamma,  #array
    a,      #array
    alpha_int,
    vg,
    W,  #array
    d,
    L,  #array
    real_G,
    costs, # for returning output
    nan_counts = None,  # for returning output
    timing = False,
):
    N_params = len(Gamma)
    N_I = len(Is)
    N_S0 = len(S0s)


    if timing:  
        ############ warm-up ############

        # Force CUDA context creation + JIT warmup
        cuda.synchronize()
        _ = cuda.device_array(1, dtype=np.float64)
        dummy_arr = cuda.to_device(np.array([0.0]))

        calc_gain_matrix_param_sweep_kernel[
            (1, 1, 1),
            (1, 1, 1)
        ](
            dummy_arr,
            dummy_arr,

            tau_sp,
            n0,
            np.array([Gamma[0]]),
            np.array([a[0]]),
            alpha_int,
            vg,
            np.array([W[0]]),
            d,
            np.array([L[0]]),
            cuda.device_array((1, 1, 1), dtype=np.float64),
            cuda.device_array((1, 1, 1), dtype=np.bool_)
        )
        cuda.synchronize()

    ############ start timers ############
    start = cuda.event()
    end = cuda.event()

    start.record()

    d_G_out = cuda.device_array(
        (N_params, N_I, N_S0),
        dtype=np.float64,
    )

    # Currently this matrix is not really useful
    # but is left because it can be used to debug
    # for which param combos the gain estimation fails
    # (ie.: not only the number of fails but its positions)
    d_nan_mask = cuda.device_array(
        (N_params, N_I, N_S0),
        dtype=np.bool_,
    )

    d_S0s = cuda.to_device(S0s)
    d_Is = cuda.to_device(Is)
    d_real_G = cuda.to_device(real_G)

    d_Gamma = cuda.to_device(Gamma)
    d_a = cuda.to_device(a)
    d_W = cuda.to_device(W)
    d_L = cuda.to_device(L)

    # launch first kernel to calc gains for all param combinations
    # divide threads in 3D: (param_combination, I, S0)

    threads_per_block = (4, 8, 8)   # 256 threads per block

    blocks_per_grid = (
        (N_params + threads_per_block[0] - 1) // threads_per_block[0],
        (N_I + threads_per_block[1] - 1) // threads_per_block[1],
        (N_S0 + threads_per_block[2] - 1) // threads_per_block[2],
    )
    calc_gain_matrix_param_sweep_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        #inputs
        d_S0s,
        d_Is,

        tau_sp,
        n0,
        d_Gamma,
        d_a,
        alpha_int,
        vg,
        d_W,
        d,
        d_L,

        d_G_out,
        d_nan_mask
    )

    cuda.synchronize()
    
    ############ stop timers ############
    end.record()
    elapsed_ms_kernel_1 = cuda.event_elapsed_time(start, end)

    if timing: 
    
        ############ warm up second kernel ############
        warmup_S0s = np.array([0.0], dtype=np.float64)
        warmup_Is = np.array([0.0], dtype=np.float64)

        warmup_Gamma = np.array([Gamma[0]], dtype=np.float64)
        warmup_a = np.array([a[0]], dtype=np.float64)
        warmup_W = np.array([W[0]], dtype=np.float64)
        warmup_L = np.array([L[0]], dtype=np.float64)

        d_warmup_S0s = cuda.to_device(warmup_S0s)
        d_warmup_Is = cuda.to_device(warmup_Is)

        d_warmup_G_out = cuda.device_array(
            (1, 1, 1),
            dtype=np.float64,
        )

        d_warmup_nan_mask = cuda.device_array(
            (1, 1, 1),
            dtype=np.bool_,
        )

        calc_gain_matrix_param_sweep_kernel[
            (1, 1, 1),
            (1, 1, 1)
        ](
            d_warmup_S0s,
            d_warmup_Is,

            tau_sp,
            n0,
            warmup_Gamma,
            warmup_a,
            alpha_int,
            vg,
            warmup_W,
            d,
            warmup_L,

            d_warmup_G_out,
            d_warmup_nan_mask
        )

        cuda.synchronize()
    d_costs = cuda.device_array(
        N_params,
        dtype=np.float64,
    )

    d_nan_counts = cuda.device_array(
        N_params,
        dtype=np.float64,
    )


    # launch second kernel to calculate the cost function for all parameter combinations
    # divide threads in 1D: (param_combination)
    threads_per_block = (256,)
    blocks_per_grid = (
        (N_params + threads_per_block[0] - 1),
    )
    calc_cost_from_gain_matrix_param_sweep_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        d_G_out,
        d_real_G,
        d_costs,
        d_nan_counts,
    )

    d_costs.copy_to_host(costs)

    if nan_counts is not None:
        d_nan_counts.copy_to_host(nan_counts)

    ############ stop timers ############
    end.record()

    elapsed_ms_kernel_2 = cuda.event_elapsed_time(start, end)

    return (elapsed_ms_kernel_1, elapsed_ms_kernel_2)
