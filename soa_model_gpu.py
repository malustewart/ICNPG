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

    return x1

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

    G_out[i, j] = G

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
    d,  #array
    L,
    real_G,
    costs, # for returning output
    nan_counts = None  # for returning output
):
    N_params = len(Gamma)
    N_I = len(Is)
    N_S0 = len(S0s)

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

    d_costs = cuda.device_array(
        N_params,
        dtype=np.float64,
    )

    d_nan_counts = cuda.device_array(
        N_params,
        dtype=np.float64,
    )

    d_S0s = cuda.to_device(S0s)
    d_Is = cuda.to_device(Is)
    d_real_G = cuda.to_device(real_G)

    d_Gamma_vals = cuda.to_device(Gamma)
    d_a_vals = cuda.to_device(a)
    d_W_vals = cuda.to_device(W)
    d_d_vals = cuda.to_device(d)

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
        Gamma,
        a,
        alpha_int,
        vg,
        W,
        d,
        L,

        d_G_out,
        d_nan_mask
    )

    cuda.synchronize()

    # launch second kernel to calculate the cost function for all parameter combinations
    # divide threads in 1D: (param_combination)
    threads_per_block = (256,)
    blocks_per_grid = (
        (N_params + threads_per_block[0] - 1), # threads_per_block[0],
    )
    calc_cost_from_gain_matrix_param_sweep_kernel[
        threads_per_block,
        blocks_per_grid
    ](
        d_G_out,
        d_real_G,
        d_costs,
        d_nan_counts,
    )
    d_costs.copy_to_host(costs)
    if nan_counts is not None:
        d_nan_counts.copy_to_host(nan_counts)


# ============================================================
# TEST
# ============================================================

def main():
    I=0.300              # [A]

    # Reasonable-ish SOA parameters
    soa_params = [
        1e-9,   #tau_sp [s]
        1e24,   #n0
        0.3,    #Gamma
        3e-20,  #a
        1000,   #alpha_int [1/m]
        8e7,    #vg [m/s]
        2e-6,   #W [m]
        0.2e-6, #d [m]
        500e-6  #L [m]
    ]

    gain_map = np.load('measurement/processed/soa_2d_gain_map.npz')

    Is = gain_map["soa_current_mA"] * 1e-3
    Pins = gain_map["input_power_W"]
    gain = gain_map["gain_linear"]

    # fix dataset with numerical errors at edges that include nan
    Pin_range = slice(1,len(Pins) - 1)
    Is_range = slice(4,len(Is))

    gain = gain[Is_range, Pin_range].copy() #copy is necessary so that memory is contiguous (which does not happen if working with view)
    Is = Is[Is_range].copy()
    Pins = Pins[Pin_range].copy()
    S0s = get_S0_from_P(Pins, WAVELENGTH, *soa_params)

    a_range = np.logspace(-19.52, -19.52, 1)
    gamma_range = np.logspace(-1, -0.1, 2)
    W_range = np.logspace(-6, -5, 2)
    L_range = np.logspace(-4, -3, 2)

    a_s, Ws, Ls, gammas = np.meshgrid(a_range, W_range, L_range, gamma_range, indexing="ij") # indexing: [a,W,L,gamma]

    a_s = a_s.flatten()
    Ws = Ws.flatten()
    Ls = Ls.flatten()
    gammas = gammas.flatten()

    costs = np.zeros_like(Ws)
    nan_counts = np.zeros_like(Ws)

    soa_params[2] = gammas
    soa_params[3] = a_s
    soa_params[6] = Ws
    soa_params[8] = Ls

    G_matrix = calc_gain_matrix(
        S0s,
        Is,
        soa_params[0],
        soa_params[1],
        soa_params[2][-1],
        soa_params[3][-1],
        soa_params[4],
        soa_params[5],
        soa_params[6][-1],
        soa_params[7],
        soa_params[8][-1],
    )


    print(np.nanmax(G_matrix))
    I_index, S0_index = np.unravel_index(np.nanargmax(G_matrix), G_matrix.shape)

    print(np.count_nonzero(G_matrix<0))

    I_max = Is[I_index]
    S0_max = S0s[S0_index]
    Pin_max = Pins[S0_index]

    print(I_index, S0_index)
    print(I_max, Pin_max)

    print(G_matrix[0,14])
    c1 = C1(soa_params[0],
        soa_params[1],
        soa_params[2][-1],
        soa_params[3][-1],
        soa_params[4],
        soa_params[5],
        soa_params[6][-1],
        soa_params[7],
        soa_params[8][-1],
    )

    c2 = C2(soa_params[0],
        soa_params[1],
        soa_params[2][-1],
        soa_params[3][-1],
        soa_params[4],
        soa_params[5],
        soa_params[6][-1],
        soa_params[7],
        soa_params[8][-1],
    )


    # calc_cost_param_sweep(S0s, Is, *soa_params, gain, costs, nan_counts)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(Ws, gammas, Ls, c=costs, cmap='PRGn')
    # ax.set_xlabel("W")
    # ax.set_ylabel("Gamma")
    # ax.set_zlabel("L")
    # fig.colorbar(scatter, ax=ax)

    # for a, W, L, gamma, cost_, nan_count in zip(a_s, Ws, Ls, gammas, costs, nan_counts):
    #     print(f"{a:10.2e} {W:10.2e} {L:10.2e} {gamma:.2e} {cost_:f} {nan_count}")

    # plt.show()

if __name__ == "__main__":
    main()
