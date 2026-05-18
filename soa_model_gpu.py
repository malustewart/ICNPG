import math
from scipy.constants import elementary_charge as q  # carga del electron
from scipy.constants import h
from scipy.constants import c as c0
import matplotlib.pyplot as plt
from numba import cuda

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
    return math.exp(C1(I) * L)

@cuda.jit(device=True)
def G_inflection(S0, I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    return C1(I)/C2()/S0/alpha_int

def get_S0_from_P(P, lamda0, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
    nu = c0/lamda0
    return P/(W * d * h * nu * vg)

# Transcendental equation find root of
@cuda.jit(device=True)
def f(G:float, S0:float, I:float, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):

    C1 = C1(I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)
    C2 = C2(tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)

    numerator = (
        C1
        - alpha_int * C2 * S0
    )

    denominator = (
        C1
        - alpha_int * C2 * G * S0
    )

    # Invalid logarithm region
    if math.abs(denominator) <= 1e-12:
        return math.nan

    return (
        C1 * L
        - math.log(math.abs(G))
        - (
            (alpha_int + C1)
            / alpha_int
        )
        * math.log(math.abs(numerator / denominator))
    )

# def solve_gain(S0, I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L):
#     """
#     Cálculo de G a partir de parámetros del SOA y la potencia de entrada S0:
#     """

#     f_wrapper = lambda G: f(G,S0,I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)

#     # Solve f(G)=0
#     G_max = G_inflection(S0, I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)
#     x0 = G_max / 10
#     solution = root_scalar(
#         f_wrapper,
#         bracket=[1e-1, G_max],
#         method="secant",
#         x0=x0
#     )

#     if solution.converged and solution.root > 0:
#         return solution.root
    
#     # Try again with different conditions
#     G_max = G_inflection(S0, I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)/10
#     x0 = G_max / 10
#     solution = root_scalar(
#         f_wrapper,
#         bracket=[1e-1, G_max],
#         method="secant",
#         x0=x0
#     )

#     if solution.converged and solution.root > 0:
#         return solution.root

#     # Try again with different conditions
#     G_max = G_inflection(S0, I, tau_sp, n0, Gamma, a, alpha_int, vg, W, d, L)
#     x0 = G_max / 2
#     solution = root_scalar(
#         f_wrapper,
#         bracket=[1e-1, G_max],
#         method="secant",
#         x0=x0
#     )

#     if solution.converged and solution.root > 0:
#         return solution.root

#     return math.nan



# def calc_gain_curve(S0s: np.ndarray, I: float, params : SOAParameters):
#     """
#     Cálculo de G a partir de parámetros del SOA para un conjunto de potencias de entrada S0:
#     """
#     Gs = [solve_gain(S0, I, params) for S0 in S0s]
#     return np.array(Gs)

# def calc_gain_matrix(S0s: np.ndarray, Is: np.ndarray, params: SOAParameters):
#     return np.array([[solve_gain(S0, I, params) for S0 in S0s] for I in Is])


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    I=0.300              # [A]

    # Reasonable-ish SOA parameters
    soa_params = (
        1e-9,   #tau_sp [s]
        1e24,   #n0
        0.3,    #Gamma
        3e-20,  #a
        1000,   #alpha_int [1/m]
        8e7,    #vg [m/s]
        2e-6,   #W [m]
        0.2e-6, #d [m]
        500e-6  #L [m]
    )



    print("================================================")
    print("Testing constants")
    print("================================================")

    c1 = C1(I, *soa_params)
    c2 = C2(I, *soa_params)

    print(f"C1 = {c1:.6e}")
    print(f"C2 = {c2:.6e}")

    if not math.isfinite(c1):
        raise ValueError("C1 is not finite")

    if not math.isfinite(c2):
        raise ValueError("C2 is not finite")

    # print("\n================================================")
    # print("Curve test")
    # print("================================================")


    # Pins = np.linspace(2.2e-3, 2.4e-3, 12)
    # S0s = [params.get_S0_from_P(P, WAVELENGTH) for P in Pins]

    # try:
    #     Gs = calc_gain_curve(S0s, I, params)

    #     for P, s, g in zip(Pins, S0s, Gs):
    #         print(f"P = {P:.2} S0 = {s:.3e}   G = {g:.6e}")


    #     # Basic sanity checks
    #     if np.any(~np.isfinite(Gs)):
    #         raise ValueError("Non-finite gains detected")

    #     if np.any(Gs <= 0):
    #         raise ValueError("Non-positive gains detected")

    #     print("\nCurve computed successfully:\n")
    #     print("\nAll tests passed.")

    # except Exception as e:
    #     print("\nCurve computation failed:")
    #     print(e)


    # # Pin in Watts
    # # I in mA
    # def plot_gain_vs_Pin(gain, Pin, I):
    #     plt.figure()
    #     plt.semilogy(Pin*1e3, gain, linestyle='', marker='.')
    #     plt.xlabel("P in [mW]")
    #     plt.ylabel("Gain (linear)")
    #     plt.title(f"Gain vs. P_in - I_soa: {I}mA")

    # # Pout in Watts
    # # I in mA
    # def plot_gain_vs_Pout(gain, Pout, I):
    #     plt.figure()
    #     plt.scatter(Pout*1e3, gain)
    #     plt.xlabel("P out [mW]")
    #     plt.ylabel("Gain (linear)")
    #     plt.title(f"Gain vs. P_out - I_soa: {I}mA")

    # plot_gain_vs_Pin(Gs, Pins, I*1e3)
    # plt.show()