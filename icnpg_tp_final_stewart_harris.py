import numpy as np
from scipy.optimize import root_scalar
from scipy.constants import elementary_charge as q  # carga del electron
from scipy.constants import h
from scipy.constants import c as c0
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class SOAParameters:
    #TODO: fix units to more reasonable choices (such as um for lengths)
    # Electrical
    I: float              # Injection current [A]

    # Carrier dynamics
    tau_sp: float         # Spontaneous lifetime [s]
    n0: float             # Transparency carrier density [m^-3]

    # Optical
    Gamma: float          # Confinement factor
    a: float              # Differential gain [m^2]
    alpha_int: float      # Internal loss [m^-1]
    vg: float             # Group velocity [m/s]

    # Geometry
    W: float              # Width [m]
    d: float              # Active layer thickness [m]
    L: float              # Length [m]

    def C1(self):
        return (
            self.I * self.tau_sp * self.Gamma * self.a
            / (q * self.W * self.d * self.L)
            - self.n0 * self.Gamma * self.a
            - self.alpha_int
        )

    def C2(self):

        return (
            self.tau_sp
            * self.Gamma
            * self.vg
            * self.a
        )
    
    def G_small_signal(self):
        return np.exp(self.C1() * self.L)

    def G_inflection(self, S0):
        return self.C1()/self.C2()/S0/self.alpha_int

    def get_S0_from_P(self, P, lamda0):
        nu = c0/lamda0
        return P/(self.W * self.d * h * nu * self.vg)

# Transcendental equation find root of
def f(G:float, S0:float, p:SOAParameters):

    C1 = p.C1()
    C2 = p.C2()

    numerator = (
        C1
        - p.alpha_int * C2 * S0
    )

    denominator = (
        C1
        - p.alpha_int * C2 * G * S0
    )

    # Invalid logarithm region
    if np.abs(denominator) <= 1e-12:
        return np.nan

    return (
        C1 * p.L
        - np.log(np.abs(G))
        - (
            (p.alpha_int + C1)
            / p.alpha_int
        )
        * np.log(np.abs(numerator / denominator))
    )

def solve_gain(S0, p: SOAParameters):
    """
    Cálculo de G a partir de parámetros del SOA y la potencia de entrada S0:
    """

    f_wrapper = lambda G: f(G,S0,p)

    # Solve f(G)=0
    G_max = params.G_inflection(S0)
    solution = root_scalar(
        f_wrapper,
        bracket=[1e-1, G_max],
        method="secant",
        x0=G_max/10
    )

    if not solution.converged or solution.root < 0:
        print("Root method did not converge")
        return np.nan

    return solution.root

def calc_curve(S0s: np.ndarray, params : SOAParameters):
    """
    Cálculo de G a partir de parámetros del SOA para un conjunto de potencias de entrada S0:
    """
    Gs = [solve_gain(S0, params) for S0 in S0s]
    return np.array(Gs)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Reasonable-ish SOA parameters
    params = SOAParameters(
        I=0.15,              # 150 mA
        tau_sp=1e-9,
        n0=1e24,
        Gamma=0.3,
        a=3e-20,
        alpha_int=1000,      # 1000 1/m
        vg=8e7,
        W=2e-6,
        d=0.2e-6,
        L=500e-6,
    )

    # P = 1e-3
    # S0 = params.get_S0_from_P(P, 1.55e-6)
    # print(S0)

    # print(params.C2()*S0)

    print("================================================")
    print("Testing constants")
    print("================================================")

    C1 = params.C1()
    C2 = params.C2()

    print(f"C1 = {C1:.6e}")
    print(f"C2 = {C2:.6e}")

    if not np.isfinite(C1):
        raise ValueError("C1 is not finite")

    if not np.isfinite(C2):
        raise ValueError("C2 is not finite")

    print("\n================================================")
    print("Single-point solve")
    print("================================================")

    lamda = 1.55e-6

    S0 = params.get_S0_from_P(1e-3, 1.55e-6)

    G_num = solve_gain(S0, params)

    print(f"\nGain for S0={S0:.3e}: G={G_num:.6e}")

    Gs = np.logspace(-2, np.log10(params.G_small_signal()), 50)
    Ps = np.logspace(-9,0,10)
    S0s = params.get_S0_from_P(Ps, lamda)

    C1 = params.C1()
    C2 = params.C2()
    alpha = params.alpha_int

    plt.figure()
    for S0, P in zip(S0s,Ps):
        fs = [f(G, S0, params) for G in Gs]
        G_calc = solve_gain(S0, params)
        print(f"Gain for {P}: {G_calc}")
        plt.axvline(G_calc)
        plt.semilogx(Gs, fs, label=f"{S0:.2}")
    plt.grid()
    plt.legend()
    plt.show()


    # print("\n================================================")
    # print("Curve test")
    # print("================================================")

    # S0s = np.logspace(14, 20, 10)

    # try:
    #     Gs = calc_curve(S0s, params)

    #     print("\nCurve computed successfully:\n")

    #     for s, g in zip(S0s, Gs):
    #         print(f"S0 = {s:.3e}   G = {g:.6e}")

    #     # Basic sanity checks
    #     if np.any(~np.isfinite(Gs)):
    #         raise ValueError("Non-finite gains detected")

    #     if np.any(Gs <= 0):
    #         raise ValueError("Non-positive gains detected")

    #     print("\nAll tests passed.")

    # except Exception as e:
    #     print("\nCurve computation failed:")
    #     print(e)