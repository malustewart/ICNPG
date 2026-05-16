import numpy as np
from scipy.optimize import root_scalar
from scipy.constants import elementary_charge as q  # carga del electron
from scipy.constants import h
from scipy.constants import c as c0
from dataclasses import dataclass
import matplotlib.pyplot as plt

WAVELENGTH = 1.55e-6

@dataclass(frozen=True)
class SOAParameters:

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

    def C1(self, I):
        return (
            I * self.tau_sp * self.Gamma * self.a
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
    
    def G_small_signal(self, I):
        return np.exp(self.C1(I) * self.L)

    def G_inflection(self, S0, I):
        return self.C1(I)/self.C2()/S0/self.alpha_int

    def get_S0_from_P(self, P, lamda0):
        nu = c0/lamda0
        return P/(self.W * self.d * h * nu * self.vg)

# Transcendental equation find root of
def f(G:float, S0:float, I:float, p:SOAParameters):

    C1 = p.C1(I)
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

def solve_gain(S0, I, p: SOAParameters):
    """
    Cálculo de G a partir de parámetros del SOA y la potencia de entrada S0:
    """

    f_wrapper = lambda G: f(G,S0,I,p)

    # Solve f(G)=0
    G_max = params.G_inflection(S0, I)
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

def calc_curve(S0s: np.ndarray, I: float, params : SOAParameters):
    """
    Cálculo de G a partir de parámetros del SOA para un conjunto de potencias de entrada S0:
    """
    Gs = [solve_gain(S0, I, params) for S0 in S0s]
    return np.array(Gs)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    I=0.15              # 150 mA

    # Reasonable-ish SOA parameters
    params = SOAParameters(
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


    print("================================================")
    print("Testing constants")
    print("================================================")

    C1 = params.C1(I)
    C2 = params.C2()

    print(f"C1 = {C1:.6e}")
    print(f"C2 = {C2:.6e}")

    if not np.isfinite(C1):
        raise ValueError("C1 is not finite")

    if not np.isfinite(C2):
        raise ValueError("C2 is not finite")

    print("\n================================================")
    print("Curve test")
    print("================================================")


    Pins = np.linspace(0.5e-3, 3.5e-3, 12)
    S0s = [params.get_S0_from_P(P, WAVELENGTH) for P in Pins]

    try:
        Gs = calc_curve(S0s, I, params)

        for P, s, g in zip(Pins, S0s, Gs):
            print(f"P = {P:.2} S0 = {s:.3e}   G = {g:.6e}")


        # Basic sanity checks
        if np.any(~np.isfinite(Gs)):
            raise ValueError("Non-finite gains detected")

        if np.any(Gs <= 0):
            raise ValueError("Non-positive gains detected")

        print("\nCurve computed successfully:\n")
        print("\nAll tests passed.")

    except Exception as e:
        print("\nCurve computation failed:")
        print(e)
