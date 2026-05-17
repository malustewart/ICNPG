import numpy as np
from soa_model import SOAParameters, calc_gain_matrix, solve_gain
import numba as nb
from scipy.optimize import root_scalar
import optimization
from dataclasses import replace
import matplotlib.pyplot as plt


WAVELENGTH = 1.55e-6

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

gain_map = np.load('measurement/processed/soa_2d_gain_map.npz')

Is = gain_map["soa_current_mA"] * 1e-3
Pins = gain_map["input_power_W"]
gain = gain_map["gain_linear"]

# fix dataset with numerical errors at edges that include nan
Pin_range = slice(1,len(Pins) - 1)
Is_range = slice(4,len(Is))

gain = gain[Is_range, Pin_range]
Is = Is[Is_range]
Pins = Pins[Pin_range]
S0s = params.get_S0_from_P(Pins, WAVELENGTH)

# a_range = np.logspace(-21.2, -19.5, 5)
gamma_range = np.logspace(-1, -0.1, 10)
W_range = np.logspace(-6, -5, 10)
L_range = np.logspace(-4, -3, 10)

Ws, Ls, gammas = np.meshgrid(W_range, L_range, gamma_range, indexing="ij") # indexing: [W,L, gamma]

costs = np.zeros_like(Ws)
nans = np.zeros_like(Ws)


for i, W in enumerate(W_range):
    for j, L in enumerate(L_range):
        for k, gamma in enumerate(gamma_range):
            p = replace(params, L=L, W=W, Gamma=gamma)
            Gs = calc_gain_matrix(S0s, Is, p)

            cost = optimization.cost(gain, Gs)
            costs[i,j,k] = cost
            nans[i,j,k] = np.count_nonzero(np.isnan(Gs))
            # print(f"{a:10.2e} {cost}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(Ws, gammas, Ls, c=costs, cmap='PRGn')
ax.set_xlabel("W")
ax.set_ylabel("Gamma")
ax.set_zlabel("L")
fig.colorbar(scatter, ax=ax)


for i, W in enumerate(W_range):
    for j, L in enumerate(L_range):
        for k, gamma in enumerate(gamma_range):
            print(f"{W:10.2e} {L:10.2e} {gamma:.2e} {costs[i,j,k]:10.2e} {nans[i,j,k]}")
    print()


plt.show()