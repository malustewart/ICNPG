import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N_I = 14
N_P = 400

# csv columns: gamma,a_s,W,d,cost, nan_count

# Pin in Watts
# I in mA
def plot_gain_vs_Pin(gain, Pin, I):
    plt.figure()
    plt.semilogy(Pin*1e3, gain, linestyle='', marker='.')
    plt.xlabel("P in [mW]")
    plt.ylabel("Gain (linear)")
    plt.title(f"Gain vs. P_in - I_soa: {I}mA")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <data.csv>")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    # Output file: same name, different extension
    output_file = input_file.with_name(f"{input_file.stem}_nans.png")

    # CSV format:
    # W,gamma,L,cost
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    gammas = data[:, 0]
    a_s = data[:, 1]
    Ws = data[:, 2]
    Ls = data[:, 3]
    costs = data[:, 4]
    nan_counts = data[:, 5]


    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    nan_counts_norm = nan_counts / N_I / N_P

    # Scatter plot
    scatter = ax.scatter(
        Ws,
        gammas,
        Ls,
        c=nan_counts_norm,
        cmap="Reds",
    )

    
    # Labels
    ax.set_xlabel("W")
    ax.set_ylabel("Gamma")
    ax.set_zlabel("L")

    # Colorbar
    fig.colorbar(scatter, ax=ax, label="% of non-convergence")
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


    print(f"Saved plot to: {output_file}")

###########################################################

    # Output file: same name, different extension
    output_file = input_file.with_name(f"{input_file.stem}_cost.png")

    # CSV format:
    # W,gamma,L,cost
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    gammas = data[:, 0]
    a_s = data[:, 1]
    Ws = data[:, 2]
    Ls = data[:, 3]
    costs = np.log2(data[:, 4])
    nan_counts = data[:, 5]

    mask = np.isfinite(nan_counts)

    Ws = Ws[mask]
    gammas = gammas[mask]
    Ls = Ls[mask]
    costs = costs[mask]


    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    scatter = ax.scatter(
        Ws,
        gammas,
        Ls,
        c=costs,
        cmap="Wistia",
        norm=LogNorm(
            vmin=np.min(costs[costs > 0]),
            vmax=np.max(costs)
        )
    )

    

    # Labels
    ax.set_xlabel("W")
    ax.set_ylabel("Gamma")
    ax.set_zlabel("L")

    # Colorbar
    fig.colorbar(scatter, ax=ax, label="Cost")
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


    print(f"Saved plot to: {output_file}")

if __name__ == "__main__":
    main()