import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import griddata


DATA_FOLDER = Path("measurement")

# Output dataset
OUTPUT_CSV = DATA_FOLDER/"processed/soa_gain_dataset.csv"

OUTPUT_KEY = "CH1"
INPUT_KEY = "CH2"

scope_zin = 50
pd_input_R = 1
pd_output_R = 1
tap_input_T_to_scope = 0.0548
tap_input_T_to_soa =  0.8337
tbf_loss = 0.9

SMOOTH = True
SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 3

# ============================================================
# HELPERS
# ============================================================

def extract_current_mA(filename: str) -> float:
    """
    Extract SOA current from filenames like:
        soa_I_60mA_2026051517h42m21s.npz
    """
    match = re.search(r"I_(\d+(?:\.\d+)?)mA", filename)

    if match is None:
        raise ValueError(f"Could not parse current from: {filename}")

    return float(match.group(1))

def linear_gain(p_out, p_in):
    return p_out / p_in

def gain_dB(p_out, p_in):
    return 10 * np.log10(p_out / p_in)

# Pin in Watts
# I in mA
def plot_gain_vs_Pin(gain, Pin, I):
    plt.figure()
    plt.semilogy(Pin*10e3, gain, linestyle='', marker='.')
    plt.xlabel("P in [mW]")
    plt.ylabel("Gain (linear)")
    plt.title(f"Gain vs. P_in - I_soa: {I}mA")

# Pout in Watts
# I in mA
def plot_gain_vs_Pout(gain, Pout, I):
    plt.figure()
    plt.scatter(Pout*10e3, gain)
    plt.xlabel("P out [mW]")
    plt.ylabel("Gain (linear)")
    plt.title(f"Gain vs. P_out - I_soa: {I}mA")

def create_soa_gain_dataset(data_folder, outfile=None):
    rows = []

    files = sorted(DATA_FOLDER.glob("*.npz"))

    if not files:
        raise RuntimeError(f"No .npz files found in: {DATA_FOLDER}")

    for file in files:

        current_mA = extract_current_mA(file.name)
        data = np.load(file)
        v_in = np.asarray(data[INPUT_KEY], dtype=float)[:-4]
        v_out = np.asarray(data[OUTPUT_KEY], dtype=float)[:-4]
        v_in = savgol_filter(v_in, 500, 3)
        v_out = savgol_filter(v_out, 500, 3)
        v_out -= np.min(v_out) # remove amplified spontaneous emissions output power
        
        # --------------------------------------------------------
        # Convert from V to power
        # --------------------------------------------------------

        p_in = v_in / pd_input_R / scope_zin / tap_input_T_to_scope * tap_input_T_to_soa
        p_out = v_out / pd_output_R /scope_zin / tbf_loss
        valid = (
            np.isfinite(p_in)
            & np.isfinite(p_out)
            & (p_in > 0)
            & (p_out > 0)
        )
        p_in = p_in[valid]
        p_out = p_out[valid]

        # --------------------------------------------------------
        # Compute gain
        # --------------------------------------------------------

        g_lin = linear_gain(p_out, p_in)
        g_db = gain_dB(p_out, p_in)

        # --------------------------------------------------------
        # Store rows
        # --------------------------------------------------------

        for pin, pout, glin, gdb in zip(p_in, p_out, g_lin, g_db):

            rows.append(
                {
                    "soa_current_mA": current_mA,
                    "input_power_W": pin,
                    "output_power_W": pout,
                    "gain_linear": glin,
                    "gain_dB": gdb,
                }
            )

    df = pd.DataFrame(rows)

    # Sort nicely
    df = df.sort_values(
        by=["soa_current_mA", "input_power_W"]
    ).reset_index(drop=True)

    # Save
    if outfile:
        df.to_csv(outfile, index=False)
    
    return df

def interpolate_soa_gain_to_grid(currents, pin, gain, outdir="."):

    current_grid = np.array([60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320])

    pin_grid = np.geomspace(
        pin[pin > 0].min(),
        pin.max(),
        400,
    )

    P, I = np.meshgrid(pin_grid, current_grid)

    G = griddata(
        points=(pin, currents),
        values=gain,
        xi=(P, I),
        method="linear",
    )

    G_dB = 10 * np.log10(G)

    np.savez(
        f"{outdir}/soa_2d_gain_map.npz",
        soa_current_mA=current_grid,
        input_power_W=pin_grid,
        gain_linear=G,
        gain_dB=G_dB,
    )

    return current_grid, pin_grid, G, G_dB


def plot_gain_grid(current_grid, pin_grid, G, G_dB, outdir=".", linear_filename="soa_2d_gain_map_linear.png", log_filename="soa_2d_gain_map_dB.png"):

    P, I = np.meshgrid(pin_grid, current_grid)

    plt.figure(figsize=(8, 5))

    pcm = plt.pcolormesh(
        P,
        I,
        G,
        shading="auto",
    )

    plt.xlabel("Input Power (W)")
    plt.ylabel("SOA Current (mA)")
    plt.title("SOA 2D Gain Map (Linear Gain)")

    cbar = plt.colorbar(pcm)
    cbar.set_label("Gain")

    plt.tight_layout()

    plt.savefig(
        f"{outdir}/{linear_filename}",
        dpi=300,
    )

    # ============================================================
    # PLOT dB GAIN MAP
    # ============================================================

    plt.figure(figsize=(8, 5))
    
    pcm = plt.pcolormesh(
        P,
        I,
        G_dB,
        shading="auto",
    )

    plt.xlabel("Input Power (W)")
    plt.ylabel("SOA Current (mA)")
    plt.title("SOA 2D Gain Map (dB)")

    cbar = plt.colorbar(pcm)
    cbar.set_label("Gain (dB)")

    plt.tight_layout()

    plt.savefig(
        f"{outdir}/{log_filename}",
        dpi=300,
    )


if __name__ == '__main__':
    outfile = OUTPUT_CSV
    
    df = create_soa_gain_dataset("measurement", outfile)

    print("\n================================================")
    print(f"Saved dataset to: {outfile}")
    print("================================================")


    # remove numerically unstable cases
    filtered_df = df.loc[(df['input_power_W'] >= 0.5e-3) & (df['gain_linear'] <= 100)]


    soa_current_mA = filtered_df["soa_current_mA"].values
    pin = filtered_df["input_power_W"].values
    gain_linear = filtered_df["gain_linear"].values

    current_grid, pin_grid, G, G_dB = interpolate_soa_gain_to_grid(soa_current_mA, pin, gain_linear, outdir="measurement/processed")

    plot_gain_grid(current_grid, pin_grid, G, G_dB, outdir="measurement/processed")
