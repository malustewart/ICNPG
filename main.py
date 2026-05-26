#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import soa_model_gpu as gpu
import soa_model as cpu


# Reasonable default SOA parameters
soa_params_default = [
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

WAVELENGTH = 1.55e-6

def load_gain_map():
    return np.load('measurement/processed/soa_2d_gain_map.npz')

def calc_costs_in_gpu(gain_map, gammas, a_s, Ws, Ls, costs, nan_counts, timing):


    ### LOAD EXPERIMENTAL RESULTS ###

    Is = gain_map["soa_current_mA"] * 1e-3  # convert to A
    Pins = gain_map["input_power_W"]
    gain = gain_map["gain_linear"]

    # fix dataset with numerical errors at edges that include nan
    Pin_range = slice(1,len(Pins) - 1)
    Is_range = slice(4,len(Is))
    gain = gain[Is_range, Pin_range].copy() #copy is necessary so that memory is contiguous (which does not happen if working with view)
    Is = Is[Is_range].copy()
    Pins = Pins[Pin_range].copy()

    tau_sp, n0, _, _, alpha_int, vg, _, d, _ = soa_params_default

    S0s = np.array([gpu.get_S0_from_P(P, WAVELENGTH, tau_sp, n0, gamma, a, alpha_int, vg, W, d, L) for P, gamma, a, W, L in zip(Pins, gammas, a_s, Ws, Ls)])

    soa_params = soa_params_default.copy()
    soa_params[2] = gammas
    soa_params[3] = a_s
    soa_params[6] = Ws
    soa_params[8] = Ls

    return gpu.calc_cost_param_sweep(S0s, Is, *soa_params, gain, costs, nan_counts, timing)


def calc_costs_in_cpu(gain_map, gammas, a_s, Ws, Ls, costs, nan_counts, timing):


    ### LOAD EXPERIMENTAL RESULTS ###

    Is = gain_map["soa_current_mA"] * 1e-3  # convert to A
    Pins = gain_map["input_power_W"]
    gain = gain_map["gain_linear"]

    # fix dataset with numerical errors at edges that include nan
    Pin_range = slice(1,len(Pins) - 1)
    Is_range = slice(4,len(Is))
    gain = gain[Is_range, Pin_range].copy() #copy is necessary so that memory is contiguous (which does not happen if working with view)
    Is = Is[Is_range].copy()
    Pins = Pins[Pin_range].copy()

    tau_sp, n0, _, _, alpha_int, vg, _, d, _ = soa_params_default
    S0s = np.array([[gpu.get_S0_from_P(P, WAVELENGTH, tau_sp, n0, gamma, a, alpha_int, vg, W, d, L) for P in Pins] for gamma, a, W, L in zip(gammas, a_s, Ws, Ls)])
    
    soa_params = cpu.SOAParameters(*soa_params_default)
    return cpu.calc_cost_param_sweep(S0s, Is, soa_params, gammas,a_s, Ws, Ls, gain, costs, nan_counts )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SOA simulations with experimental results."
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file containing parameter sweeps.",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU execution.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Enable CPU execution (default: False).",
    )

    parser.add_argument(
        "--timing",
        action="store_true",
        default=True,
        help="Enable timing measurements (default: False).",
    )

    parser.add_argument(
        "--N_I",
        type=int,
        default=None,
        help="Optional N_I value.",
    )

    parser.add_argument(
        "--N_P",
        type=int,
        default=None,
        help="Optional N_P value.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = args.input_csv
    run_gpu    = args.gpu
    run_cpu    = args.cpu
    timing = args.timing
    N_I    = args.N_I
    N_P    = args.N_P
    output = args.output

    data = np.loadtxt(
        input_csv,
        delimiter=",",
        skiprows=1,
    )

    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in CSV, got {data.shape[1]}")

    gammas = data[:, 0].copy()  # copy so that the array is contiguous (needed for gpu transfer)
    a_s    = data[:, 1].copy()
    Ws     = data[:, 2].copy()
    Ls     = data[:, 3].copy()


    gain_map = load_gain_map()
    
    costs = np.zeros_like(gammas)
    nan_counts = np.zeros_like(gammas)
    
    if run_gpu:
        time_k1, time_k2 = calc_costs_in_gpu(gain_map, gammas, a_s, Ws, Ls, costs, nan_counts, timing)

        output_data = np.column_stack((
            gammas,
            a_s,
            Ws,
            Ls,
            costs,
            nan_counts,
        ))

        output_path = (
            Path(output).with_name(
                Path(output).stem + "_with_cost_gpu.csv"
            )
            if output is not None
            else input_csv.with_name(
                input_csv.stem + "_with_cost_gpu.csv"
            )
        )

        np.savetxt(
            output_path,
            output_data,
            delimiter=",",
            header="gamma,a_s,W,d,cost, nan_count",
            comments="",
        )

        print(f"{time_k1:12.3f} {time_k2:12.3f} ", end="")


    if run_cpu:
        time_total_cpu = calc_costs_in_cpu(gain_map, gammas, a_s, Ws, Ls, costs, nan_counts, timing)

        output_data = np.column_stack((
            gammas,
            a_s,
            Ws,
            Ls,
            costs,
            nan_counts,
        ))

        output_path = (
            Path(output).with_name(
                Path(output).stem + "_with_cost_cpu.csv"
            )
            if output is not None
            else input_csv.with_name(
                input_csv.stem + "_with_cost_cpu.csv"
            )
        )

        np.savetxt(
            output_path,
            output_data,
            delimiter=",",
            header="gamma,a_s,W,d,cost, nan_count",
            comments="",
        )

        print(f"{time_total_cpu:12.3f} ", end="")


if __name__ == "__main__":
    main()