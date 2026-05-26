# Python Script for CPU/GPU Runtime and Speedup Plots

import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_cpu_file(path):
    data = {}

    with open(path, 'r') as f:
        for line in f:
            parts = line.split()

            # Skip malformed lines
            if len(parts) != 2:
                continue

            try:
                param = int(parts[0])
                runtime = float(parts[1])
            except ValueError:
                continue

            # Keep last occurrence
            data[param] = runtime

    params = np.array(sorted(data.keys()))
    runtimes = np.array([data[p] for p in params])

    return params, runtimes


def load_gpu_file(path):
    data = {}

    with open(path, 'r') as f:
        for line in f:
            parts = line.split()

            # Skip incomplete lines
            if len(parts) != 3:
                continue

            try:
                param = int(parts[0])
                first_half = float(parts[1])
                second_half = float(parts[2])
            except ValueError:
                continue

            # Keep last occurrence
            data[param] = (first_half, second_half)

    params = np.array(sorted(data.keys()))

    first = np.array([data[p][0] for p in params])
    second = np.array([data[p][1] for p in params])

    return params, first, second


# =========================
# Load data
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--cpu")
parser.add_argument("--gpu")
args = parser.parse_args()

# =========================
# Extract columns
# =========================

params_cpu, cpu_runtime = load_cpu_file(args.cpu)
params_gpu, gpu_first, gpu_second = load_gpu_file(args.gpu)

gpu_total = gpu_first + gpu_second



plt.figure(figsize=(8, 5))

plt.plot(params_cpu, cpu_runtime, marker='o', label='CPU Runtime')
plt.plot(params_gpu, gpu_total, marker='o', label='GPU Total Runtime')

reference = cpu_runtime[0] * (params_cpu / params_cpu[0]) * 0.9

plt.plot(params_cpu, reference, '--', label='Linear Growth')

plt.xscale('log', base=2)
plt.yscale('log')

plt.xlabel('Parameter Number')
plt.ylabel('Runtime (ms)')
plt.title('CPU vs GPU Runtime')
plt.grid(True, which='both')
plt.legend()

plt.savefig('fig_runtime_comparison.png', dpi=300, bbox_inches='tight')

# =========================
# Plot 2:
# GPU first half vs second half
# =========================

plt.figure(figsize=(8, 5))

plt.plot(params_gpu, gpu_first, marker='o', label='GPU First Half')
plt.plot(params_gpu, gpu_second, marker='o', label='GPU Second Half')

plt.xscale('log', base=2)
plt.yscale('log')

plt.xlabel('Parameter Number')
plt.ylabel('Runtime (ms)')
plt.title('GPU Runtime Breakdown')
plt.grid(True, which='both')
plt.legend()

plt.savefig('fig_gpu_breakdown.png', dpi=300, bbox_inches='tight')

# =========================
# Plot 3:
# GPU speedup
# =========================

# Match CPU and GPU parameter sets
common_params = np.intersect1d(params_cpu, params_gpu)

cpu_runtime = np.array([
    cpu_runtime[np.where(params_cpu == p)[0][0]]
    for p in common_params
])

gpu_total_common = np.array([
    gpu_total[np.where(params_gpu == p)[0][0]]
    for p in common_params
])

speedup = cpu_runtime / gpu_total_common
plt.figure(figsize=(8, 5))

plt.plot(common_params, speedup, marker='o', label='GPU Speedup')

plt.xscale('log', base=2)
plt.yscale('log')

plt.xlabel('Parameter Number')
plt.ylabel('Speedup (CPU / GPU)')
plt.title('GPU Speedup vs Parameter Number')
plt.grid(True, which='both')
plt.legend()

plt.savefig('fig_gpu_speedup.png', dpi=300, bbox_inches='tight')
