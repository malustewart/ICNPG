import numpy as np

OUTPUTFILE_ROOT = "param_sweep"


def generate_param_sweep(N_a, N_g, N_w, N_L):
    a_range = np.logspace(-19.5,-19.9, N_a)
    gamma_range = np.logspace(-1, -0.1, N_g)
    W_range = np.logspace(-6, -5, N_w)
    L_range = np.logspace(-4.5, -2.5, N_L)

    a_s, Ws, Ls, gammas = np.meshgrid(a_range, W_range, L_range, gamma_range, indexing="ij") # indexing: [a, W, L, gamma]

    a_s = a_s.flatten()
    Ws = Ws.flatten()
    Ls = Ls.flatten()
    gammas = gammas.flatten()

    output_data = np.column_stack((
            gammas,
            a_s,
            Ws,
            Ls,
        ))

    N_x = N_a*N_g*N_L*N_w
    np.savetxt(
        f"{OUTPUTFILE_ROOT}_{N_x}.csv",
        output_data,
        delimiter=",",
        header="gamma,a_s,W,L",
        comments="",
    )

generate_param_sweep(N_a=1, N_g=2, N_w=2, N_L=2)
generate_param_sweep(N_a=1, N_g=4, N_w=2, N_L=2)
generate_param_sweep(N_a=1, N_g=4, N_w=4, N_L=2)
generate_param_sweep(N_a=1, N_g=4, N_w=4, N_L=4)
generate_param_sweep(N_a=1, N_g=8, N_w=4, N_L=4)
generate_param_sweep(N_a=1, N_g=8, N_w=8, N_L=4)
generate_param_sweep(N_a=1, N_g=8, N_w=8, N_L=8)
generate_param_sweep(N_a=1, N_g=16, N_w=8, N_L=8)
generate_param_sweep(N_a=1, N_g=16, N_w=16, N_L=8)
generate_param_sweep(N_a=1, N_g=16, N_w=16, N_L=16)
generate_param_sweep(N_a=1, N_g=32, N_w=16, N_L=16)
generate_param_sweep(N_a=1, N_g=32, N_w=32, N_L=16)
generate_param_sweep(N_a=1, N_g=32, N_w=32, N_L=32)
generate_param_sweep(N_a=2, N_g=32, N_w=32, N_L=32)
generate_param_sweep(N_a=4, N_g=32, N_w=32, N_L=32)