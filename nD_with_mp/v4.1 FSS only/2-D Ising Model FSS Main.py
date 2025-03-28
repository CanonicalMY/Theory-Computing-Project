"""
FSS analysis main cell
===============
Loads Ising data (.npz) from multiple lattice sizes
and performs finite-size scaling (FSS) analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os

def collapse_cost_weighted(params, data_dict, x0=2.0, nbins=20):
    """
    Weighted cost function for data collapse:
      x = (T - T_c)*L^(1/nu),  y = M * L^(beta/nu)
      w(x) = exp(- (x/x0)^2 )
    """
    T_c, beta, nu = params
    x_all = []
    y_all = []
    weights = []

    for L_val, (temps_L, mags_L) in data_dict.items():
        for T_i, M_i in zip(temps_L, mags_L):
            x_val = (T_i - T_c)*(L_val**(1.0/nu))
            y_val = M_i*(L_val**(beta/nu))
            w = np.exp(-(x_val/x0)**2)
            x_all.append(x_val)
            y_all.append(y_val)
            weights.append(w)

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    weights = np.array(weights)

    # Sort by x
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]
    w_sorted = weights[sort_idx]

    x_min, x_max = x_sorted[0], x_sorted[-1]
    bin_edges = np.linspace(x_min, x_max, nbins + 1)

    sum_sq = 0.0
    weight_sum = 0.0
    for i in range(nbins):
        left = bin_edges[i]
        right = bin_edges[i+1]
        mask = (x_sorted >= left) & (x_sorted < right)
        if np.any(mask):
            y_bin = y_sorted[mask]
            w_bin = w_sorted[mask]
            w_sum = np.sum(w_bin)
            if w_sum > 0:
                y_mean = np.average(y_bin, weights=w_bin)
                sum_sq += np.sum(w_bin*(y_bin - y_mean)**2)
                weight_sum += w_sum

    if weight_sum > 0:
        return sum_sq/weight_sum
    else:
        return 1e9

if __name__ == "__main__":
    # Lattice sizes used in your GPU simulation
    lattice_sizes = [16, 32, 64, 128]
    data = {}

    # Load data from .npz for each L
    for L in lattice_sizes:
        file_L = f"output/Ising2D_GPU_vectorized_L{L}.npz"
        if not os.path.exists(file_L):
            print(f"File {file_L} not found. Please run '2-D Ising Model FSS CuPy Support Cell.py' first.")
            continue
        loaded = np.load(file_L)
        temps_L = loaded["temps"]
        mags_L = loaded["mags"]
        data[L] = (temps_L, mags_L)

    # Grid search over T_c, beta, nu
    T_c_range = np.linspace(1.5, 3.0, 50)
    beta_range = np.linspace(0.05, 0.2, 30)
    nu_range = np.linspace(0.5, 2.0, 30)

    best_cost = 1e9
    best_params = (None, None, None)
    grid = list(itertools.product(T_c_range, beta_range, nu_range))
    print(f"FSS: Searching over {len(grid)} combos ...")
    for (T_c_guess, beta_guess, nu_guess) in tqdm(grid):
        cost = collapse_cost_weighted((T_c_guess, beta_guess, nu_guess),
                                      data_dict=data,
                                      x0=2.0, nbins=20)
        if cost < best_cost:
            best_cost = cost
            best_params = (T_c_guess, beta_guess, nu_guess)

    T_c_opt, beta_opt, nu_opt = best_params
    print(f"\nBest FSS params: T_c={T_c_opt:.3f}, beta={beta_opt:.3f}, nu={nu_opt:.3f}, cost={best_cost:.6e}")

    # Plot data collapse
    plt.figure()
    color_list = ['b', 'g', 'r', 'm', 'c', 'y']
    for i, L in enumerate(lattice_sizes):
        if L not in data:
            continue
        temps_L, mags_L = data[L]
        x_vals = (temps_L - T_c_opt)*(L**(1.0/nu_opt))
        y_vals = mags_L*(L**(beta_opt/nu_opt))
        plt.plot(x_vals, y_vals, 'o', color=color_list[i%len(color_list)], label=f"L={L}")

    plt.xlabel("x = (T - T_c)*L^(1/nu)")
    plt.ylabel("y = M * L^(beta/nu)")
    plt.title("FSS Data Collapse (GPU Data)")
    plt.legend()
    plt.savefig("output/FSS_v4_Data_Collapse.jpg")
    plt.show()


# For 2D Ising, known ~ T_c ~ 2.27, beta=1/8=0.125, nu=1
# For 3D Ising, known ~ T_c ~ 4.51, beta~0.326, nu~0.63