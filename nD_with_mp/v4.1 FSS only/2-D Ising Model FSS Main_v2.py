import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os

def collapse_cost(params, data_dict, nbins=50):
    """
    Unweighted cost function for data collapse.
    
    For each lattice size L, with simulation data (temps, mags) given by data_dict[L] = (T_array, M_array),
    we transform the data as:
    
         x = (T - T_c) * L^(1/nu)
         y = M * L^(beta/nu)
    
    Then, the x-range is divided into nbins bins. In each bin, we compute the variance
    of the y values (without weighting) and average over all bins.
    
    Parameters:
      params: tuple (T_c, beta, nu)
      data_dict: dictionary mapping lattice size L to (T_array, M_array)
      nbins: number of bins to divide the x-axis (default 50)
    
    Returns:
      cost: average variance of y values over the bins
    """
    T_c, beta, nu = params
    x_all = []
    y_all = []
    
    for L, (T_arr, M_arr) in data_dict.items():
        for T, M in zip(T_arr, M_arr):
            x_val = (T - T_c) * (L ** (1.0 / nu))
            y_val = M * (L ** (beta / nu))
            x_all.append(x_val)
            y_all.append(y_val)
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    
    # Sort the data by x
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]
    
    # Bin the data
    x_min = x_sorted[0]
    x_max = x_sorted[-1]
    bin_edges = np.linspace(x_min, x_max, nbins + 1)
    
    sum_sq = 0.0
    count = 0
    for i in range(nbins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        mask = (x_sorted >= left) & (x_sorted < right)
        y_bin = y_sorted[mask]
        if len(y_bin) > 1:
            y_mean = np.mean(y_bin)
            sum_sq += np.sum((y_bin - y_mean) ** 2)
            count += len(y_bin)
    if count > 0:
        return sum_sq / count
    else:
        return 1e9

if __name__ == "__main__":
    # Lattice sizes for which you have simulation data
    lattice_sizes = [16, 32, 64, 128]
    data = {}
    
    # Load simulation data from .npz files (each file should contain arrays "temps" and "mags")
    for L in lattice_sizes:
        file_L = f"output/Ising2D_GPU_vectorized_L{L}.npz"
        if not os.path.exists(file_L):
            print(f"File {file_L} not found. Please run your simulation code first.")
            continue
        loaded = np.load(file_L)
        temps_L = loaded["temps"]
        mags_L = loaded["mags"]
        data[L] = (temps_L, mags_L)
    
    # Define grid search ranges for T_c, beta, and nu.
    T_c_range = np.linspace(1.5, 3.0, 50)
    beta_range = np.linspace(0.05, 0.2, 30)
    nu_range = np.linspace(0.5, 2.0, 30)
    
    best_cost = 1e9
    best_params = (None, None, None)
    grid = list(itertools.product(T_c_range, beta_range, nu_range))
    print(f"FSS: Searching over {len(grid)} combinations...")
    
    for (T_c_guess, beta_guess, nu_guess) in tqdm(grid):
        cost = collapse_cost((T_c_guess, beta_guess, nu_guess), data, nbins=50)
        if cost < best_cost:
            best_cost = cost
            best_params = (T_c_guess, beta_guess, nu_guess)
    
    T_c_opt, beta_opt, nu_opt = best_params
    print(f"\nBest FSS parameters found:")
    print(f"  T_c = {T_c_opt:.3f}")
    print(f"  beta = {beta_opt:.3f}")
    print(f"  nu = {nu_opt:.3f}")
    print(f"  cost = {best_cost:.6e}")
    
    # Plot the collapsed data for each lattice size
    plt.figure()
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, L in enumerate(lattice_sizes):
        if L not in data:
            continue
        temps_L, mags_L = data[L]
        x_vals = (temps_L - T_c_opt) * (L ** (1.0 / nu_opt))
        y_vals = mags_L * (L ** (beta_opt / nu_opt))
        plt.plot(x_vals, y_vals, 'o', color=color_list[i % len(color_list)], label=f"L={L}")
    
    plt.xlabel("x = (T - T_c) * L^(1/nu)")
    plt.ylabel("y = M * L^(beta/nu)")
    plt.title("FSS Data Collapse (Unweighted)")
    plt.legend()
    plt.savefig("output/FSS_Data_Collapse_Unweighted.jpg")
    plt.show()

# For 2D Ising, known ~ T_c ~ 2.27, beta=1/8=0.125, nu=1
# For 3D Ising, known ~ T_c ~ 4.51, beta~0.326, nu~0.63
