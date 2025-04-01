import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os


def collapse_cost(params, data_dict, nbins=15, X_max=0.16):
    """
    Unweighted cost function for data collapse with a window.

    For each lattice size L, the simulation data (temps, mags) given by
      data_dict[L] = (T_array, M_array)
    is transformed as:
         x = (T - T_c) * L^(1/nu)
         y = M * L^(beta/nu)
    Only data points with |x| < X_max are retained.
    Then, the x-range is divided into nbins bins. In each bin, the variance of y
    values is computed, and the overall cost is the average variance.

    Parameters:
      params: tuple (T_c, beta, nu)
      data_dict: dictionary mapping lattice size L to (T_array, M_array)
      nbins: number of bins (default 15)
      X_max: maximum |x| to include (default 0.16)

    Returns:
      cost: average variance of y values over the bins.
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

    # Only keep data points with |x| < X_max
    mask = np.abs(x_all) < X_max
    x_all = x_all[mask]
    y_all = y_all[mask]

    if len(x_all) == 0:
        return 1e9

    # Sort data by x
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]

    # Bin the data
    x_min = x_sorted[0]
    x_max_val = x_sorted[-1]
    bin_edges = np.linspace(x_min, x_max_val, nbins + 1)

    sum_sq = 0.0
    count = 0
    for i in range(nbins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        mask_bin = (x_sorted >= left) & (x_sorted < right)
        y_bin = y_sorted[mask_bin]
        if len(y_bin) > 1:
            y_mean = np.mean(y_bin)
            sum_sq += np.sum((y_bin - y_mean) ** 2)
            count += len(y_bin)
    if count > 0:
        return sum_sq / count
    else:
        return 1e9


if __name__ == "__main__":
    # Lattice sizes for which simulation data exists
    lattice_sizes = [16, 32, 64, 128]
    data = {}

    # Load data from .npz files (each file must contain arrays "temps" and "mags")
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
    beta_range = np.linspace(0.1, 0.15, 50)
    nu_range = np.linspace(0.5, 1.5, 50)

    # Two sets of window settings to compare:
    search_settings = [
        {"nbins": 15, "X_max": 0.16},
        {"nbins": 15, "X_max": 0.15}
    ]

    results_params = []  # To store best parameters for each setting

    for setting in search_settings:
        nbins = setting["nbins"]
        X_max = setting["X_max"]
        best_cost = 1e9
        best_params = (None, None, None)
        grid = list(itertools.product(T_c_range, beta_range, nu_range))
        print(f"FSS: Searching over {len(grid)} combinations for X_max={X_max}, nbins={nbins} ...")
        for (T_c_guess, beta_guess, nu_guess) in tqdm(grid):
            cost = collapse_cost((T_c_guess, beta_guess, nu_guess), data, nbins=nbins, X_max=X_max)
            if cost < best_cost:
                best_cost = cost
                best_params = (T_c_guess, beta_guess, nu_guess)
        results_params.append({
            "X_max": X_max,
            "nbins": nbins,
            "T_c": best_params[0],
            "beta": best_params[1],
            "nu": best_params[2],
            "cost": best_cost
        })

    # Print results for both settings
    for res in results_params:
        print(
            f"X_max={res['X_max']}, nbins={res['nbins']} -> T_c={res['T_c']:.3f}, beta={res['beta']:.3f}, nu={res['nu']:.3f}, cost={res['cost']:.6e}")

    # Create a figure with 2 side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for ax, res in zip(axes, results_params):
        T_c_opt = res["T_c"]
        beta_opt = res["beta"]
        nu_opt = res["nu"]
        # Plot collapsed data for each lattice size
        for i, L in enumerate(lattice_sizes):
            if L not in data:
                continue
            temps_L, mags_L = data[L]
            x_vals = (temps_L - T_c_opt) * (L ** (1.0 / nu_opt))
            y_vals = mags_L * (L ** (beta_opt / nu_opt))
            ax.plot(x_vals, y_vals, 'o', color=color_list[i % len(color_list)], label=f"L={L}")
        ax.set_xlabel("x = (T-T_c)*L^(1/nu)")
        ax.set_ylabel("y = M*L^(beta/nu)")
        ax.set_title(
            f"X_max={res['X_max']}, nbins={res['nbins']}\nT_c={T_c_opt:.3f}, beta={beta_opt:.3f}, nu={nu_opt:.3f}\ncost={res['cost']:.2e}")
        ax.legend()

    plt.suptitle("FSS Data Collapse Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/FSS_Data_Collapse_Comparison.jpg")
    plt.show()

# For 2D Ising, known ~ T_c ~ 2.27, beta=1/8=0.125, nu=1
# For 3D Ising, known ~ T_c ~ 4.51, beta~0.326, nu~0.63