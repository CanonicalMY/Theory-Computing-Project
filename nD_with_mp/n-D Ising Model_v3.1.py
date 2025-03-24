import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys
import itertools
import os  # ### CHANGE: Added to enable file existence checking for FSS data

# Global parameters
default_J = 1.0  # Coupling constant; for J > 0, spins tend to align
default_B = 1.0  # External magnetic field
default_periodic = True  # The periodic boundary conditions


# ---------------------- Lattice Initialization ----------------------
def initial_config(random_init, lattice_size, dimension):
    if dimension == 1:
        if random_init:
            return (np.random.randint(2, size=lattice_size) - 0.5) * 2
        else:
            return np.ones(lattice_size)
    elif dimension == 2:
        N = lattice_size
        if random_init:
            return (np.random.randint(2, size=(N, N)) - 0.5) * 2
        else:
            return np.ones((N, N))
    elif dimension == 3:
        N = lattice_size
        if random_init:
            return (np.random.randint(2, size=(N, N, N)) - 0.5) * 2
        else:
            return np.ones((N, N, N))
    else:
        raise ValueError("Dimension must be 1, 2, or 3.")


# ---------------------- 1D Model Functions ----------------------
def Energy_1D(Spin, J, B, periodic):
    E = 0
    N = Spin.size
    for i in range(N - 1):
        E -= J * Spin[i] * Spin[i + 1]
        E -= B * Spin[i]
    E -= B * Spin[-1]
    if periodic:
        E -= J * Spin[-1] * Spin[0]
    return E


def flip_1D(Spin, i, T, J, B, periodic):
    S0 = np.copy(Spin)
    E0 = Energy_1D(Spin, J, B, periodic)
    Spin[i] = -Spin[i]
    dE = Energy_1D(Spin, J, B, periodic) - E0
    if dE < 0:
        return Spin
    else:
        if T != 0 and np.random.uniform(0, 1) < np.exp(-dE / T):
            return Spin
        else:
            return S0


def step_1D(Spin, n, T, J, B, periodic):
    N = Spin.size
    for _ in range(n):
        for _ in range(N):
            idx = np.random.randint(N)
            Spin = flip_1D(Spin, idx, T, J, B, periodic)
    return Spin


def M_1D(Spin):
    return np.sum(Spin) / Spin.size


# ---------------------- 2D Model Functions ----------------------
def local_energy_2D(Spin, i, j, J, B, periodic):
    N = Spin.shape[0]
    s = Spin[i, j]
    E_local = 0
    i_up = (i - 1) % N if periodic else max(i - 1, 0)
    i_down = (i + 1) % N if periodic else min(i + 1, N - 1)
    j_left = (j - 1) % N if periodic else max(j - 1, 0)
    j_right = (j + 1) % N if periodic else min(j + 1, N - 1)
    E_local -= J * s * Spin[i_up, j]
    E_local -= J * s * Spin[i_down, j]
    E_local -= J * s * Spin[i, j_left]
    E_local -= J * s * Spin[i, j_right]
    E_local -= B * s
    return E_local


def flip_2D(Spin, i, j, T, J, B, periodic):
    E_before = local_energy_2D(Spin, i, j, J, B, periodic)
    Spin[i, j] = -Spin[i, j]
    E_after = local_energy_2D(Spin, i, j, J, B, periodic)
    dE = E_after - E_before
    if dE <= 0:
        return Spin
    else:
        if T != 0 and np.random.uniform(0, 1) < np.exp(-dE / T):
            return Spin
        else:
            Spin[i, j] = -Spin[i, j]
            return Spin


def step_2D(Spin, n, T, J, B, periodic):
    N = Spin.shape[0]
    for _ in range(n):
        for _ in range(N * N):
            i_rand = np.random.randint(0, N)
            j_rand = np.random.randint(0, N)
            Spin = flip_2D(Spin, i_rand, j_rand, T, J, B, periodic)
    return Spin


def M_2D(Spin):
    return np.sum(Spin) / Spin.size


def Energy_2D(Spin, J, B, periodic):
    N = Spin.shape[0]
    E_total = 0
    for i in range(N):
        for j in range(N):
            E_total -= J * Spin[i, j] * Spin[i, (j + 1) % N]
            E_total -= J * Spin[i, j] * Spin[(i + 1) % N, j]
            E_total -= B * Spin[i, j]
    return E_total


# ---------------------- 3D Model Functions ----------------------
def local_energy_3D(Spin, x, y, z, J, B, periodic):
    N = Spin.shape[0]
    s = Spin[x, y, z]
    E_local = 0
    x_left = (x - 1) % N if periodic else max(x - 1, 0)
    x_right = (x + 1) % N if periodic else min(x + 1, N - 1)
    y_down = (y - 1) % N if periodic else max(y - 1, 0)
    y_up = (y + 1) % N if periodic else min(y + 1, N - 1)
    z_back = (z - 1) % N if periodic else max(z - 1, 0)
    z_front = (z + 1) % N if periodic else min(z + 1, N - 1)
    E_local -= J * s * Spin[x_left, y, z]
    E_local -= J * s * Spin[x_right, y, z]
    E_local -= J * s * Spin[x, y_down, z]
    E_local -= J * s * Spin[x, y_up, z]
    E_local -= J * s * Spin[x, y, z_back]
    E_local -= J * s * Spin[x, y, z_front]
    E_local -= B * s
    return E_local


def flip_3D(Spin, x, y, z, T, J, B, periodic):
    E_before = local_energy_3D(Spin, x, y, z, J, B, periodic)
    Spin[x, y, z] = -Spin[x, y, z]
    E_after = local_energy_3D(Spin, x, y, z, J, B, periodic)
    dE = E_after - E_before
    if dE <= 0:
        return Spin
    else:
        if T != 0 and np.random.uniform(0, 1) < np.exp(-dE / T):
            return Spin
        else:
            Spin[x, y, z] = -Spin[x, y, z]
            return Spin


def step_3D(Spin, n, T, J, B, periodic):
    N = Spin.shape[0]
    for _ in range(n):
        for _ in range(N ** 3):
            x_rand = np.random.randint(0, N)
            y_rand = np.random.randint(0, N)
            z_rand = np.random.randint(0, N)
            Spin = flip_3D(Spin, x_rand, y_rand, z_rand, T, J, B, periodic)
    return Spin


def M_3D(Spin):
    return np.sum(Spin) / Spin.size


def Energy_3D(Spin, J, B, periodic):
    N = Spin.shape[0]
    E_total = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                E_total -= J * Spin[x, y, z] * Spin[(x + 1) % N, y, z]
                E_total -= J * Spin[x, y, z] * Spin[x, (y + 1) % N, z]
                E_total -= J * Spin[x, y, z] * Spin[x, y, (z + 1) % N]
                E_total -= B * Spin[x, y, z]
    return E_total


# ---------------------- Generic Simulation Function ----------------------
def simulation_at_T_generic(T_value, dimension, lattice_size, steps_MC, n_runs, J, B, periodic, random_init):
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        if dimension == 1:
            Spin = initial_config(random_init, lattice_size, 1)
            Spin = step_1D(Spin, steps_MC, T_value, J, B, periodic)
            M_sum += M_1D(Spin)
            E_sum += Energy_1D(Spin, J, B, periodic)
        elif dimension == 2:
            Spin = initial_config(random_init, lattice_size, 2)
            Spin = step_2D(Spin, steps_MC, T_value, J, B, periodic)
            M_sum += M_2D(Spin)
            E_sum += Energy_2D(Spin, J, B, periodic)
        elif dimension == 3:
            Spin = initial_config(random_init, lattice_size, 3)
            Spin = step_3D(Spin, steps_MC, T_value, J, B, periodic)
            M_sum += M_3D(Spin)
            E_sum += Energy_3D(Spin, J, B, periodic)
    return (T_value, M_sum / n_runs, E_sum / n_runs)


# ---------------------- Weighted Cost Function for FSS ----------------------
def collapse_cost_weighted(params, data_dict, x0=1.0, nbins=20):
    """
    A weighted cost function where each point is weighted by exp(-(x/x0)^2).

    Parameters:
      params   : tuple (T_c, beta, nu)
      data_dict: dictionary {L: (temps_L, mags_L)}
      x0       : scaling parameter controlling the decay of the weight (default 1.0)
      nbins    : number of bins to group the scaled x values (default 20)

    Returns:
      cost value: lower cost means better collapse.
    """
    T_c, beta, nu = params
    x_all = []
    y_all = []
    weights = []
    for L_val, (temps_L, mags_L) in data_dict.items():
        for T_i, M_i in zip(temps_L, mags_L):
            x_val = (T_i - T_c) * (L_val ** (1.0 / nu))
            y_val = M_i * (L_val ** (beta / nu))
            w = np.exp(- (x_val / x0) ** 2)
            x_all.append(x_val)
            y_all.append(y_val)
            weights.append(w)

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    weights = np.array(weights)

    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]
    weights_sorted = weights[sort_idx]

    x_min, x_max = np.min(x_sorted), np.max(x_sorted)
    bin_edges = np.linspace(x_min, x_max, nbins + 1)

    sum_sq = 0.0
    weight_sum = 0.0
    for i in range(nbins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        mask = (x_sorted >= left) & (x_sorted < right)
        if np.sum(mask) > 1:
            y_bin = y_sorted[mask]
            w_bin = weights_sorted[mask]
            # NEW: Skip bin if the sum of weights is zero to avoid division by zero
            if np.sum(w_bin) == 0:
                continue
            y_mean = np.average(y_bin, weights=w_bin)
            sum_sq += np.sum(w_bin * (y_bin - y_mean) ** 2)
            weight_sum += np.sum(w_bin)

    if weight_sum > 0:
        return sum_sq / weight_sum
    else:
        return 1e9


# Original cost function remains available (for comparison)
def collapse_cost(params, data_dict):
    T_c, beta, nu = params
    x_all = []
    y_all = []
    for L_val, (temps_L, mags_L) in data_dict.items():
        for T_i, M_i in zip(temps_L, mags_L):
            x_val = (T_i - T_c) * (L_val ** (1.0 / nu))
            y_val = M_i * (L_val ** (beta / nu))
            x_all.append(x_val)
            y_all.append(y_val)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]

    nbins = 20
    x_min, x_max = x_sorted[0], x_sorted[-1]
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


# ---------------------- User Entry (Default Parameters) ----------------------
if __name__ == '__main__':
    ### CHANGE 1: Instead of interactive user input, we use default built-in parameters for testing.
    dimension = 2  # default 2
    lattice_size = 8  # default 8
    random_init = False  # default False
    steps_MC = 100  # default 100
    n_runs = 100  # default 100
    T_min = 0.0  # default 0
    T_max = 5.0  # default 5
    num_T = 25  # default 50
    J = 1.0  # default 1
    B = 0.0  # default 0

    # Generate temperature array:
    T_vals = np.linspace(T_min, T_max, num_T)

    sim_func = partial(simulation_at_T_generic,
                       dimension=dimension,
                       lattice_size=lattice_size,
                       steps_MC=steps_MC,
                       n_runs=n_runs,
                       J=J,
                       B=B,
                       periodic=default_periodic,
                       random_init=random_init)

    with mp.Pool() as pool:
        results = list(tqdm(pool.imap(sim_func, T_vals), total=len(T_vals)))

    temps = np.array([r[0] for r in results])
    mags = np.array([r[1] for r in results])
    energies = np.array([r[2] for r in results])
    heat_capacity = np.gradient(energies, temps)

    plt.figure()
    plt.scatter(temps, mags)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Magnetization (M)")
    plt.title(f"{dimension}D Ising Model (Lattice size = {lattice_size})")
    plt.savefig(f"output/Ising_{dimension}D_M_vs_T.jpg")
    plt.show()

    plt.figure()
    plt.scatter(temps, energies)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Energy (E)")
    plt.title("Energy vs Temperature")
    plt.savefig(f"output/Ising_{dimension}D_E_vs_T.jpg")
    plt.show()

    plt.figure()
    plt.scatter(temps, heat_capacity)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Heat Capacity (C)")
    plt.title("Heat Capacity vs Temperature")
    plt.savefig(f"output/Ising_{dimension}D_C_vs_T.jpg")
    plt.show()

    # Hysteresis Loop (optional)
    do_hysteresis = "0"  # Set to "1" to run hysteresis; here default is off.
    if do_hysteresis.strip() == "1":
        T_fixed = 2.0
        B_max = 2.0
        num_B = 50
        B_vals_up = np.linspace(-B_max, B_max, num_B)
        B_vals_down = np.linspace(B_max, -B_max, num_B)

        if dimension == 1:
            Spin = initial_config(random_init, lattice_size, 1)
        elif dimension == 2:
            Spin = initial_config(random_init, lattice_size, 2)
        elif dimension == 3:
            Spin = initial_config(random_init, lattice_size, 3)

        magnetizations_up = []
        for B_val in tqdm(B_vals_up, desc="Upward sweep"):
            if dimension == 1:
                Spin = step_1D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_up.append(M_1D(Spin))
            elif dimension == 2:
                Spin = step_2D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_up.append(M_2D(Spin))
            elif dimension == 3:
                Spin = step_3D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_up.append(M_3D(Spin))

        magnetizations_down = []
        for B_val in tqdm(B_vals_down, desc="Downward sweep"):
            if dimension == 1:
                Spin = step_1D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_down.append(M_1D(Spin))
            elif dimension == 2:
                Spin = step_2D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_down.append(M_2D(Spin))
            elif dimension == 3:
                Spin = step_3D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_down.append(M_3D(Spin))

        plt.figure()
        plt.plot(B_vals_up, magnetizations_up, 'o-', label="Increasing B")
        plt.plot(B_vals_down, magnetizations_down, 'o-', label="Decreasing B")
        plt.xlabel("External Field B")
        plt.ylabel("Magnetization M")
        plt.title(f"Hysteresis Loop at T = {T_fixed}")
        plt.legend()
        plt.savefig(f"output/Ising_{dimension}D_Hysteresis.jpg")
        plt.show()

    # ---------------------- Finite-Size Scaling (FSS) Analysis ----------------------
    do_fss = "1"  # Set to "1" to run FSS analysis by default.
    if do_fss.strip() == "1":
        lattice_sizes = [8, 16, 32, 64]
        T_vals = np.linspace(T_min, T_max, num_T)
        data = {}

        # ### CHANGE 2: Save and load FSS data for each lattice size
        for L in lattice_sizes:
            data_file = f"FSS_data_L{L}.npz"
            if os.path.exists(data_file):
                print(f"Loading data for lattice size {L} from {data_file}...")
                loaded = np.load(data_file)
                temps_L = loaded['temps']
                mags_L = loaded['mags']
            else:
                print(f"\n[Finite-Size Scaling] Running simulation for lattice size = {L} ...")
                sim_func_local = partial(simulation_at_T_generic,
                                         dimension=dimension,
                                         lattice_size=L,
                                         steps_MC=steps_MC,
                                         n_runs=n_runs,
                                         J=J,
                                         B=B,
                                         periodic=default_periodic,
                                         random_init=random_init)
                with mp.Pool() as pool:
                    results_L = list(tqdm(pool.imap(sim_func_local, T_vals), total=len(T_vals)))
                temps_L = np.array([r[0] for r in results_L])
                mags_L = np.array([r[1] for r in results_L])
                np.savez(data_file, temps=temps_L, mags=mags_L)
            data[L] = (temps_L, mags_L)

        # Ask user if they want to use the weighted cost function
        use_weighted = True  # For testing, we set this to True by default.
        if use_weighted:
            try:
                x0 = 0.5  # For testing, default x0 is set to 0.5
            except Exception as e:
                print("Invalid input for x0. Using default value 1.0.")
                x0 = 1.0

        # Grid search parameters for (T_c, beta, nu)
        # For this testing version, we use built-in defaults.
        T_c_range = np.linspace(2, 3, 100)
        beta_range = np.linspace(0.075, 0.275, 50)
        nu_range = np.linspace(0.5, 2, 50)

        best_params = (None, None, None)
        best_cost = 1e9
        grid = list(itertools.product(T_c_range, beta_range, nu_range))
        print(f"\n[Finite-Size Scaling] Searching for best (T_c, beta, nu) over {len(grid)} combinations...")
        for T_c_guess, beta_guess, nu_guess in tqdm(grid, desc="Grid Search for FSS parameters", total=len(grid)):
            if use_weighted:
                c = collapse_cost_weighted((T_c_guess, beta_guess, nu_guess), data, x0=x0, nbins=20)
            else:
                c = collapse_cost((T_c_guess, beta_guess, nu_guess), data)
            if c < best_cost:
                best_cost = c
                best_params = (T_c_guess, beta_guess, nu_guess)

        T_c_opt, beta_opt, nu_opt = best_params
        print(
            f"\nBest collapse parameters found:\n  T_c  = {T_c_opt:.4f}\n  beta = {beta_opt:.4f}\n  nu   = {nu_opt:.4f}\n  cost = {best_cost:.6f}")

        plt.figure()
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, L_val in enumerate(lattice_sizes):
            temps_L, mags_L = data[L_val]
            x_vals = (temps_L - T_c_opt) * (L_val ** (1.0 / nu_opt))
            y_vals = mags_L * (L_val ** (beta_opt / nu_opt))
            col = color_list[i % len(color_list)]
            plt.plot(x_vals, y_vals, 'o', label=f"L={L_val}", color=col)
        plt.xlabel("x = (T - T_c) * L^(1/nu)")
        plt.ylabel("y = M * L^(beta/nu)")
        plt.title("Data Collapse for Finite-Size Scaling")
        plt.legend()
        plt.savefig("output/FSS Data Collapse.jpg")
        plt.show()
