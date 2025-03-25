import cupy as cp
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Optional: set seeds for reproducibility
cp.random.seed(42)
np.random.seed(42)


def initial_config_cupy(random_init, lattice_size):
    """
    Allocate a 2D spin lattice on the GPU using CuPy arrays.
    Spins are stored as float32 in [-1, +1].
    """
    if random_init:
        spin_int = cp.random.randint(0, 2, size=(lattice_size, lattice_size), dtype=cp.int8)
        spin = spin_int.astype(cp.float32)
        spin = 2 * spin - 1  # Now in {-1, +1}
    else:
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.float32)
    return spin


def random_site_update(spin, T, J, B, periodic):
    """
    Perform a Metropolis update at a single random site.
    Returns True if the spin is flipped.
    """
    N = spin.shape[0]
    i = int(cp.random.randint(0, N))
    j = int(cp.random.randint(0, N))

    if periodic:
        up = spin[(i - 1) % N, j]
        down = spin[(i + 1) % N, j]
        left = spin[i, (j - 1) % N]
        right = spin[i, (j + 1) % N]
    else:
        up = spin[max(i - 1, 0), j]
        down = spin[min(i + 1, N - 1), j]
        left = spin[i, max(j - 1, 0)]
        right = spin[i, min(j + 1, N - 1)]
    neighbor_sum = up + down + left + right

    dE = 2.0 * spin[i, j] * (J * neighbor_sum + B)
    accepted = False
    if dE <= 0:
        spin[i, j] = -spin[i, j]
        accepted = True
    else:
        r = cp.random.rand()
        if r < cp.exp(-dE / T):
            spin[i, j] = -spin[i, j]
            accepted = True
    return accepted


def step_2D_cupy_random(spin, steps_MC, T, J, B, periodic=True):
    """
    Perform 'steps_MC' Monte Carlo sweeps using random-site updates.
    Each sweep consists of N*N attempted updates.
    """
    N = spin.shape[0]
    for _ in range(steps_MC):
        for _ in range(N * N):
            random_site_update(spin, T, J, B, periodic)


def M_2D_cupy(spin):
    """
    Compute the absolute magnetization = |sum of spins| / (N*N).
    """
    return cp.abs(cp.sum(spin)) / spin.size


def E_2D_cupy(spin, J, B, periodic=True):
    """
    Compute total energy using cp.roll for periodic boundary conditions.
    """
    N = spin.shape[0]
    if periodic:
        right = cp.roll(spin, shift=-1, axis=1)
        down = cp.roll(spin, shift=-1, axis=0)
    else:
        right = cp.zeros_like(spin)
        right[:, :-1] = spin[:, 1:]
        down = cp.zeros_like(spin)
        down[:-1, :] = spin[1:, :]
    E = -J * cp.sum(spin * (right + down)) - B * cp.sum(spin)
    return E


def simulation_at_T_random(T_value, lattice_size, steps_MC, n_runs, J, B,
                           periodic=True, random_init=True, warm_up_steps=0):
    """
    Run the simulation at temperature T_value on the GPU using random-site updates.
    Optionally include a warm-up phase.
    Returns (T_value, average magnetization, average energy).
    """
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        spin = initial_config_cupy(random_init, lattice_size)
        if warm_up_steps > 0:
            step_2D_cupy_random(spin, warm_up_steps, T_value, J, B, periodic)
        step_2D_cupy_random(spin, steps_MC, T_value, J, B, periodic)
        m = M_2D_cupy(spin)
        e = E_2D_cupy(spin, J, B, periodic)
        M_sum += float(m.get())
        E_sum += float(e.get())
    M_avg = M_sum / n_runs
    E_avg = E_sum / n_runs
    return T_value, M_avg, E_avg


if __name__ == "__main__":
    # Simulation parameters
    lattice_size = 64
    random_init = True
    steps_MC = 1000  # Number of MC sweeps (increase if needed for equilibration)
    n_runs = 50
    T_min = 1.8
    T_max = 2.8
    num_T = 50
    J = 1.0
    B = 0.0
    periodic = True
    warm_up_steps = 0

    # Create a temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    # Run simulation for each temperature and collect results
    results = []
    for T in tqdm(T_vals, desc="Simulating temperatures"):
        tval, mag, ener = simulation_at_T_random(T, lattice_size, steps_MC, n_runs, J, B,
                                                 periodic, random_init, warm_up_steps)
        results.append((tval, mag, ener))
    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    # Save simulation data
    out_file = f"output/Ising2D_GPU_random_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

    # Compute heat capacity (using finite differences)
    heat_capacity = np.gradient(energies, temps)

    # Plot Magnetization vs Temperature
    plt.figure()
    plt.plot(temps, mags, 'o-', label='Magnetization')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.title(f"2D Ising (GPU, random-site) L={lattice_size}")
    plt.legend()
    plt.show()

    # Plot Energy vs Temperature
    plt.figure()
    plt.plot(temps, energies, 'o-', label='Energy')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy")
    plt.title("Energy vs T (GPU, random-site)")
    plt.legend()
    plt.show()

    # Plot Heat Capacity vs Temperature
    plt.figure()
    plt.plot(temps, heat_capacity, 'o-', label='Heat Capacity')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Heat Capacity")
    plt.title("Heat Capacity vs T (GPU, random-site)")
    plt.legend()
    plt.show()
