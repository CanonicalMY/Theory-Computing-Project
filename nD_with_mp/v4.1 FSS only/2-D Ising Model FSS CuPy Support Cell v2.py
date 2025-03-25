"""
CuPy 2D Ising Model with Two Update Methods:
-------------------------------------------
1) Checkerboard (vectorized) update
2) Random-site (asynchronous) update

You can choose which method to use by setting 'update_method' in the main section.
"""

import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------
#  Spin Initialization
# ---------------------------------------------------------------------
def initial_config_cupy(random_init, lattice_size):
    """
    Allocate a 2D spin lattice on the GPU using CuPy arrays.
    Spins stored in float32 to avoid potential integer overflow.
    """
    if random_init:
        # Generate random 0 or 1, then map to {-1, +1}
        spin_int = cp.random.randint(0, 2, size=(lattice_size, lattice_size), dtype=cp.int8)
        spin = spin_int.astype(cp.float32)
        spin = 2 * spin - 1  # Now in {-1, +1}
    else:
        # All spins = +1.0 (float32)
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.float32)
    return spin

# ---------------------------------------------------------------------
#  Energy Calculation (vectorized)
# ---------------------------------------------------------------------
def energy_2D_cupy_vectorized(spin, J, B, periodic=True):
    """
    Compute total energy using cp.roll for periodic boundary conditions.
    spin in [-1, +1].
    """
    N = spin.shape[0]
    if periodic:
        right = cp.roll(spin, shift=-1, axis=1)
        down  = cp.roll(spin, shift=-1, axis=0)
    else:
        # Non-periodic edges (not optimized)
        right = cp.zeros_like(spin)
        right[:, :-1] = spin[:, 1:]
        down = cp.zeros_like(spin)
        down[:-1, :] = spin[1:, :]

    E = -J * cp.sum(spin * (right + down)) - B * cp.sum(spin)
    return E

# ---------------------------------------------------------------------
#  Magnetization (absolute)
# ---------------------------------------------------------------------
def M_2D_cupy(spin):
    """
    Compute the absolute magnetization = |sum of spins| / (N*N).
    """
    return cp.abs(spin).sum() / spin.size

# ---------------------------------------------------------------------
#  Checkerboard (Synchronous) Update
# ---------------------------------------------------------------------
def checkerboard_update(spin, T, J, B, periodic, parity):
    """
    Vectorized checkerboard update for one sublattice (parity=0 or 1).
    spin is float32 in [-1, +1].
    """
    N = spin.shape[0]
    i_indices = cp.arange(N).reshape(N, 1)
    j_indices = cp.arange(N).reshape(1, N)
    mask = ((i_indices + j_indices) % 2) == parity

    top    = cp.roll(spin,  1, axis=0)
    bottom = cp.roll(spin, -1, axis=0)
    left   = cp.roll(spin,  1, axis=1)
    right  = cp.roll(spin, -1, axis=1)
    neighbor_sum = top + bottom + left + right

    # ΔE = 2 * spin * (J * neighbor_sum + B)
    deltaE = 2.0 * spin * (J * neighbor_sum + B)

    rnd = cp.random.rand(N, N)
    accept = (deltaE <= 0) | (rnd < cp.exp(-deltaE / T))

    flip_mask = mask & accept
    spin[flip_mask] = -spin[flip_mask]

def step_2D_cupy_vectorized(spin, steps_MC, T, J, B, periodic=True):
    """
    Perform steps_MC Monte Carlo sweeps using checkerboard updates.
    Each sweep updates sublattice=0, then sublattice=1.
    """
    for _ in range(steps_MC):
        checkerboard_update(spin, T, J, B, periodic, parity=0)
        checkerboard_update(spin, T, J, B, periodic, parity=1)

# ---------------------------------------------------------------------
#  Random-Site (Asynchronous) Update
# ---------------------------------------------------------------------
def random_site_update(spin, T, J, B, periodic):
    """
    Attempt a Metropolis spin flip at one random site (i, j).
    """
    N = spin.shape[0]
    i = cp.random.randint(0, N)
    j = cp.random.randint(0, N)

    # Compute neighbor sum
    top    = spin[(i-1) % N, j] if periodic else spin[max(i-1, 0), j]
    bottom = spin[(i+1) % N, j] if periodic else spin[min(i+1, N-1), j]
    left   = spin[i, (j-1) % N] if periodic else spin[i, max(j-1, 0)]
    right  = spin[i, (j+1) % N] if periodic else spin[i, min(j+1, N-1)]
    neighbor_sum = top + bottom + left + right

    dE = 2.0 * spin[i, j] * (J * neighbor_sum + B)

    if dE <= 0:
        spin[i, j] = -spin[i, j]
    else:
        r = cp.random.rand()
        if r < cp.exp(-dE / T):
            spin[i, j] = -spin[i, j]

def step_2D_cupy_random(spin, steps_MC, T, J, B, periodic=True):
    """
    Perform steps_MC Monte Carlo sweeps using random-site updates.
    Each sweep attempts N*N random flips.
    """
    N = spin.shape[0]
    for _ in range(steps_MC):
        for __ in range(N*N):
            random_site_update(spin, T, J, B, periodic)

# ---------------------------------------------------------------------
#  Main Simulation Function
# ---------------------------------------------------------------------
def simulation_at_T_cupy(
    T_value, lattice_size, steps_MC, n_runs, J, B,
    periodic=True, random_init=True,
    update_method="checkerboard"
):
    """
    Run the simulation at temperature T_value on the GPU using either
    checkerboard or random-site updates.
    Repeat n_runs times for averaging.
    """
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        spin = initial_config_cupy(random_init, lattice_size)

        # Perform Monte Carlo sweeps
        if update_method == "checkerboard":
            step_2D_cupy_vectorized(spin, steps_MC, T_value, J, B, periodic)
        else:
            # "random"
            step_2D_cupy_random(spin, steps_MC, T_value, J, B, periodic)

        m = M_2D_cupy(spin)
        e = energy_2D_cupy_vectorized(spin, J, B, periodic)
        M_sum += float(m.get())
        E_sum += float(e.get())

    M_avg = M_sum / n_runs
    E_avg = E_sum / n_runs
    return T_value, M_avg, E_avg

# ---------------------------------------------------------------------
#  if __main__
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # User parameters
    lattice_size = 8
    random_init = True
    steps_MC = 10000      # Increase to ensure better equilibration
    n_runs = 1
    T_min = 1.8
    T_max = 2.8
    num_T = 5
    J = 1.0
    B = 0.0
    periodic = True

    # CHOOSE UPDATE METHOD: "checkerboard" or "random"
    update_method = "random"  # <--- Switch here

    # Prepare temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    print(f"\nRunning 2D Ising on GPU with L={lattice_size}, steps_MC={steps_MC}, n_runs={n_runs}")
    print(f"Update method = {update_method}")

    results = []
    for T in tqdm(T_vals):
        tval, mag, ener = simulation_at_T_cupy(
            T, lattice_size, steps_MC, n_runs, J, B,
            periodic, random_init, update_method=update_method
        )
        results.append((tval, mag, ener))

    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    # Save data
    out_file = f"output/Ising2D_GPU_{update_method}_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

    # Plot
    heat_capacity = np.gradient(energies, temps)

    plt.figure()
    plt.plot(temps, mags, 'o-', label='Magnetization')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.title(f"2D Ising (GPU, {update_method}) L={lattice_size}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(temps, energies, 'o-', label='Energy')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy")
    plt.title(f"Energy vs T (GPU, {update_method})")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(temps, heat_capacity, 'o-', label='Heat Capacity')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Heat Capacity")
    plt.title(f"Heat Capacity vs T (GPU, {update_method})")
    plt.legend()
    plt.show()
