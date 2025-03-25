"""
CuPy-based 2D Ising with Vectorized Checkerboard Updates (Improved)
===================================================================
"""

import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def initial_config_cupy(random_init, lattice_size):
    """
    Allocate a 2D spin lattice on the GPU using CuPy arrays.
    Store spins in float32 to avoid potential int overflow or underflow.
    """
    if random_init:
        # Generate random 0 or 1 in int8, then map to {-1, +1} in float32
        spin_int = cp.random.randint(0, 2, size=(lattice_size, lattice_size), dtype=cp.int8)
        spin = spin_int.astype(cp.float32)
        spin = 2 * spin - 1  # Now in {-1, +1} as float32
    else:
        # All spins = +1.0 (float32)
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.float32)
    return spin


def energy_2D_cupy_vectorized(spin, J, B, periodic=True):
    """
    Compute total energy using cp.roll for periodic boundary conditions.
    spin is float32 in [-1, +1].
    """
    if periodic:
        right = cp.roll(spin, shift=-1, axis=1)
        down = cp.roll(spin, shift=-1, axis=0)
    else:
        # Non-periodic: just zero out the 'beyond' edges (not optimized)
        N = spin.shape[0]
        right = cp.zeros_like(spin)
        right[:, :-1] = spin[:, 1:]
        down = cp.zeros_like(spin)
        down[:-1, :] = spin[1:, :]

    # Sum neighbor contributions and subtract B field
    # Each bond is only counted once (right + down)
    E = -J * cp.sum(spin * (right + down)) - B * cp.sum(spin)
    return E


def checkerboard_update(spin, T, J, B, periodic, parity):
    """
    Vectorized checkerboard update for one sublattice (parity=0 or 1).
    spin is float32 in [-1, +1].
    """
    N = spin.shape[0]
    # Checkerboard mask: sublattice (i + j) % 2 == parity
    i_indices = cp.arange(N).reshape(N, 1)
    j_indices = cp.arange(N).reshape(1, N)
    mask = ((i_indices + j_indices) % 2) == parity

    # Compute neighbor sum for each site
    top    = cp.roll(spin,  1, axis=0)
    bottom = cp.roll(spin, -1, axis=0)
    left   = cp.roll(spin,  1, axis=1)
    right  = cp.roll(spin, -1, axis=1)
    neighbor_sum = top + bottom + left + right

    # Delta E = E_new - E_old = 2 * spin * (J * neighbor_sum + B)
    deltaE = 2.0 * spin * (J * neighbor_sum + B)

    # Generate random matrix for acceptance
    rnd = cp.random.rand(N, N)
    # Accept flip if deltaE <= 0 or rnd < exp(-deltaE / T)
    accept = (deltaE <= 0) | (rnd < cp.exp(-deltaE / T))

    # Only flip spins in the chosen sublattice
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


def M_2D_cupy(spin):
    """
    Compute the absolute magnetization = |sum of spins| / (N*N).
    spin is float32 in [-1, +1].
    """
    return cp.abs(spin).sum() / spin.size


def simulation_at_T_cupy_vectorized(T_value, lattice_size, steps_MC, n_runs, J, B,
                                    periodic=True, random_init=True):
    """
    Run the simulation at temperature T_value on the GPU using vectorized updates.
    Repeat n_runs times for averaging.
    """
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        spin = initial_config_cupy(random_init, lattice_size)
        # Perform Monte Carlo sweeps
        step_2D_cupy_vectorized(spin, steps_MC, T_value, J, B, periodic)
        m = M_2D_cupy(spin)
        e = energy_2D_cupy_vectorized(spin, J, B, periodic)
        # Bring results back to CPU
        M_sum += float(m.get())
        E_sum += float(e.get())
    M_avg = M_sum / n_runs
    E_avg = E_sum / n_runs
    return T_value, M_avg, E_avg


if __name__ == "__main__":
    # Simulation parameters
    lattice_size = 8
    random_init = True
    steps_MC = 1000     # Increased for better equilibration
    n_runs = 10
    T_min = 1.8
    T_max = 2.8
    num_T = 50
    J = 1.0
    B = 0.0
    periodic = True

    # Temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    # Run the simulation
    print(f"Running 2D Ising (GPU, vectorized) L={lattice_size}, steps_MC={steps_MC}, n_runs={n_runs} ...")
    results = []
    for T in tqdm(T_vals):
        tval, mag, ener = simulation_at_T_cupy_vectorized(
            T, lattice_size, steps_MC, n_runs, J, B, periodic, random_init
        )
        results.append((tval, mag, ener))

    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    # Save results
    out_file = f"output/Ising2D_GPU_vectorized_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

    # Plot
    heat_capacity = np.gradient(energies, temps)

    plt.figure()
    plt.plot(temps, mags, 'o-', label='Magnetization')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.title(f"2D Ising (GPU, vectorized) L={lattice_size}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(temps, energies, 'o-', label='Energy')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy")
    plt.title("Energy vs T (GPU, vectorized)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(temps, heat_capacity, 'o-', label='Heat Capacity')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Heat Capacity")
    plt.title("Heat Capacity vs T (GPU, vectorized)")
    plt.legend()
    plt.show()
