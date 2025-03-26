"""
Optimized CuPy-based 2D Ising Simulation with Vectorized Checkerboard Updates
=============================================================================
This code simulates the 2D Ising model on the GPU using CuPy.
It uses a vectorized checkerboard update and cp.roll for neighbor calculations.
Results (temperature, magnetization, energy) are saved to an .npz file and plotted.
"""

import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Optional: Set random seeds for reproducibility
cp.random.seed(42)
np.random.seed(42)


def initial_config_cupy(random_init, lattice_size):
    """
    Allocate a 2D spin lattice on the GPU using CuPy arrays.
    Spins are stored as float32 in [-1, +1].
    """
    if random_init:
        # Generate random integers 0 or 1, map them to -1 or +1 in float32.
        spin_int = cp.random.randint(0, 2, size=(lattice_size, lattice_size), dtype=cp.int8)
        spin = spin_int.astype(cp.float32)
        spin = 2 * spin - 1
    else:
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.float32)
    return spin


def energy_2D_cupy_vectorized(spin, J, B, periodic=True):
    """
    Compute total energy using cp.roll for periodic boundary conditions.
    Each bond is counted once (using right and down neighbors).
    """
    if periodic:
        right = cp.roll(spin, shift=-1, axis=1)
        down = cp.roll(spin, shift=-1, axis=0)
    else:
        N = spin.shape[0]
        right = cp.zeros_like(spin)
        right[:, :-1] = spin[:, 1:]
        down = cp.zeros_like(spin)
        down[:-1, :] = spin[1:, :]
    E = -J * cp.sum(spin * (right + down)) - B * cp.sum(spin)
    return E


def checkerboard_update(spin, T, J, B, periodic, parity):
    """
    Perform a vectorized update on the checkerboard sublattice given by parity (0 or 1).
    Updates spins in place using the Metropolis criterion.
    """
    N = spin.shape[0]
    # Create a mask for the sublattice: (i+j)%2 == parity.
    i_indices = cp.arange(N).reshape(N, 1)
    j_indices = cp.arange(N).reshape(1, N)
    mask = ((i_indices + j_indices) % 2) == parity

    # Compute neighbor contributions using periodic boundaries via cp.roll.
    top = cp.roll(spin, 1, axis=0)
    bottom = cp.roll(spin, -1, axis=0)
    left = cp.roll(spin, 1, axis=1)
    right = cp.roll(spin, -1, axis=1)
    neighbor_sum = top + bottom + left + right

    # Compute the energy change if a spin were flipped.
    deltaE = 2.0 * spin * (J * neighbor_sum + B)

    # Generate a random matrix for acceptance probability.
    rnd = cp.random.rand(N, N)
    accept = (deltaE <= 0) | (rnd < cp.exp(-deltaE / T))

    # Only update spins on the chosen sublattice.
    flip_mask = mask & accept
    spin[flip_mask] = -spin[flip_mask]


def step_2D_cupy_vectorized(spin, steps_MC, T, J, B, periodic=True):
    """
    Perform 'steps_MC' Monte Carlo sweeps using vectorized checkerboard updates.
    Each sweep consists of updating sublattice with parity 0 and then parity 1.
    """
    for _ in range(steps_MC):
        checkerboard_update(spin, T, J, B, periodic, parity=0)
        checkerboard_update(spin, T, J, B, periodic, parity=1)


def M_2D_cupy(spin):
    """
    Compute the absolute magnetization: |sum of spins| / (N*N).
    """
    return cp.abs(cp.sum(spin)) / spin.size


def simulation_at_T_cupy_vectorized(T_value, lattice_size, steps_MC, n_runs, J, B,
                                    periodic=True, random_init=True):
    """
    Run the simulation at temperature T_value on the GPU using vectorized checkerboard updates.
    Repeat n_runs times for averaging, then return (T, average magnetization, average energy).
    """
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        spin = initial_config_cupy(random_init, lattice_size)
        step_2D_cupy_vectorized(spin, steps_MC, T_value, J, B, periodic)
        m = M_2D_cupy(spin)
        e = energy_2D_cupy_vectorized(spin, J, B, periodic)
        M_sum += float(m.get())
        E_sum += float(e.get())
    M_avg = M_sum / n_runs
    E_avg = E_sum / n_runs
    return T_value, M_avg, E_avg


if __name__ == "__main__":
    # Simulation parameters
    lattice_size = 128  # Increase as needed
    random_init = False
    steps_MC = 20000  # Number of MC sweeps; adjust for equilibration # L=64, s=5000; 128,20000; 256,80000.
    n_runs = 20  # Number of independent runs for averaging
    T_min = 1.8  # Temperature range (choose appropriately)
    T_max = 2.8
    num_T = 100
    J = 1.0
    B = 0.0
    periodic = True

    # Create temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    # Run simulation for each temperature
    results = []
    for T in tqdm(T_vals, desc="Simulating temperatures"):
        tval, mag, ener = simulation_at_T_cupy_vectorized(T, lattice_size, steps_MC, n_runs, J, B, periodic,
                                                          random_init)
        results.append((tval, mag, ener))
    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    # Save simulation data
    out_file = f"output/Ising2D_GPU_vectorized_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

    # Compute heat capacity (using finite differences)
    heat_capacity = np.gradient(energies, temps)

    # Plot Magnetization vs Temperature
    plt.figure()
    plt.plot(temps, mags, 'o-', label='Magnetization')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.title(f"2D Ising (GPU, Checkerboard) L={lattice_size}")
    plt.legend()
    plt.show()

    # Plot Energy vs Temperature
    plt.figure()
    plt.plot(temps, energies, 'o-', label='Energy')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy")
    plt.title("Energy vs T (GPU, Checkerboard)")
    plt.legend()
    plt.show()

    # Plot Heat Capacity vs Temperature
    plt.figure()
    plt.plot(temps, heat_capacity, 'o-', label='Heat Capacity')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Heat Capacity")
    plt.title("Heat Capacity vs T (GPU, Checkerboard)")
    plt.legend()
    plt.show()
