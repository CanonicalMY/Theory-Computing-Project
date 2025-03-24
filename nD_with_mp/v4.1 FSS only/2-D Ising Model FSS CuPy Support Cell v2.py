"""
CuPy support cell for FSS with vectorized energy calculation and Checkerboard updates
=====================================
GPU-accelerated 2D Ising simulation using CuPy with vectorized (checkerboard) updates.
Saves results to .npz files for later FSS analysis.
"""

import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# ---------------------- 2D Ising GPU Functions (Vectorized) ----------------------

def initial_config_cupy(random_init, lattice_size):
    """
    Allocate a 2D spin lattice on the GPU using CuPy arrays.
    If random_init=True, spins are +1/-1 at random; otherwise, all spins = +1.
    """
    if random_init:
        spin = cp.random.randint(2, size=(lattice_size, lattice_size))
        spin = 2 * spin - 1  # Map {0,1} -> {-1,+1}
    else:
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.int8)
    return spin

def energy_2D_cupy_vectorized(spin, J, B, periodic=True):
    """
    Compute the total energy for a 2D lattice using vectorized operations.
    Uses cp.roll to shift the lattice and sum contributions from right and down neighbors.
    """
    if periodic:
        right = cp.roll(spin, shift=-1, axis=1)
        down = cp.roll(spin, shift=-1, axis=0)
    else:
        # Non-periodic version (not optimized)
        right = cp.zeros_like(spin)
        right[:, :-1] = spin[:, 1:]
        down = cp.zeros_like(spin)
        down[:-1, :] = spin[1:, :]
    energy = -J * cp.sum(spin * (right + down)) - B * cp.sum(spin)
    return energy

def checkerboard_update(spin, T, J, B, periodic, parity):
    """
    Perform a vectorized update on one sublattice (checkerboard update).
    'parity' is 0 for black sites and 1 for white sites.
    """
    N = spin.shape[0]
    # Create a mask for checkerboard sites where (i+j)%2 == parity
    i_indices = cp.arange(N).reshape(N, 1)
    j_indices = cp.arange(N).reshape(1, N)
    mask = ((i_indices + j_indices) % 2) == parity

    # Compute neighbor sum using periodic boundaries via cp.roll
    neighbor_sum = (cp.roll(spin, shift=1, axis=0) + cp.roll(spin, shift=-1, axis=0) +
                    cp.roll(spin, shift=1, axis=1) + cp.roll(spin, shift=-1, axis=1))
    # Energy difference if the spin is flipped: ΔE = 2 * spin * (J * neighbor_sum + B)
    deltaE = 2 * spin * (J * neighbor_sum + B)
    # Generate a random matrix for all sites
    random_matrix = cp.random.rand(N, N)
    # Acceptance: flip if ΔE <= 0 or with probability exp(-ΔE/T)
    acceptance = (deltaE <= 0) | (random_matrix < cp.exp(-deltaE / T))
    # Only update sites in the checkerboard mask
    flip_mask = mask & acceptance
    spin[flip_mask] = -spin[flip_mask]

def step_2D_cupy_vectorized(spin, steps_MC, T, J, B, periodic=True):
    """
    Perform 'steps_MC' Monte Carlo sweeps using vectorized checkerboard updates.
    Each sweep updates the black sublattice and then the white sublattice.
    """
    for _ in range(steps_MC):
        checkerboard_update(spin, T, J, B, periodic, parity=0)
        checkerboard_update(spin, T, J, B, periodic, parity=1)

def M_2D_cupy(spin):
    """
    Compute absolute magnetization = |sum of spins| / (N*N).
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
        # Use the vectorized Monte Carlo sweeps (checkerboard update)
        step_2D_cupy_vectorized(spin, steps_MC, T_value, J, B, periodic)
        m = M_2D_cupy(spin)
        e = energy_2D_cupy_vectorized(spin, J, B, periodic)
        # Bring results back to CPU
        M_sum += float(m.get())
        E_sum += float(e.get())
    M_avg = M_sum / n_runs
    E_avg = E_sum / n_runs
    return T_value, M_avg, E_avg

# ---------------------- Main: GPU Simulation & Save ----------------------
if __name__ == "__main__":
    # Simulation parameters (for testing)
    lattice_size = 8
    random_init = True
    steps_MC = 100        # You can increase this for better equilibration
    n_runs = 10
    T_min = 1.8
    T_max = 2.8
    num_T = 50
    J = 1.0
    B = 0.0
    periodic = True

    # Prepare temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    results = []
    print(f"Running 2D Ising on GPU (vectorized) with L={lattice_size}, steps_MC={steps_MC}, n_runs={n_runs} ...")
    for T in tqdm(T_vals):
        res = simulation_at_T_cupy_vectorized(T, lattice_size, steps_MC, n_runs, J, B, periodic, random_init)
        results.append(res)

    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    out_file = f"output/Ising2D_GPU_vectorized_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

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
