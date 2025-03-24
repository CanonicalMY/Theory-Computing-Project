"""
CuPy support cell for FSS using checkerboard updates
=================
Naive GPU-accelerated 2D Ising simulation using CuPy.
Saves results to .npz files for later FSS analysis.
"""

import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------- Lattice Initialization ----------------------
def initial_config_cupy(random_init, lattice_size):
    """
    Create a 2D lattice on the GPU.
    If random_init is True, spins are assigned randomly to ±1.
    Otherwise, all spins are set to +1.
    """
    if random_init:
        # Generate random integers {0,1} and map to {-1, +1}
        spin = cp.random.randint(2, size=(lattice_size, lattice_size))
        spin = 2 * spin - 1
    else:
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.int8)
    return spin

def checkerboard_masks(lattice_size):
    """
    Return two boolean masks (as CuPy arrays) for even and odd sites.
    Even sites: (i+j) mod 2 == 0, Odd sites: (i+j) mod 2 == 1.
    """
    # Create index arrays for the lattice
    indices = cp.indices((lattice_size, lattice_size))
    i = indices[0]
    j = indices[1]
    mask_even = ((i + j) % 2) == 0
    mask_odd = ((i + j) % 2) == 1
    return mask_even, mask_odd

# ---------------------- Checkerboard Update ----------------------
def step_2D_cupy_checkerboard(spin, steps_MC, T, J, B, periodic):
    """
    Perform Monte Carlo sweeps using checkerboard updates.
    For each sweep, update all even sites concurrently, then update all odd sites.
    """
    N = spin.shape[0]
    mask_even, mask_odd = checkerboard_masks(N)

    for _ in range(steps_MC):
        # --- Even Update ---
        # Compute neighbor sum using vectorized operations with periodic boundaries.
        neighbors = (cp.roll(spin, 1, axis=0) + cp.roll(spin, -1, axis=0) +
                     cp.roll(spin, 1, axis=1) + cp.roll(spin, -1, axis=1))
        # Energy change if spin is flipped: dE = 2 * s * (J * neighbors + B)
        dE = 2 * spin * (J * neighbors + B)
        # For even sites only:
        dE_even = dE[mask_even]
        # Generate random numbers for even sites
        rand_even = cp.random.rand(*spin.shape)[mask_even]
        # Accept flip if dE <= 0 or if rand < exp(-dE/T)
        accept_even = (dE_even <= 0) | (rand_even < cp.exp(-dE_even / T))
        # Flip spins on even sites where accepted
        even_vals = spin[mask_even]
        even_vals[accept_even] = -even_vals[accept_even]
        spin[mask_even] = even_vals

        # --- Odd Update ---
        # Recompute neighbors after updating even sites
        neighbors = (cp.roll(spin, 1, axis=0) + cp.roll(spin, -1, axis=0) +
                     cp.roll(spin, 1, axis=1) + cp.roll(spin, -1, axis=1))
        dE = 2 * spin * (J * neighbors + B)
        dE_odd = dE[mask_odd]
        rand_odd = cp.random.rand(*spin.shape)[mask_odd]
        accept_odd = (dE_odd <= 0) | (rand_odd < cp.exp(-dE_odd / T))
        odd_vals = spin[mask_odd]
        odd_vals[accept_odd] = -odd_vals[accept_odd]
        spin[mask_odd] = odd_vals

    return spin

# ---------------------- Observables ----------------------
def M_2D_cupy(spin):
    """
    Compute the absolute magnetization per spin.
    """
    return cp.abs(spin).sum() / spin.size

def energy_2D_cupy(spin, J, B, periodic):
    """
    Compute the total energy using vectorized operations.
    Note: Each bond is counted twice, so we divide the sum by 2.
    """
    neighbors = (cp.roll(spin, 1, axis=0) + cp.roll(spin, -1, axis=0) +
                 cp.roll(spin, 1, axis=1) + cp.roll(spin, -1, axis=1))
    E = -J * spin * neighbors - B * spin
    return cp.sum(E) / 2

# ---------------------- Simulation Function ----------------------
def simulation_at_T_cupy(T_value, lattice_size, steps_MC, n_runs, J, B,
                         periodic=True, random_init=True):
    """
    Run the 2D Ising simulation at a given temperature T_value on the GPU.
    Repeats n_runs times to average the observables.
    """
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        spin = initial_config_cupy(random_init, lattice_size)
        spin = step_2D_cupy_checkerboard(spin, steps_MC, T_value, J, B, periodic)
        m = M_2D_cupy(spin)
        e = energy_2D_cupy(spin, J, B, periodic)
        # Bring observables back to CPU and accumulate
        M_sum += float(m.get())
        E_sum += float(e.get())
    return T_value, M_sum / n_runs, E_sum / n_runs

# ---------------------- Main Routine ----------------------
if __name__ == "__main__":
    # Simulation parameters (modify as needed)
    lattice_size = 64         # Lattice size (e.g., 64x64)
    random_init = True        # Use random initialization (change to False for ordered)
    steps_MC = 1000           # Number of Monte Carlo sweeps per temperature
    n_runs = 50               # Number of independent runs per temperature
    T_min = 1.5               # Minimum temperature
    T_max = 3.5               # Maximum temperature
    num_T = 50                # Number of temperature points
    J = 1.0
    B = 0.0
    periodic = True

    # Prepare temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    # Run simulation over temperature points
    results = []
    print(f"Running 2D Ising with checkerboard updates on GPU (L={lattice_size})...")
    for T in tqdm(T_vals):
        results.append(simulation_at_T_cupy(T, lattice_size, steps_MC, n_runs, J, B, periodic, random_init))
    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    # Save simulation data to file
    out_file = f"Ising2D_GPU_checkerboard_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

    # Quick plotting
    plt.figure()
    plt.plot(temps, mags, 'o-', label='Magnetization')
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.title(f"2D Ising (GPU Checkerboard) L={lattice_size}")
    plt.legend()
    plt.show()
