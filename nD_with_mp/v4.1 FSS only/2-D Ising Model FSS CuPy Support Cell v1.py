"""
CuPy support cell for FSS
=================
Naive GPU-accelerated 2D Ising simulation using CuPy.
Saves results to .npz files for later FSS analysis.
"""

import cupy as cp
import numpy as np
import os
from tqdm import tqdm

# ---------------------- 2D Ising GPU Functions ----------------------
def initial_config_cupy(random_init, lattice_size):
    """
    Allocate a 2D spin lattice on the GPU using CuPy arrays.
    If random_init=True, spins are +1/-1 at random;
    otherwise, all spins = +1.
    """
    if random_init:
        # Generate random 0 or 1, map to +1 or -1
        spin = cp.random.randint(2, size=(lattice_size, lattice_size))
        spin = 2*spin - 1  # map {0,1} -> {-1,+1}
    else:
        spin = cp.ones((lattice_size, lattice_size), dtype=cp.int8)
    return spin


def local_energy_2D_cupy(spin, i, j, J, B, periodic):
    """
    Compute the local energy contribution of spin[i,j] with
    its 4 neighbors + external field B.
    spin is a CuPy array on the GPU.
    """
    N = spin.shape[0]
    s = spin[i, j]

    # Periodic boundary indices
    i_up    = (i - 1) % N if periodic else max(i - 1, 0)
    i_down  = (i + 1) % N if periodic else min(i + 1, N - 1)
    j_left  = (j - 1) % N if periodic else max(j - 1, 0)
    j_right = (j + 1) % N if periodic else min(j + 1, N - 1)

    # Local energy (neighbors + external field)
    # We'll do these as Python float ops on GPU scalars
    E_local = - J * s * (spin[i_up, j] + spin[i_down, j]
                         + spin[i, j_left] + spin[i, j_right])
    E_local -= B * s
    return E_local


def flip_2D_cupy(spin, i, j, T, J, B, periodic):
    """
    Attempt a Metropolis spin flip at spin[i,j].
    spin is on the GPU. Return None; spin is updated in-place if accepted.
    """
    E_before = local_energy_2D_cupy(spin, i, j, J, B, periodic)
    old_val = spin[i, j]
    spin[i, j] = -old_val
    E_after = local_energy_2D_cupy(spin, i, j, J, B, periodic)

    dE = E_after - E_before
    # If dE>0, accept with probability exp(-dE/T)
    if dE > 0:
        if T != 0:
            # We do random check on GPU or CPU
            r = cp.random.rand()
            if r >= cp.exp(-dE / T):
                # Reject flip
                spin[i, j] = old_val


def step_2D_cupy(spin, steps_MC, T, J, B, periodic):
    """
    Perform steps_MC Monte Carlo sweeps on the GPU for a 2D lattice.
    In each sweep, we attempt N*N random spin flips.
    """
    N = spin.shape[0]
    for _ in range(steps_MC):
        # For each sweep, pick N*N random sites
        idx_i = cp.random.randint(0, N, size=(N*N,))
        idx_j = cp.random.randint(0, N, size=(N*N,))
        for k in range(N*N):
            i_rand = idx_i[k]
            j_rand = idx_j[k]
            flip_2D_cupy(spin, i_rand, j_rand, T, J, B, periodic)


def M_2D_cupy(spin):
    """
    Compute absolute magnetization = |sum of spins| / (N*N).
    spin is a CuPy array on GPU.
    """
    return cp.abs(spin).sum() / spin.size


def energy_2D_cupy(spin, J, B, periodic):
    """
    Compute total energy for a 2D lattice using a simple sum
    over pairs to avoid double counting. (Naive approach.)
    """
    N = spin.shape[0]
    E_total = cp.float32(0.0)
    for i in range(N):
        for j in range(N):
            # Right neighbor
            E_total -= J * spin[i, j] * spin[i, (j+1) % N]
            # Down neighbor
            E_total -= J * spin[i, j] * spin[(i+1) % N, j]
            # External field
            E_total -= B * spin[i, j]
    return E_total


def simulation_at_T_cupy(T_value, lattice_size, steps_MC, n_runs, J, B,
                         periodic=True, random_init=True):
    """
    Run the simulation at temperature T_value on the GPU.
    Repeated n_runs times for averaging.
    """
    M_sum = 0.0
    E_sum = 0.0
    for _ in range(n_runs):
        spin = initial_config_cupy(random_init, lattice_size)
        step_2D_cupy(spin, steps_MC, T_value, J, B, periodic)
        m = M_2D_cupy(spin)
        e = energy_2D_cupy(spin, J, B, periodic)
        # Copy back to CPU for summation
        M_sum += float(m.get())
        E_sum += float(e.get())

    M_avg = M_sum / n_runs
    E_avg = E_sum / n_runs
    return T_value, M_avg, E_avg


# ---------------------- Main: GPU Simulation & Save ----------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dimension = 2
    lattice_size = 8
    random_init = True
    steps_MC = 100
    n_runs = 10
    T_min = 1.8
    T_max = 2.8
    num_T = 50
    J = 1.0
    B = 0.0
    periodic = True

    # Prepare temperature array
    T_vals = np.linspace(T_min, T_max, num_T)

    # Run simulation across temperatures
    results = []
    print(f"Running 2D Ising on GPU (CuPy) with L={lattice_size}, steps_MC={steps_MC}, n_runs={n_runs} ...")
    for T in tqdm(T_vals):
        tval, mag, ener = simulation_at_T_cupy(T, lattice_size, steps_MC, n_runs, J, B,
                                              periodic, random_init)
        results.append((tval, mag, ener))

    results = np.array(results)
    temps = results[:, 0]
    mags = results[:, 1]
    energies = results[:, 2]

    # Save data
    out_file = f"output/Ising2D_GPU_L{lattice_size}.npz"
    np.savez(out_file, temps=temps, mags=mags, energies=energies)
    print(f"Simulation data saved to {out_file}")

    # Quick plot
    heat_capacity = np.gradient(energies, temps)

    plt.figure()
    plt.plot(temps, mags, 'o-', label='Mag')
    plt.xlabel("T")
    plt.ylabel("Magnetization")
    plt.title(f"2D Ising (GPU) L={lattice_size}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(temps, energies, 'o-', label='Energy')
    plt.xlabel("T")
    plt.ylabel("Energy")
    plt.title("Energy vs T")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(temps, heat_capacity, 'o-', label='Heat Capacity')
    plt.xlabel("T")
    plt.ylabel("C")
    plt.title("Heat Capacity vs T")
    plt.legend()
    plt.show()
