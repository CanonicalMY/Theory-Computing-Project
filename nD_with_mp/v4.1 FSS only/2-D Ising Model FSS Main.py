import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys

# Global parameters
default_J = 1.0            # Coupling constant; for J > 0, spins tend to align
default_B = 0.0            # External magnetic field
default_periodic = True    # The periodic boundary conditions


# ---------------------- Lattice Initialization ----------------------
# Initialize spin configuration based on dimension and lattice size.
# For 2D/3D, lattice_size forms a square/cubic lattice.
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
    """
    Compute the total energy of a 1D spin chain.
    """
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
    """
    Attempt to flip the spin at index i in the 1D chain using the Metropolis criterion.
    """
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
    """
    Perform n Monte Carlo steps for a 1D spin chain.
    In each MC step, attempt to flip N randomly chosen spins.
    """
    N = Spin.size
    for _ in range(n):
        for _ in range(N):
            idx = np.random.randint(N)
            Spin = flip_1D(Spin, idx, T, J, B, periodic)
    return Spin

def M_1D(Spin):
    """
    Calculate the average magnetization of the 1D spin chain.
    """
    return np.sum(Spin) / Spin.size

# ---------------------- 2D Model Functions ----------------------
# Initialize spin configuration based on dimension and lattice size.
# For 2D/3D, lattice_size forms a square/cubic lattice.
def local_energy_2D(Spin, i, j, J, B, periodic):
    """
    Calculate the local energy contribution for the spin at position (i, j) in a 2D lattice.
    Only nearest neighbors and the external field are considered.
    """
    N = Spin.shape[0]
    s = Spin[i, j]
    E_local = 0
    i_up    = (i - 1) % N if periodic else max(i - 1, 0)
    i_down  = (i + 1) % N if periodic else min(i + 1, N - 1)
    j_left  = (j - 1) % N if periodic else max(j - 1, 0)
    j_right = (j + 1) % N if periodic else min(j + 1, N - 1)
    E_local -= J * s * Spin[i_up, j]
    E_local -= J * s * Spin[i_down, j]
    E_local -= J * s * Spin[i, j_left]
    E_local -= J * s * Spin[i, j_right]
    E_local -= B * s
    return E_local

def flip_2D(Spin, i, j, T, J, B, periodic):
    """
    Attempt to flip the spin at position (i, j) in the 2D lattice using the Metropolis criterion.
    """
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
    """
    Perform n Monte Carlo steps for a 2D lattice.
    In each MC step, attempt to flip N*N randomly chosen spins.
    """
    N = Spin.shape[0]
    for _ in range(n):
        for _ in range(N * N):
            i_rand = np.random.randint(0, N)
            j_rand = np.random.randint(0, N)
            Spin = flip_2D(Spin, i_rand, j_rand, T, J, B, periodic)
    return Spin

def M_2D(Spin):
    """
    Calculate the average magnetization of the 2D lattice.
    """
    return np.sum(Spin) / Spin.size

# ---------------------- 3D Model Functions ----------------------
def local_energy_3D(Spin, x, y, z, J, B, periodic):
    """
    Calculate the local energy contribution for the spin at position (x, y, z) in a 3D lattice.
    Considers interactions with the 6 nearest neighbors and the external field.
    """
    N = Spin.shape[0]
    s = Spin[x, y, z]
    E_local = 0
    x_left  = (x - 1) % N if periodic else max(x - 1, 0)
    x_right = (x + 1) % N if periodic else min(x + 1, N - 1)
    y_down  = (y - 1) % N if periodic else max(y - 1, 0)
    y_up    = (y + 1) % N if periodic else min(y + 1, N - 1)
    z_back  = (z - 1) % N if periodic else max(z - 1, 0)
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
    """
    Attempt to flip the spin at position (x, y, z) in the 3D lattice using the Metropolis criterion.
    """
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
    """
    Perform n Monte Carlo steps for a 3D lattice.
    In each MC step, attempt to flip N^3 randomly chosen spins.
    """
    N = Spin.shape[0]
    for _ in range(n):
        for _ in range(N ** 3):
            x_rand = np.random.randint(0, N)
            y_rand = np.random.randint(0, N)
            z_rand = np.random.randint(0, N)
            Spin = flip_3D(Spin, x_rand, y_rand, z_rand, T, J, B, periodic)
    return Spin

def M_3D(Spin):
    """
    Calculate the average magnetization of the 3D lattice.
    """
    return np.sum(Spin) / Spin.size


# ---------------------- Generic Simulation Function ----------------------
def simulation_at_T_generic(T_value, dimension, lattice_size, steps_MC, n_runs, J, B, periodic, random_init):
    """
    Run the simulation at a given temperature T_value.
    The simulation is repeated n_runs times for averaging.
    random_init controls whether spins start randomly or all +1.
    """
    M_sum = 0.0
    for _ in range(n_runs):
        if dimension == 1:
            Spin = initial_config(random_init, lattice_size, 1)
            Spin = step_1D(Spin, steps_MC, T_value, J, B, periodic)
            M_sum += M_1D(Spin)
        elif dimension == 2:
            Spin = initial_config(random_init, lattice_size, 2)
            Spin = step_2D(Spin, steps_MC, T_value, J, B, periodic)
            M_sum += M_2D(Spin)
        elif dimension == 3:
            Spin = initial_config(random_init, lattice_size, 3)
            Spin = step_3D(Spin, steps_MC, T_value, J, B, periodic)
            M_sum += M_3D(Spin)
    return (T_value, M_sum / n_runs)

# ---------------------- User Entry ----------------------
if __name__ == '__main__':
    # mp.freeze_support()  # Necessary on Windows; optional on macOS

    # User input parameters
    try:
        dimension = int(input("Please enter the model dimension (1, 2, or 3): "))
    except:
        print("Invalid input.")
        sys.exit(1)
    if dimension not in [1, 2, 3]:
        print("Invalid input.")
        sys.exit(1)

    try:
        random_init_input = int(input("Random initialization? (1 for True, else for False): "))
        random_init = (random_init_input == 1)  # If input 1 then random_init is True.
    except:
        print("Invalid input.")
        sys.exit(1)

    try:
        lattice_size = int(input("Please enter the lattice size: "))
    except:
        print("Invalid input.")
        sys.exit(1)

    try:
        steps_MC = int(input("Please enter the number of Monte Carlo steps (eg., 100): "))
    except:
        print("Invalid input.")
        sys.exit(1)

    try:
        n_runs = int(input("Please enter the number of runs per temperature (e.g., 100): "))
    except:
        print("Invalid input.")
        sys.exit(1)

    try:
        T_min = float(input("Please enter the minimum temperature (e.g., 0): "))
        T_max = float(input("Please enter the maximum temperature (e.g., 5): "))
        num_T = int(input("Please enter the number of temperature points (e.g., 25): "))
    except:
        print("Invalid input.")
        sys.exit(1)

    try:
        J = float(input("Please enter the coupling constant J (default 1): "))
    except:
        print("Invalid input.")
        sys.exit(1)
    try:
        B = float(input("Please enter the external magnetic field B (default 0): "))
    except:
        print("Invalid input.")
        sys.exit(1)

    # Generate temperature array:
    T_vals = np.linspace(T_min, T_max, num_T)

    # Use functools.partial to bind the simulation parameters.
    # Then we only pass the temperature value when mapping over the temperature array.
    # This ensures every worker process gets the correct configuration.
    sim_func = partial(simulation_at_T_generic,
                       dimension=dimension,
                       lattice_size=lattice_size,
                       steps_MC=steps_MC,
                       n_runs=n_runs,
                       J=J,
                       B=B,
                       periodic=default_periodic,
                       random_init=random_init)

    # Parallel computation of average magnetization for each temperature
    with mp.Pool() as pool:
        results = list(tqdm(pool.imap(sim_func, T_vals), total=len(T_vals)))

    # Process results
    temps = np.array([r[0] for r in results])
    mags = np.array([r[1] for r in results])

    # Plot and save the results
    plt.scatter(temps, mags)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Magnetization (M)")
    plt.title(f"{dimension}D Ising Model (Lattice size = {lattice_size})")
    filename = f"Ising_{dimension}D_with_mp.jpg"
    plt.savefig(filename)
    print(f"Figure saved as {filename}")
    plt.show()

# Run in Terminal:
# python '/Users/thomas/Chty/Softwares/Pycharm Project/year2sem2/Computing/nD_with_mp/n-D Ising Model.py'
# kill $(pgrep -f python) to kill any running Python processes.
