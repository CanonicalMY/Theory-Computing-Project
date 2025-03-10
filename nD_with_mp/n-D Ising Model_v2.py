import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys
from scipy.optimize import curve_fit


# Global parameters
default_J = 1.0            # Coupling constant; for J > 0, spins tend to align
default_B = 1.0            # External magnetic field
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

def Energy_2D(Spin, J, B, periodic):
    """
    Compute the total energy for a 2D lattice.
    To avoid double counting, only sum over right and down neighbors.
    """
    N = Spin.shape[0]
    E_total = 0
    for i in range(N):
        for j in range(N):
            E_total -= J * Spin[i, j] * Spin[i, (j+1) % N]  # right neighbor (using periodicity)
            E_total -= J * Spin[i, j] * Spin[(i+1) % N, j]  # down neighbor
            E_total -= B * Spin[i, j]
    return E_total

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

def Energy_3D(Spin, J, B, periodic):
    """
    Compute the total energy for a 3D lattice.
    Sum over a subset of neighbors to avoid double counting (e.g., positive directions only).
    """
    N = Spin.shape[0]
    E_total = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                E_total -= J * Spin[x, y, z] * Spin[(x+1) % N, y, z]  # x+ direction
                E_total -= J * Spin[x, y, z] * Spin[x, (y+1) % N, z]  # y+ direction
                E_total -= J * Spin[x, y, z] * Spin[x, y, (z+1) % N]  # z+ direction
                E_total -= B * Spin[x, y, z]
    return E_total


# ---------------------- Generic Simulation Function ----------------------
def simulation_at_T_generic(T_value, dimension, lattice_size, steps_MC, n_runs, J, B, periodic, random_init):
    """
    Run the simulation at a given temperature T_value.
    The simulation is repeated n_runs times for averaging.
    random_init controls whether spins start randomly or all +1.
    # Now also compute the average energy over the runs.
    """
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
        lattice_size = int(input("Please enter the lattice size: "))
    except:
        print("Invalid input.")
        sys.exit(1)

    try:
        random_init_input = int(input("Random initialization? (1 for True, else for False): "))
        random_init = (random_init_input == 1)  # If input 1 then random_init is True.
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

    # Process results: now each result is (T, magnetization, energy)
    temps = np.array([r[0] for r in results])
    mags = np.array([r[1] for r in results])
    energies = np.array([r[2] for r in results])

    # Estimate heat capacity C = dE/dT (using finite differences)
    heat_capacity = np.gradient(energies, temps)

    # Plot Average Magnetization vs Temperature
    plt.figure()
    plt.scatter(temps, mags)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Magnetization (M)")
    plt.title(f"{dimension}D Ising Model (Lattice size = {lattice_size})")
    plt.savefig(f"Ising_{dimension}D_M_vs_T.jpg")
    plt.show()

    # Plot Average Energy vs Temperature
    plt.figure()
    plt.scatter(temps, energies)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Energy (E)")
    plt.title("Energy vs Temperature")
    plt.savefig(f"Ising_{dimension}D_E_vs_T.jpg")
    plt.show()

    # Plot Heat Capacity vs Temperature
    plt.figure()
    plt.scatter(temps, heat_capacity)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Heat Capacity (C)")
    plt.title("Heat Capacity vs Temperature")
    plt.savefig(f"Ising_{dimension}D_C_vs_T.jpg")
    plt.show()

    # Magnetization vs External Field B, Hysteresis Loop (Optional Run)
    # Below the Tc, the system is in a magnetically ordered (ferromagnetic) phase. 
    # As sweeping B from negative to positive and back, the magnetization lags behind, 
    # producing a large, characteristic hysteresis loop.
    do_hysteresis = input(
        "Do you want to run the M vs B graph? (Enter 1 for yes, any other number to skip this part): ")
    if do_hysteresis.strip() == "1":
        try:
            T_fixed = float(input("Enter the fixed temperature (below T_c) (e.g., 1.0): "))
            B_max = float(input("Enter the maximum absolute value of the external field B (e.g., 2.0): "))
            num_B = int(input("Enter the number of B points for the sweep (e.g., 50): "))
        except Exception as e:
            print("Invalid input for hysteresis parameters.", e)
            sys.exit(1)

        # Create arrays for upward and downward sweeps
        B_vals_up = np.linspace(-B_max, B_max, num_B)
        B_vals_down = np.linspace(B_max, -B_max, num_B)

        # Initialize spin configuration based on dimension
        if dimension == 1:
            Spin = initial_config(random_init, lattice_size, 1)
        elif dimension == 2:
            Spin = initial_config(random_init, lattice_size, 2)
        elif dimension == 3:
            Spin = initial_config(random_init, lattice_size, 3)

        # Upward sweep (increasing B)
        magnetizations_up = []
        for B_val in B_vals_up:
            if dimension == 1:
                Spin = step_1D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_up.append(M_1D(Spin))
            elif dimension == 2:
                Spin = step_2D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_up.append(M_2D(Spin))
            elif dimension == 3:
                Spin = step_3D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_up.append(M_3D(Spin))

        # Downward sweep (decreasing B)
        magnetizations_down = []
        for B_val in B_vals_down:
            if dimension == 1:
                Spin = step_1D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_down.append(M_1D(Spin))
            elif dimension == 2:
                Spin = step_2D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_down.append(M_2D(Spin))
            elif dimension == 3:
                Spin = step_3D(Spin, steps_MC, T_fixed, J, B_val, default_periodic)
                magnetizations_down.append(M_3D(Spin))

        # Plot the hysteresis loop: M vs B
        plt.figure()
        plt.plot(B_vals_up, magnetizations_up, 'o-', label="Increasing B")
        plt.plot(B_vals_down, magnetizations_down, 'o-', label="Decreasing B")
        plt.xlabel("External Field B")
        plt.ylabel("Magnetization M")
        plt.title(f"Hysteresis Loop at T = {T_fixed}")
        plt.legend()
        filename_hyst = f"Ising_{dimension}D_Hysteresis.jpg"
        plt.savefig(filename_hyst)
        plt.show()

    # ---------------------- Finite-Size Analysis: Exponential Decay of M vs T ----------------------
    do_finite_size = input(
        "Do you want to run finite-size analysis for M vs T exponential decay? (Enter 1 for yes, any other number to skip): ")
    if do_finite_size.strip() == "1":

        # Define the lattice sizes.
        lattice_sizes = [4, 6, 8, 10, 12]

        # Use the same T_min, T_max, and num_T from earlier.
        T_vals = np.linspace(T_min, T_max, num_T)

        # Container for fitted exponents for each lattice size.
        exponents = []  # each entry will be (lattice_size, k_fit)

        # Create a combined figure for all lattice sizes.
        plt.figure()
        # Define a list of colors.
        colors = ['b', 'g', 'r', 'c', 'm']


        # Define the exponential decay function: M(T) = A * exp(-k * T)
        def exp_decay(T, A, k):
            return A * np.exp(-k * T)


        for idx, L in enumerate(lattice_sizes):
            print(f"Running simulation for lattice size {L}...")
            # Create a local simulation function with lattice_size set to L.
            sim_func_local = partial(simulation_at_T_generic,
                                     dimension=dimension,
                                     lattice_size=L,
                                     steps_MC=steps_MC,
                                     n_runs=n_runs,
                                     J=J,
                                     B=B,
                                     periodic=default_periodic,
                                     random_init=random_init)

            # Run the simulation in parallel over the temperature range.
            with mp.Pool() as pool:
                results_L = list(tqdm(pool.imap(sim_func_local, T_vals), total=len(T_vals)))

            # Extract temperatures and magnetizations for lattice size L.
            temps_L = np.array([r[0] for r in results_L])
            mags_L = np.array([r[1] for r in results_L])

            # Plot the simulation data for this lattice size.
            plt.plot(temps_L, mags_L, 'o-', color=colors[idx], label=f"Lattice {L} (Sim)")

            # Choose a temperature range for fitting (e.g., up to T_max/2).
            T_fit_max = T_max / 2.0
            indices = np.where(temps_L <= T_fit_max)
            T_fit = temps_L[indices]
            M_fit = mags_L[indices]

            # Perform the curve fitting.
            try:
                popt, _ = curve_fit(exp_decay, T_fit, M_fit, p0=(M_fit[0], 1.0))
                A_fit, k_fit = popt
                exponents.append((L, k_fit))
                # Generate a smooth line for the fitted curve.
                T_fit_line = np.linspace(T_fit.min(), T_fit.max(), 100)
                plt.plot(T_fit_line, exp_decay(T_fit_line, *popt), '--', color=colors[idx], label=f"Lattice {L} (Fit)")
            except Exception as e:
                print(f"Curve fitting failed for lattice size {L} with error: {e}")
                exponents.append((L, None))

        plt.xlabel("Temperature (T)")
        plt.ylabel("Magnetization (M)")
        plt.title("Finite-Size Analysis: M vs T for Different Lattice Sizes")
        plt.legend()
        plt.savefig("FiniteSize_All_LatticeSizes.jpg")
        plt.show()

        # Plot the fitted exponent vs lattice size.
        valid_exponents = [(L, k) for L, k in exponents if k is not None]
        if valid_exponents:
            lattice_list = [pair[0] for pair in valid_exponents]
            k_list = [pair[1] for pair in valid_exponents]
            plt.figure()
            plt.plot(lattice_list, k_list, 'o-')
            plt.xlabel("Lattice Size")
            plt.ylabel("Fitted Exponent k")
            plt.title("Fitted Exponent vs Lattice Size")
            plt.savefig("Fitted_Exponent_vs_LatticeSize.jpg")
            plt.show()
            print("Fitted exponents:", valid_exponents)

# For a 4*4 random int 1000 MC and 500 runs from 0 to 4 in 200 with 1 and 1 JB, approx. time is 50:56
# For a 4*4 random int 10000 MC and 100 runs from 0 to 5 in 100 with 1 and 1 JB, approx. time is 47:23
