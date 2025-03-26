Below is the answer in English, formatted in Markdown:

---

# How to Set Monte Carlo Sweeps for Different Lattice Sizes in FSS

When performing finite-size scaling (FSS) studies—especially for grid search to extract critical parameters—the equilibrium time (or the number of Monte Carlo sweeps needed) increases with the lattice size \( L \). Here are some key points and suggestions:

---

## 1. Theoretical Background: Dynamic Exponent \( z \)

- **Equilibration Time Scaling:**  
  In Monte Carlo simulations of the Ising model, the equilibration (or correlation) time typically scales as:
  \[
  \tau \sim L^z
  \]
  where \( z \) is the dynamic exponent.

- **Typical Values for 2D Ising:**  
  For local update algorithms (e.g., Metropolis or Glauber dynamics) for the 2D Ising model, the dynamic exponent \( z \) is often estimated to be around 2.0–2.2 (some studies give \( z \approx 2.16 \)).

---

## 2. Should You Use the Same Number of Sweeps for All \( L \)?

- **Uniform Sweep Count:**  
  Using the same number of Monte Carlo sweeps for all \( L \) might cause:
  - **Small \( L \):** Oversimulation (the system fully equilibrates and is over-sampled).
  - **Large \( L \):** Insufficient equilibration (the system remains out of equilibrium).

- **Different Sweep Counts:**  
  It is **better** to set the number of MC sweeps as a function of \( L \) to ensure that each system reaches a similar relative equilibrium. This means using:
  \[
  \text{steps\_MC}(L) \sim A \times L^z
  \]
  where \( A \) is a constant chosen from preliminary tests or literature values.

---

## 3. How to Decide on the Number of Sweeps?

- **Using the Dynamic Exponent:**  
  If you set \( \text{steps\_MC}(L) \sim L^z \) with \( z \approx 2 \), then:
  - For example, if you use 5,000 sweeps for \( L = 64 \), then for \( L = 128 \):
    \[
    \text{steps\_MC}(128) \approx 5000 \times \left(\frac{128}{64}\right)^2 = 5000 \times 4 = 20000 \text{ sweeps.}
    \]
  - Similarly, for \( L = 256 \):
    \[
    \text{steps\_MC}(256) \approx 5000 \times \left(\frac{256}{64}\right)^2 = 5000 \times 16 = 80000 \text{ sweeps.}
    \]

- **Practical Testing:**  
  In practice, you might need to **test** and check if the physical quantities (magnetization, energy, etc.) have converged. One way is to:
  - Monitor the time series of these quantities.
  - Calculate the autocorrelation time \(\tau_{\mathrm{int}}\) and ensure that your total simulation time is much larger than \(\tau_{\mathrm{int}}\).

- **Warm-Up Phase:**  
  Often, a separate "warm-up" (or "thermalization") period is used where the data are not recorded. This ensures that the system reaches equilibrium before measurements begin.

---

## 4. Is It Just Trial and Error?

- **Yes and No:**  
  There is some trial and error in practice. However, **the dynamic scaling relation provides a theoretical guideline**. By using:
  \[
  \text{steps\_MC}(L) \sim L^z,
  \]
  you can make a reasonable initial guess. Then, by monitoring convergence (or using autocorrelation analysis), you can fine-tune the constant \( A \).

- **Autocorrelation Analysis:**  
  A more rigorous method involves measuring the integrated autocorrelation time \(\tau_{\mathrm{int}}\) for your observables. Your simulation should run for a time much longer than \(\tau_{\mathrm{int}}\) to obtain independent measurements.

---

## 5. Summary & Recommendations

1. **Different \( L \) Should Use Different Sweep Numbers:**  
   Use \( \text{steps\_MC}(L) \sim A \times L^z \) rather than a uniform number of sweeps.

2. **Determine \( A \) and \( z \) from Preliminary Tests or Literature:**  
   For 2D Ising, \( z \) is typically around 2.0–2.2. Choose \( A \) so that smaller systems are well equilibrated.

3. **Incorporate a Warm-Up Phase:**  
   Run a set number of sweeps before collecting data to ensure equilibrium.

4. **Monitor Autocorrelation Times:**  
   This helps you determine if the simulation has run long enough for reliable statistics.

By applying these ideas, you can ensure that in your grid search for critical parameters (like \( T_c \), \(\beta\), \(\nu\)), each lattice size is sufficiently equilibrated and compared on an equal footing.

---

This approach is more scientifically sound than simply trying one number for all \( L \) and will provide more reliable estimates of the critical parameters in your FSS analysis.

Feel free to copy and adjust this explanation as needed!
