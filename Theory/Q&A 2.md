生成这些不同L的数据的目的是通过grid search找到critical parameter，对于这个目的，我该如何设定每个L的sweep次数呢？他们应该统一吗（尽管可能在大L不会达到equilibrium），还是说对于每个不同的L设定不同的sweep次数呢？这个次数又该如何决定呢？不会只能试出来吧？有没有对应的可以提供简单预测的理论呢？

---

When performing finite-size scaling (FSS) to extract critical parameters via grid search, the key is to have all system sizes reach a comparable degree of equilibration. In Monte Carlo simulations, the number of sweeps needed to reach equilibrium increases with the system size \( L \). In fact, the equilibration (or correlation) time \( \tau \) typically scales as

\[
\tau \sim L^z,
\]

where \( z \) is the dynamic critical exponent. For the 2D Ising model with local updates (like Metropolis or Glauber dynamics), \( z \) is typically around 2.0–2.2 (for instance, about 2.16).

---

## How to Set the Number of Sweeps for Different \( L \)

1. **Unified Sweeps vs. \( L \)-dependent Sweeps:**

   - **Unified Sweeps:**  
     If you use the same number of sweeps for all system sizes, smaller systems (e.g., \( L = 64 \)) might be well equilibrated, while larger systems (e.g., \( L = 256 \)) may remain far from equilibrium. This mismatch can lead to systematic errors when comparing quantities across different \( L \).

   - **\( L \)-dependent Sweeps:**  
     To ensure a fair comparison, you should increase the number of Monte Carlo sweeps with \( L \) according to the scaling law:

     \[
     \text{steps\_MC}(L) \sim A \times L^z,
     \]

     where \( A \) is a constant that you can determine from preliminary tests or from literature.

2. **Example Calculation:**

   - Suppose for \( L = 64 \) you use \( N_{MC} \) sweeps. Then for \( L = 128 \), you should use roughly:

     \[
     N_{MC}(128) \approx N_{MC}(64) \times \left( \frac{128}{64} \right)^z.
     \]

     If \( z \approx 2 \), that’s about 4 times as many sweeps. For \( L = 256 \), it would be about 16 times as many sweeps as for \( L = 64 \).

3. **Determining the Constant \( A \):**

   - **Trial and Error Guided by Theory:**  
     The exponent \( z \) gives you the scaling but not the absolute number of sweeps. You can start with a reasonable number for a small system (like 5000 sweeps for \( L = 64 \)) and then scale up according to \( L^z \).

   - **Autocorrelation Analysis:**  
     A more systematic method is to measure the integrated autocorrelation time \( \tau_{\text{int}} \) for an observable (such as magnetization). Then, ensure that the simulation length is significantly larger (e.g., 50–100 times \( \tau_{\text{int}} \)) for each system size.

---

## Summary and Recommendation

- **Do Not Use the Same Number of Sweeps for All \( L \):**  
  Use a sweep count that increases with \( L \), typically following a power law \( L^z \) (with \( z \approx 2 \) for the 2D Ising model with local updates). This way, even though larger systems require more computational time, they reach an equivalent level of equilibration as the smaller ones.

- **Practical Approach:**  
  For example, if you use 5000 sweeps for \( L = 64 \), then for \( L = 128 \) you might try around 20,000 sweeps, and for \( L = 256 \) about 80,000 sweeps. You may need to fine-tune these numbers by checking if the measured observables (like magnetization) have reached a steady state (i.e., have stabilized with little drift).

- **Theoretical Guidance:**  
  The scaling \( \tau \sim L^z \) with \( z \approx 2 \) (or 2.2) provides a theoretical basis to predict how the equilibration time should change with system size. This is much more systematic than trying random numbers, even though some testing is usually needed to confirm that equilibrium is reached in your simulations.

Using this approach, you ensure that each system size is simulated for a comparable “relative time” (relative to its correlation time), which is crucial for a reliable finite-size scaling analysis and subsequent grid search for the critical parameters.
