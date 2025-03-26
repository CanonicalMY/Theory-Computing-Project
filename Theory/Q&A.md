**At random_init = True, MC_steps = 1000, n_runs = 100, T from 1.8 to 2.8 in 50 points, for L = 64，128，256的三组数据，为什么随着L增加 phase transition反而不明显(not sharp)了呢？**

---

From the plots, it looks like **for larger lattice sizes (L = 128, 256), the magnetization at low temperatures is no longer close to 1.0**, making the phase transition appear “less sharp.” In principle, **for the 2D Ising model**, we expect that *larger* L yields a *sharper* transition near T_c ≈ 2.27. The fact that you see the opposite (lower magnetization for T < T_c as L grows) typically indicates **incomplete equilibration** of the larger systems at low temperatures.

Below are common reasons why this happens, plus suggestions on how to fix it:

---

## 1. Insufficient Monte Carlo Sweeps for Large L

- **Longer Correlation Times:**  
  As the lattice size increases, the system can take far more sweeps to reach equilibrium, especially below T_c. Large domains of flipped spins need time to form (or to merge) so that the system fully magnetizes.

- **Evidence in Your Plots:**  
  For L = 256 at T = 1.8, you only see M ≈ 0.5. In true equilibrium, 2D Ising should have a near-fully magnetized state (M ≈ 1) at T = 1.8. The partial magnetization is a hallmark of not having enough sweeps to fully order.

**What to do:**

1. **Increase** `steps_MC` significantly for larger L. For a 256×256 system at T < 2, you might need 5,000–50,000 sweeps or more to get close to equilibrium.
2. **Add a warm-up phase** (e.g., `warm_up_steps`) that does not contribute to measurements.

---

## 2. Random Initialization at Each Temperature

- If you run the simulation independently at each T with `random_init=True`, you start from a 50/50 random state. For large L, it can take many sweeps to reach the fully ordered state below T_c.

- **More Efficient Approach:**
  1. Start at a lower temperature (or an all-up state) and do a long warm-up.
  2. Then measure magnetization.
  3. Move to the next temperature by *re-using the final spin configuration* from the previous temperature (ascending or descending). This “T-sweep” approach can drastically reduce equilibration times at each step.

---

## 3. Checkerboard Update Synchronization

- While the vectorized checkerboard update is fast, **it can sometimes slow down domain growth** in very large systems if the acceptance of flips is not high enough. Usually it still works well, but you do need enough sweeps.

- **If domain walls are large**, the system can remain partially disordered for many sweeps. A random-site update (asynchronous) or additional optimization might sometimes equilibrate faster.

---

## 4. Practical Tips to Get a Sharper Transition

1. **Increase `steps_MC` for Larger L**  
   - Try 5–10 times more sweeps for L = 256 than for L = 64.

2. **Use a “Warm-Up”**  
   - For example, run `warm_up_steps=5000` at each temperature before measuring.  
   - Or run a separate warm-up at a low temperature to get a nearly ordered state, then measure.

3. **Sweep Temperatures in Sequence**  
   - Start from T < T_c with a fully ordered or random config, do a long run.  
   - Then *increase* T step by step, each time re-using the final spin from the previous T.  
     This way, you don’t have to re-equilibrate from scratch each time.

4. **Check Partial Magnetization Over Time**  
   - Print out the magnetization every few hundred sweeps for the largest L. If it’s still drifting after thousands of sweeps, you need more.

---

## Why a Larger System Should Show a Sharper Transition (In Equilibrium)

- In the thermodynamic limit, the magnetization at T < T_c is close to a finite value (the spontaneous magnetization), then it drops to 0 at T > T_c. For large L, the “jump” (in finite-size sense) becomes steeper near T_c.

- If you see a smaller magnetization at low T for bigger L, that almost always means you haven’t let the system fully order (i.e., it’s stuck in a partially disordered or metastable state).

---
---
---

**生成这些不同L的数据的目的是通过grid search找到critical parameter，对于这个目的，我该如何设定每个L的sweep次数呢？他们应该统一吗（尽管可能在大L不会达到equilibrium），还是说对于每个不同的L设定不同的sweep次数呢？这个次数又该如何决定呢？不会只能试出来吧？有没有对应的可以提供简单预测的理论呢？**

---

When performing finite-size scaling (FSS) to extract critical parameters via grid search, the key is to have all system sizes reach a comparable degree of equilibration. In Monte Carlo simulations, the number of sweeps needed to reach equilibrium increases with the system size L. In fact, the equilibration (or correlation) time τ typically scales as:

    τ ~ Lᶻ

where z is the dynamic critical exponent. For the 2D Ising model with local updates (like Metropolis or Glauber dynamics), z is typically around 2.0–2.2 (for instance, about 2.16).

---

## How to Set the Number of Sweeps for Different L

1. **Unified Sweeps vs. L-dependent Sweeps**

   - **Unified Sweeps:**  
     If you use the same number of sweeps for all system sizes, smaller systems (e.g., L = 64) might be well equilibrated, while larger systems (e.g., L = 256) may remain far from equilibrium. This mismatch can lead to systematic errors when comparing quantities across different L.

   - **L-dependent Sweeps:**  
     To ensure a fair comparison, you should increase the number of Monte Carlo sweeps with L according to the scaling law:

         steps_MC(L) ~ A × Lᶻ

     where A is a constant that you can determine from preliminary tests or from literature.

2. **Example Calculation**

   - Suppose for L = 64 you use N_MC sweeps. Then for L = 128, you should use roughly:

         N_MC(128) ≈ N_MC(64) × (128 / 64)ᶻ

     If z ≈ 2, that’s about 4 times as many sweeps. For L = 256, it would be about 16 times as many sweeps as for L = 64.

3. **Determining the Constant A**

   - **Trial and Error Guided by Theory:**  
     The exponent z gives you the scaling but not the absolute number of sweeps. You can start with a reasonable number for a small system (like 5000 sweeps for L = 64) and then scale up according to Lᶻ.

   - **Autocorrelation Analysis:**  
     A more systematic method is to measure the integrated autocorrelation time τ_int for an observable (such as magnetization). Then, ensure that the simulation length is significantly larger (e.g., 50–100 times τ_int) for each system size.

---

## Summary and Recommendation

- **Do Not Use the Same Number of Sweeps for All L:**  
  Use a sweep count that increases with L, typically following a power law Lᶻ (with z ≈ 2 for the 2D Ising model with local updates). This way, even though larger systems require more computational time, they reach an equivalent level of equilibration as the smaller ones.

- **Practical Approach:**  
  For example, if you use 5000 sweeps for L = 64, then for L = 128 you might try around 20,000 sweeps, and for L = 256 about 80,000 sweeps. You may need to fine-tune these numbers by checking if the measured observables (like magnetization) have reached a steady state (i.e., have stabilized with little drift).

- **Theoretical Guidance:**  
  The scaling τ ~ Lᶻ with z ≈ 2 (or 2.2) provides a theoretical basis to predict how the equilibration time should change with system size. This is much more systematic than trying random numbers, even though some testing is usually needed to confirm that equilibrium is reached in your simulations.

---

Using this approach, you ensure that each system size is simulated for a comparable “relative time” (relative to its correlation time), which is crucial for a reliable finite-size scaling analysis and subsequent grid search for the critical parameters.

