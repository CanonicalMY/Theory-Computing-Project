From the plots, it looks like **for larger lattice sizes (L=128, 256), the magnetization at low temperatures is no longer close to 1.0**, making the phase transition appear “less sharp.” In principle, **for the 2D Ising model**, we expect that *larger* \( L \) yields a *sharper* transition near \( T_c \approx 2.27 \). The fact that you see the opposite (lower magnetization for \( T < T_c \) as \( L \) grows) typically indicates **incomplete equilibration** of the larger systems at low temperatures.

Below are common reasons why this happens, plus suggestions on how to fix it:

---

## 1. Insufficient Monte Carlo Sweeps for Large \( L \)

- **Longer Correlation Times:**  
  As the lattice size increases, the system can take far more sweeps to reach equilibrium, especially below \( T_c \). Large domains of flipped spins need time to form (or to merge) so that the system fully magnetizes.

- **Evidence in Your Plots:**  
  For \( L = 256 \) at \( T = 1.8 \), you only see \( M \approx 0.5 \). In true equilibrium, 2D Ising should have a near-fully magnetized state (\( M \approx 1 \)) at \( T = 1.8 \). The partial magnetization is a hallmark of not having enough sweeps to fully order.

**What to do:**

1. **Increase** `steps_MC` significantly for larger \( L \). For a 256×256 system at \( T < 2 \), you might need 5,000–50,000 sweeps or more to get close to equilibrium.  
2. **Add a warm-up phase** (e.g., `warm_up_steps`) that does not contribute to measurements.

---

## 2. Random Initialization at Each Temperature

- If you run the simulation independently at each \( T \) with `random_init=True`, you start from a 50/50 random state. For large \( L \), it can take many sweeps to reach the fully ordered state below \( T_c \).

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

1. **Increase** `steps_MC` **for Larger** \( L \)  
   - Try 5–10 times more sweeps for \( L = 256 \) than for \( L = 64 \).

2. **Use a “Warm-Up”**  
   - For example, run `warm_up_steps=5000` at each temperature before measuring.  
   - Or run a separate warm-up at a low temperature to get a nearly ordered state, then measure.

3. **Sweep Temperatures in Sequence**  
   - Start from \( T < T_c \) with a fully ordered or random config, do a long run.  
   - Then *increase* \( T \) step by step, each time re-using the final spin from the previous \( T \).  
     This way, you don’t have to re-equilibrate from scratch each time.

4. **Check Partial Magnetization Over Time**  
   - Print out the magnetization every few hundred sweeps for the largest \( L \). If it’s still drifting after thousands of sweeps, you need more.

---

## Why a Larger System Should Show a Sharper Transition (In Equilibrium)

- In the thermodynamic limit, the magnetization at \( T < T_c \) is close to a finite value (the spontaneous magnetization), then it drops to 0 at \( T > T_c \). For large \( L \), the “jump” (in finite-size sense) becomes steeper near \( T_c \).

- If you see a smaller magnetization at low \( T \) for bigger \( L \), that almost always means you haven’t let the system fully order (i.e., it’s stuck in a partially disordered or metastable state).
