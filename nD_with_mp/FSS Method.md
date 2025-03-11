The rigorous way to find critical exponents is through finite‐size scaling (FSS). The idea is:

1. **Simulate Many Lattice Sizes:** \(L = 8, 16, 32, 64, $\dots\$)  
2. **Measure Magnetization Near $\(T_c\)$:** For each $\(L\)$, collect data $\(M_L(T)\)$ for a range of temperatures around the expected critical temperature.  
3. **Apply a Scaling Form:** Theoretical finite‐size scaling predicts that near $\(T_c\)$:

$$\
M_L(T) \approx L^{-\beta/\nu}\ \tilde{M}\!\Bigl((T - T_c)\ L^{1/\nu}\Bigr)
\$$

   where $$\nu\$$ is another critical exponent (for the correlation length). $\tilde{M}$ is a universal scaling function.

4. **Data Collapse:** Then attempt to “collapse” all the $M_L(T)$ data onto a **single** universal curve by plotting

$$
\left[ M_L(T) L^{\beta/\nu} \right] \quad \text{vs.} \quad \left[ (T - T_c) L^{1/\nu} \right]
$$

   and adjusting $\beta$, $\nu$, and $T_c$ until the curves for all lattice sizes overlap as much as possible. This is often done via numerical methods (e.g., a search algorithm or systematic scanning).

5. **Extract the Exponents:**  
   - Once achieved a good collapse, the values of \(\beta\), \(\nu\), and \(T_c\) used in the collapse are **finite‐size scaling estimates** of the true critical exponents and critical temperature.  
   - As $L$ grows, we get closer to the thermodynamic limit. Typically, the estimates converge to the known exact values $\beta=1/8$, \($\nu=1$\) for 2D Ising).
  

**Note:**
The **cost function** is a quantitative measure used to assess how well the data from different lattice sizes collapse onto a single universal curve when scaled according to a set of trial parameters \((T_c, \beta, \nu)\). 

1. **Data Scaling**  
   For each lattice size \(L\) and temperature \(T\) with corresponding magnetization \(M\), we apply the following transformations:
   - $\(x = (T - T_c) \times L^{1/\nu}\)$
   - $\(y = M \times L^{\beta/\nu}\)$
   The idea is that if the chosen parameters \(T_c\), \(\beta\), and \(\nu\) are correct, all the scaled data points \((x, y)\) from different \(L\) values should collapse onto a single universal curve.

2. **Measuring the Spread (Local Variance)**  
   After scaling, we sort all the \(x\) values and then divide the \(x\) range into a number of bins (for example, 20 bins). For each bin, we compute the variance of the corresponding \(y\) values. This variance tells us how spread out the data is in that bin.

3. **Defining the Cost Function**  
   The cost function is defined as the average variance of the \(y\) values across all bins. If the data collapses very well (i.e., data points from different lattice sizes overlap tightly), the variances in each bin will be small, leading to a low cost. Conversely, if the collapse is poor, the variances will be larger, resulting in a high cost.

4. **Purpose**  
   By minimizing this cost function, we search for the best parameters \((T_c, \beta, \nu)\) that produce the optimal data collapse. These optimal parameters are our estimates for the true critical temperature and critical exponents derived from the finite-size scaling analysis.

In summary, the cost function is used to quantify how well the scaling transformation (based on a given \((T_c, \beta, \nu)\)) causes the data from different lattice sizes to overlap. A lower cost indicates a better collapse, and thus, better estimates for the critical parameters.
