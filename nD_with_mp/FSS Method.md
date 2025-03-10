The rigorous way to find critical exponents is through finite‐size scaling (FSS). The idea is:

1. **Simulate Many Lattice Sizes:** \(L = 8, 16, 32, 64, $\dots\$)  
2. **Measure Magnetization Near $\(T_c\)$:** For each $\(L\)$, collect data $\(M_L(T)\)$ for a range of temperatures around the expected critical temperature.  
3. **Apply a Scaling Form:** Theoretical finite‐size scaling predicts that near $\(T_c\)$:
$
   \[
   M_L(T) \approx L^{-\beta/\nu}\,\tilde{M}\!\Bigl((T - T_c)\,L^{1/\nu}\Bigr),
   \]
$
   where \(\nu\) is another critical exponent (for the correlation length). \(\tilde{M}\) is a universal scaling function.

4. **Data Collapse:** You then attempt to “collapse” all your \(M_L(T)\) data onto a **single** universal curve by plotting

   \[
   \bigl[\,M_L(T)\,L^{\beta/\nu}\bigr] \quad\text{vs.}\quad \bigl[(T - T_c)\,L^{1/\nu}\bigr]
   \]

   and adjusting \(\beta\), \(\nu\), and \(T_c\) until the curves for all lattice sizes overlap as much as possible. This is often done via numerical methods (e.g., a search algorithm or systematic scanning).

5. **Extract the Exponents:**  
   - Once you achieve a good collapse, the values of \(\beta\), \(\nu\), and \(T_c\) used in the collapse are your **finite‐size scaling estimates** of the true critical exponents and critical temperature.  
   - As \(L\) grows, you get closer to the thermodynamic limit. Typically, your estimates converge to the known exact values (\(\beta=1/8\), \(\nu=1\) for 2D Ising).
