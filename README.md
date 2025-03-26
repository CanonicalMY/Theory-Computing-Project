Welcome to the personal repository for the UoM 2nd Year Physics with Theo. Phys. Theory Computing Project on the Monte Carlo Simulation of Phase Transitions. This repository serves as our collaborative workspace, where Wenyi and I explore the use of Monte Carlo methods to simulate phase transitions in magnetic systems, with a particular focus on the Ising model.

Here you'll find our project code, documentation, and regular updates on our progress. I will be detailing our methodologies, challenges, and findings as the project evolves.

---

2025.3.3 updates:
*Uploads the n-d Ising Model.py with multi-proessing that run the code in parallel, it also contains the interactive user entrying cell to analyse M vs T for different dimensions, lattice size, Random initialization, Monte Carlo steps, runs per temperatures, coupling constant J, and the external field B.

2025.3.9 updates:
*Upload the n-d Ising Model_v2.py that can now plot E vs T, C vs T, M vs B thus the hysteresis loop. The additional cell on Finite-Size Analysis of Exponentials is now added. All with interactive user entrying.

2025.3.10 updates: 
*Upload the n-d Ising Model_v2.1.py that modifies the Finite-Size Analysis with the old exponential function fit to a more reasonable power law fit, ie the Finite‐Size Scaling (FSS).

2025.3.26 updates: 
*In v4.1, separates the FSS with others, and rewrite the code with CuPy, change the update method from random site update to checkerboard update, in order to utilise the high computational ability of my GPU in doing parallel calculations to save the time. Now the code can run lattice size 128 in 2d with MS steps 20000 and 5 n runs in less then 5 hours, while originally takes 2 days.
