# Logarithmic-encoding
This repository will contain the code developped during the project on Quantum Logarithmic Encoding at UMA.
## Reports
1. Initial long report (mainly theoretical) on Overleaf (visualization only): https://fr.overleaf.com/read/ymgzyfphgdzb
2. Final report (main theoretical results and tests) (visualization only): https://fr.overleaf.com/read/nvhgnmzjmzvm

# Packages used

## Standard packages
- numpy 1.24.4
- matplotlib 3.5.1
- pandas 1.3.5
- qiskit 0.40.0. This was because of its compatibility with qiskit_ibm_runtime 0.9.0, although we finally did not test the algorithms on a real quanum device.
- networkx 3.1
- scypy 1.10.1
- psfam 0.0.16


## Custom packages

- dense_ev 2023.5.22. This version was modified to avoid the usage of qiskit.optflow (deprecated package of Qiskit). This was with the objective of running the tests in a real quantum device in the future. This modified version is available at: https://github.com/LeDernier/dense-ev. The original package is available at: https://github.com/atlytle/dense-ev.
- graph_functions.py (located in the folder "code")
- expectation_functions (located in the folder "code")
