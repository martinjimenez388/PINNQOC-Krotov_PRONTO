# Quantum gate optimization with PINNQOC, Krotov, and PRONTO

This repository accompanies [*Performance of Krotov, PRONTO and PINN for optimal control of quantum gates*](https://arxiv.org/abs/2607.23276) by M. D. Jimenez, M. D. Forlevesi, E. F. de Lima, and L. K. Castelano.

The code benchmarks a physics-informed neural network (PINNQOC) approach against Krotov and PRONTO for designing control pulses that implement the X, Hadamard, and S gates while limiting population leakage out of the computational subspace. The benchmarks use a three-level fluxonium model and a four-level nitrogen-vacancy center coupled to a carbon-13 nuclear spin. See the paper for the Hamiltonians, optimization settings, and results.

## Installation (Linux)

We recommend to create a virtual environment to install the required dependencies on it. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

It also requires Pytorch (we used version: 2.10.0 for CPU). Install it following the instructions from the official webpage ([Install Pytorch](https://pytorch.org/get-started/locally/)).

## Description of the repository content

The jupyter notebooks in the root directory and the Python scripts in the PRONTO subdirectory contain the code to perform the optimization of control fields for the afore mentioned systems using the approaches described in the paper. They follow the same pipeline:

* Define input parameters:
    - Time grid where the evolution will be performed.
    - Target quantum gate.
    - Set of initial states: we took $|0\rangle$ and $|1\rangle$.
    - Guess fields (either loaded from file or defined in the same code).
    - Leakage penalization factor.
    - Maximum number of iterations of the optimization method.
    - Tolerance for the functional to minimize.
    - For PINNQOC, define the network architechture.
    - Parameters of the optimization method: learning rate, loss weights, etc.
* Perform the evolution under the guess fields. For PINNQOC, also train the neural network to fit both the guess fields and the state dynamics under them (pre-training).
* Run the optimization: apply a gradient-descent based method or a Newton-like approach to iteratively search for the minimum of the required loss functional.
* Once a stopping criterion is reached, simulate state dynamics for a set of Haar random states (over the computational space) under the optimized fields and compute mean fidelity and leakage.
