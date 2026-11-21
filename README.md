# Quantum gate optimization with PINNQOC, Krotov, and PRONTO

This repository accompanies [*Performance of Krotov, PRONTO and PINN for optimal control of quantum gates*](https://arxiv.org/abs/2607.23276) by M. D. Jimenez, M. D. Forlevesi, E. F. de Lima, and L. K. Castelano.

The code benchmarks a physics-informed neural network (PINNQOC) against Krotov and PRONTO for designing control pulses that implement the X, Hadamard, and S gates while limiting population leakage out of the computational subspace. The benchmarks use a three-level fluxonium model and a four-level nitrogen-vacancy center coupled to a carbon-13 nuclear spin. See the paper for the Hamiltonians, optimization settings, and results.

## Setup (Linux)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

It also requires Pytorch (we used version: 2.10.0). We recommend to install it following the specific instructions for your operative system and if you use or not CUDA ([Install Pytorch](https://pytorch.org/get-started/locally/)).