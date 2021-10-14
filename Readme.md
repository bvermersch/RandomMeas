# RandomMeas: Python Interface to Random Measurements

This repository offer scripts to reconstruct the purity and cross-platform fidelities from randomized measurements.
<img src="Pics/RandomMeasurements.png" alt="drawing" width="200"/>

## Purity from randomized measurement
The purity is reconstructed from statistical correlations between randomized measurements, which are obtained via random single qubit gates

+ [Original Paper](https://science.sciencemag.org/content/364/6437)
+ [Python Script](PurityRM.py)
+ Typical use: Up to 10 qubits

## Fidelities from randomized measurement
The fidelity between quantum states realized on two different quantum devices is obtained by cross-correlating randomized measurements.

+ [Original Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.010504)
+ [Python Script](FidelityRM.py)
+ Typical use: Up to 10 qubits

## Purity from importance sampling of randomized measurement
The purity is obtained via importance sampling of randomized measurements, obtained by sampling random single qubit unitaries  with respect to the targeted quantum state

+ [Original Paper](https://arxiv.org/abs/2102.13524)
+ [Tutorial](GHZ_markdown.ipynb)
+ [Python Script](FidelityRM.py)
+ Typical use: Up to 24 qubits


License: Apache 2.0

Second version: Oct 2021
