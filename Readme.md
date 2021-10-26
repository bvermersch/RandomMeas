# RandomMeas: Python Interface to random measurements

We provide scripts to reconstruct the purity and cross-platform fidelities from randomized measurements.
<img src="Pics/RandomMeasurements.png" alt="drawing" width="250"/>

<<<<<<< HEAD
=======
## Purity from randomized measurements
The purity is reconstructed from statistical correlations between randomized measurements, which are obtained via random single qubit gates

+ [Original Paper](https://science.sciencemag.org/content/364/6437)
+ [Python Script](PurityRM.py)
+ Typical use: Up to 10 qubits

## Fidelities from randomized measurements
The fidelity between quantum states realized on two different quantum devices is obtained by cross-correlating randomized measurements.

+ [Original Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.010504)
+ [Python Script](FidelityRM.py)
+ Typical use: Up to 10 qubits

## Purity from importance sampling of randomized measurements
The purity is obtained with exponentially less measurements compared to the standard approach of uniform sampling. This is based on importance sampling of random single qubit unitaries, with respect to an approximation of the quantum state.

+ [Original Paper](https://arxiv.org/abs/2102.13524)
+ [Tutorial](TutorialImportanceSampling.ipynb)
+ [Python Script](PurityImportanceSampling.py)
+ Typical use: Up to 25 qubits

## Fidelity from importance sampling of randomized measurements
The Fidelity between two noisy versions of the same state realized on two different devices is obtained with exponentially less measurements compared to the standard approach of uniform sampling. This is again based on importance sampling of random single qubit unitaries, with respect to the approximation of the ideal  quantum state, and same sampled unitaries are applied to both the devices.

+ [Original Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.010504)
+ [Python Script](FidelityImportanceSampling.py)
+ Typical use: Up to 25 qubits

## Topological entanglement entropy from importance sampling of randomized measurements
The topological entanglement entropy "S_topo" is extracted using standard approach of uniform sampling and the new method of importance sampling of random single qubit unitaries, from the modelled approximation of the quantum state.

+ [Original Paper](https://arxiv.org/pdf/2104.01180.pdf)
+ [Python Script](PurityImportanceSamplingToricCode.py)
+ Typical use: Up to 15 qubits

>>>>>>> dev_AE

License: Apache 2.0

Second version: Oct 2021
