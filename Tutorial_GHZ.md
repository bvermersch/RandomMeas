# Illustration of importance sampling for GHZ states

The script evalates the purity of noisy GHZ states by performing importance sampling of the random unitaries from the ideal pure GHZ state as proposed in https://arxiv.org/pdf/2102.13524.pdf .
We also provide comparisons to the old protocol of uniform sampling of the random unitaries.

The goal is to estimate the purity of a large N qubit noisy GHZ state prepared in a NISQ device. 

-> We sample the 'important' unitaries from ideal GHZ state. This is done with the help of a metropolis algorithm that samples local unitaries from the probability distribution contructed from the pure GHZ state.

-> We perform randomized measurements with Nu such sampled unitaries and collect Nm readout measurements for each applied unitary.

-> The unbiased estimator is contructed from Eq.(3) and the purity is computed as a weighted average over the applied unitaries Eq.(15).
