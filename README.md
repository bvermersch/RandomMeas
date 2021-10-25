# Illustration of importance sampling for GHZ states

The script evalates the purity <img src="https://render.githubusercontent.com/render/math?math={[p_2]}_{IS}"> of noisy GHZ states by performing importance sampling of the local random unitaries from the ideal pure GHZ state as proposed in https://arxiv.org/pdf/2102.13524.pdf .
We also provide comparisons to the old protocol of uniform sampling of the random unitaries.

The goal is to estimate the purity <img src="https://render.githubusercontent.com/render/math?math=p_{2}"> of a large N qubit noisy GHZ state prepared in a NISQ device. 

-> We sample the 'important' unitaries from ideal GHZ state. This is done with the help of a metropolis algorithm that samples local unitaries from the probability distribution <img src="https://render.githubusercontent.com/render/math?math=p_\mathrm{IS}(u)"> contructed from the pure GHZ state function <img src="https://render.githubusercontent.com/render/math?math=X_\mathrm{GHZ}(u)">.

-> We perform randomized measurements with <img src="https://render.githubusercontent.com/render/math?math=N_u"> such sampled unitaries and collect <img src="https://render.githubusercontent.com/render/math?math=N_M"> readout measurements for each applied unitary.

-> The unbiased estimator for each is contructed from Eq.(3): <img src="https://render.githubusercontent.com/render/math?math=X_e(u^{(r)}) = \frac{2^N}{N_M(N_M-1)}
    \sum_{m \neq m'} (-2) ^{-D[s_m^{(r)},s^{(r)}_{m'}]}"> 
    
-> and the purity is computed as a weighted average over the applied unitaries Eq.(15).

The script evalates the purity ${[p_2]}_{IS}$ of noisy GHZ states by performing importance sampling of the local random unitaries from the ideal pure GHZ state as proposed in https://arxiv.org/pdf/2102.13524.pdf .
We also provide comparisons to the old protocol of uniform sampling of the random unitaries.

The goal is to estimate the purity $p_{2}$ of a large $N$ qubit noisy GHZ state prepared in a NISQ device. 

-> We sample the 'important' unitaries from ideal GHZ state. This is done with the help of a metropolis algorithm that samples local unitaries from the probability distribution $p_\mathrm{IS}(u)$ contructed from the pure GHZ state function $X_\mathrm{GHZ}(u)$.

-> We perform randomized measurements with $N_u$ such sampled unitaries and collect $N_M$ readout measurements for each applied unitary.

-> The unbiased estimator for each applied $u^{(r)}$ with $r = 1, \dots, N_u$, is contructed from Eq.(3): $X_e(u^{(r)}) = \frac{2^N}{N_M(N_M-1)}
    \sum_{m \neq m'} (-2) ^{-D[s_m^{(r)},s^{(r)}_{m'}]}$ 
    
-> The purity ${[p_2]}_{IS}$ is computed as a weighted average over the applied unitaries Eq.(15) given by:
   $[p_2]_\mathrm{IS} = \frac{ 1}{N_s}
    \sum_{r = 1}^{N_u} \frac{n^{(r)} X_e(u^{(r)}) }{p_\mathrm{IS}(u^{(r)})}$
   
   where $N_s$ is the total number of samples of unitaries $u$ collected by the metropolis algorithm in which we collect $N_u$ distinct ones

