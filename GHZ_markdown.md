
# Illustration of importance sampling for GHZ states

The script evalates the purity ${[p_2]}_{\rm IS}$ of noisy GHZ states by performing importance sampling of the local random unitaries from the ideal pure GHZ state as proposed in https://arxiv.org/pdf/2102.13524.pdf .
We also provide performance comparisons to evalaute the purity ${[p_2]_{\rm uni}}$ using the old protocol of uniform sampling of the random unitaries.

The goal is to estimate the purity of a large $N$ qubit noisy GHZ state prepared in a NISQ device. This state is represented by an addition of global depolarisation noise of strength $p_{depo}$ to the pure GHZ state:
$\rho(p_{depo}) = (1-p_{depo})|\mathrm{GHZ}_N><\mathrm{GHZ}_N| + p_{depo}\,\mathbb{1}/2^N$.

-> We sample the 'important' unitaries from ideal GHZ state. This is done with the help of a metropolis algorithm that samples local unitaries from the probability distribution $p_\mathrm{IS}(u)$ contructed from the $N$ qubit pure GHZ state function $X_\mathrm{GHZ}(u)$.

-> We perform randomized measurements with $N_u$ such sampled unitaries and collect $N_M$ readout measurements for each applied unitary.

-> The unbiased estimator $X_e(u^{(r)})$ for each applied $u^{(r)}$ with $r = 1, \dots, N_u$, is contructed
<img src="https://render.githubusercontent.com/render/math?math=X_e(u^{(r)}) = \frac{2^N}{N_M(N_M-1)} \sum_{m \neq m'} (-2) ^{-D[s_m^{(r)},s^{(r)}_{m'}]}">
    
-> The purity ${[p_2]}_{IS}$ is computed as a weighted average over the applied unitaries  given by:

<img src="https://render.githubusercontent.com/render/math?math="[p_2]_\mathrm{IS} = \frac{ 1}{N_s} \sum_{r = 1}^{N_u} \frac{n^{(r)} X_e(u^{(r)}) }{p_\mathrm{IS}(u^{(r)})}">
   
   where $N_s$ is the total number of unitaries $u$ collected by the metropolis algorithm which contains $N_u$ distinct ones and $n^{(r)}$ takes into account the occurence of each of unitary $u^{(r)}$ of the $N_u$ distinct samples and satisfies $N_s=\sum_r n^{(r)}$.
   



```python

```
