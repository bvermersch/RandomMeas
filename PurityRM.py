#   Copyright 2020 Benoit Vermersch, Andreas Elben, Aniket Rath
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

## Code to reconstruct the purity of artifically generated randomized measurement data of a N qubit system
## Protocol: Elben, Vermersch, et al: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.050406
##           Apply Nu random unitaries made of single qubit rotations, measure for each unitary Nm bitstrings k_{s=1,..,NM}
## Estimator used in the present code: https://science.sciencemag.org/content/364/6437/260 and https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.052323
##           purity = Average(sum_{s,s'} 2^(Number of qubits)(-2)^(-Hamming_Distance[s,s']) ) ( - statistical bias )

import random
import numpy as np
from scipy import linalg
import sys
sys.path.append("src")
from ObtainMeasurements import *
from AnalyzeMeasurements import *
#from qutip import *


## Parameters
N = 7 # Number of qubits to analyze
Nu = 500 # Number of random unitaries to be used
NM = 500 # Number of projective measurements (shots) per random unitary
mode = 'CUE'
Partitions =  [range(Nsub) for Nsub in range(1,N+1)]
TracedSystems =  [ [x for x in range(N) if x not in p ] for p in Partitions]

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Step 1:: Create a quantum state
## A GHZ state
psi = np.zeros(2**N)
psi[0] = 1./np.sqrt(2)
psi[-1] = 1./np.sqrt(2)
if (N>16):
    print('Please reduce N (or adapt the call to np.unpackbits)')
p = 0.

### Or a qutip random density matrix combined with a pure product state qubit
#rho = qutip.tensor(qutip.rand_dm(2**(N-1),pure=True),qutip.basis(2,0)*qutip.basis(2,0).dag()).data.todense()

## Optional: calculate exact purities with Qutip partial trace function
import qutip
psiQ = qutip.Qobj(psi,dims=[ [2]*N,[1]*N] )
rho = (1-p)*psiQ*psiQ.dag()+p/2**N
print('Exact Purities')
for Partition in Partitions:
        rhop = rho.ptrace(Partition)
        print('Partition ',Partition, ":", (rhop*rhop).tr())

### Step 2:: Perform randomized measurements
psi = np.reshape(psi,[2]*N)
Meas_Data = np.zeros((Nu,NM),dtype='int64')
u = [0]*N
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    for i in range(N):
        u[i] = SingleQubitRotation(random_gen,mode)
    prob = Simulate_Meas_pseudopure(N, psi, p, u)
    Meas_Data[iu,:] = Sampling_Meas(prob,N,NM)
    #Meas_Data[iu,:] = Simulate_Meas_mixed(N, rho, NM, u)
print('Measurement data generated')

### Step 3:: Reconstruct purities from measured bitstrings
N_part = len(Partitions)
X = np.zeros((Nu,N_part))
for iu in range(Nu):
    print('PostProcessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = get_prob(Meas_Data[iu,:],N)
    for i_part in range(N_part):
        prob_subsystem = reduce_prob(prob,N,TracedSystems[i_part])
        X[iu,i_part] = get_X(prob_subsystem,len(Partitions[i_part]),NM)
Purity = np.mean(X,0)
Purity = unbias(np.mean(X,0),N,NM)
for i_part in range(N_part):
    print('Partition ',Partitions[i_part], ":", Purity[i_part], '2nd Renyi Entropy : ',-np.log2(Purity[i_part]))
