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
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *
#from qutip import *


## Parameters
N = 7 # Number of qubits to analyze
Nu = 500 # Number of random unitaries to be used
NM = 500 # Number of projective measurements (shots) per random unitary
mode = 'CUE'
Partitions =  [list(range(Nsub)) for Nsub in range(1,N+1)]

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Step 1:: Create a quantum state

# The quantum state is represented by a variable qstate and is stored as numpy.array of type numpy.complex_ )
# It can be a pure state psi of dimension (2**N,) or a mixed state rho of dimension (2**N,2**N).
#
# An additional parameter p can be specified to admix the identity
# |psi><psi| ->  (1-p)*|psi><psi| + p*1/2**N or rho ->  (1-p)*rho + p*1/2**N

## A GHZ state
qstate = np.zeros(2**N,dtype=np.complex_)
qstate[0] = 1./np.sqrt(2)
qstate[-1] = 1./np.sqrt(2)
if (N>16):
    print('Please reduce N (or adapt the call to np.unpackbits)')
p = 0.

### A random mixed state
#import qutip
#qstate = qutip.rand_dm(2**N).full()
#p=0.



## Optional: calculate exact purities with Qutip partial trace function
from  src.ObtainExactValues import obtainExactPurities as obtExPur
purities = obtExPur(N,qstate, Partitions,p)
print('Exact Purities')
for i in range(len(purities)):
    print('Partition', Partitions[i], ':', np.round(purities[i],3))

### Step 2:: Perform randomized measurements
Meas_Data = np.zeros((Nu,NM),dtype='int64')
u = [0]*N
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    for i in range(N):
        u[i] = SingleQubitRotation(random_gen,mode)
    prob = ObtainOutcomeProbabilities(N, qstate, u , p)
    Meas_Data[iu,:] = Sampling_Meas(prob,N,NM)
print('Measurement data generated')

### Step 3:: Reconstruct purities from measured bitstrings
N_part = len(Partitions)
X = np.zeros((Nu,N_part))
Purity = np.zeros(N_part)
TracedSystems =  [ [x for x in range(N) if x not in p ] for p in Partitions]
for iu in range(Nu):
    print('PostProcessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = get_prob(Meas_Data[iu,:],N)
    for i_part in range(N_part):
        prob_subsystem = reduce_prob(prob,N,TracedSystems[i_part])
        X[iu,i_part] = unbias(get_X(prob_subsystem,len(Partitions[i_part])), len(Partitions[i_part]), NM)
Purity = np.mean(X,0)
#Purity = unbias(np.mean(X,0),N,NM)
for i_part in range(N_part):
    print('Partition ',Partitions[i_part], ":", Purity[i_part], '2nd Renyi Entropy : ',-np.log2(Purity[i_part]))
