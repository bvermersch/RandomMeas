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


## Code to reconstruct the purity via Common Randomized Measurements (CRM) of artifically generated randomized measurement data of a N qubit system
## Purity estimator used in the present code: https://science.sciencemag.org/content/364/6437/260 and https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.052323
##           purity = Average(sum_{s,s'} 2^(Number of qubits)(-2)^(-Hamming_Distance[s,s']) ) ( - statistical bias )


import random
import numpy as np
from scipy import linalg
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *


## Parameters
N = 10 # Number of qubits to analyze
Nu = 10 # Number of random unitaries to be used
NM = 2000 # Number of projective measurements (shots) per random unitary
mode = 'CUE'
Partitions =  [list(range(Nsub)) for Nsub in range(1,N+1)]

### Step 1:: Create a quantum state

# The quantum state qstate is stored as numpy.array of type numpy.complex_

# qstate can be
# - a pure state |psi> represented by a numpy array of shape (2**N,)
# - a mixed state rho reprresented by a numpy array of shape (2**N, 2**N)

# An additional parameter p can be specified to admix the identity
#  - |psi><psi| ->  (1-p)*|psi><psi| + p*1/2**N or
#  - rho ->  (1-p)*rho + p*1/2**N
#  - sigma -> |psi><psi|

## A GHZ state
qstate = np.zeros(2**N,dtype=np.complex_)
qstate[0] = 1./np.sqrt(2)
qstate[-1] = 1./np.sqrt(2)
if (N>16):
    print('Please reduce N (or adapt the call to np.unpackbits)')
    
# Consider realizing a noisy version of the GHZ state experimentally (rho). Noise given by depolarization noise strength p
p = 0.15 


### A random mixed state
#import qutip
#qstate = qutip.rand_dm(2**N).full()
#p=0.1

## Optional: calculate exact purities with Qutip partial trace function

from  src.ObtainExactValues import obtainExactPurities as obtExPur
purities_exact = obtExPur(N,qstate, Partitions,p)
print('Exact Purities')
for i in range(len(purities_exact)):
    print('Partition', Partitions[i], ':', np.round(purities_exact[i],4))


### Step 2:: Perform randomized measurements

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Generate Random Unitaries
unitaries=np.zeros((Nu,N,2,2),dtype=np.complex_)
for iu in range(Nu):
    for i in range(N):
        unitaries[iu,i]=SingleQubitRotation(random_gen,mode)
print('Random unitaries generated')

### Simulate the randomized measurements for rho in the experiment
Meas_Data = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = ObtainOutcomeProbabilities(N, qstate, unitaries[iu] , p)
    Meas_Data[iu,:] = Sampling_Meas(prob,N,NM)
print('Measurement data generated')




### Step 3:: Reconstruct purities from measured bitstrings
N_part = len(Partitions)
X_rho = np.zeros((Nu,N_part))
Purity_uni = np.zeros(N_part)
TracedSystems =  [ [x for x in range(N) if x not in p ] for p in Partitions]
for iu in range(Nu):
    print('PostProcessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = get_prob(Meas_Data[iu,:],N)
    for i_part in range(N_part):
        prob_subsystem = reduce_prob(prob,N,TracedSystems[i_part])
        X_rho[iu,i_part] = unbias(get_X(prob_subsystem,len(Partitions[i_part])), len(Partitions[i_part]), NM)
     
### Purity estimated by uniform sampling
Purity_uni = np.mean(X_rho,0)

print("Measured Purities uniform")
for i_part in range(N_part):
    print('Partition ',Partitions[i_part], ":", np.round(Purity_uni[i_part],4), " Error: ", np.round(100*np.abs(Purity_uni[i_part] - purities_exact[i_part])/purities_exact[i_part],3), "%")
        
### Step 4: offline classical post-processing on theory state sigma 
### Simulate the randomized measurements classically for the theory state sigma using the same unitaries as in the experiment
X_sigma = np.zeros((Nu,N_part))
Purity_CRM = np.zeros(N_part)
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = ObtainOutcomeProbabilities(N, qstate, unitaries[iu] , 0) ## p is taken to be 0 
    prob = np.reshape(prob, [2]*N)
    for i_part in range(N_part):
        prob_subsystem = reduce_prob(prob,N,TracedSystems[i_part])
        X_sigma[iu,i_part] = get_X(prob_subsystem,len(Partitions[i_part]))
    

### theoretical purity values for sigma 
purities_exact_sigma = obtExPur(N,qstate, Partitions,0)

### CRM purities:
Purity_CRM = (np.sum(X_rho,0) - np.sum(X_sigma,0) + Nu*np.array(purities_exact_sigma))/Nu
print("Measured Purities CRM")
for i_part in range(N_part):
    print('Partition ',Partitions[i_part], ":", np.round(Purity_CRM[i_part],4), " Error: ", np.round(100*np.abs(Purity_CRM[i_part] - purities_exact[i_part])/purities_exact[i_part],3), "%")