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


# Code to reconstruct the quantum state fidelity between two quantum devices (or one quantum device and a classical simulator)
# Based on https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.010504

import random
import numpy as np
from scipy import linalg
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *
from qutip import *


## Parameters
N = 7 # Number of qubits to analyze
Nu = 1000 # Number of random unitaries to be used
NM = 5000 # Number of projective measurements (shots) per random unitary
mode = 'CUE'
Partitions =  [range(Nsub) for Nsub in range(1,N+1)]


### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)


### Step 1:: Create a quantum state

# The quantum state qstate is stored as numpy.array of type numpy.complex_

# qstate can be
# - a pure state |psi> represented by a numpy array of shape (2**N,)
# - a mixed state rho reprresented by a numpy array of shape (2**N, 2**N)

# An additional parameter p can be specified to admix the identity
#  - |psi><psi| ->  (1-p)*|psi><psi| + p*1/2**N or
#  - rho ->  (1-p)*rho + p*1/2**N

## A GHZ state
qstate = np.zeros(2**N,dtype=np.complex_)
qstate[0] = 1./np.sqrt(2)
qstate[-1] = 1./np.sqrt(2)
    
p_1 = 0.1 ## noisy state realized using depolarization noise in the first device  
p_2 = 0.2 ## noisy state realized using depolarization noise in the second device  


### A random mixed state
#import qutip
#qstate = qutip.rand_dm(2**N).full()
#p_1 = 0.1
#p_2 = 0.2

#print('Exact Fidelities \n')
#rho1 = Qobj((1-p_1)*qstate + p_1*1/2**N, [[2]*N, [2]*N])
#rho2 = Qobj((1-p_2)*qstate + p_2*1/2**N, [[2]*N, [2]*N])

#for Partition in Partitions:
#        rhop1 = rho1.ptrace(Partition)
#        rhop2 = rho2.ptrace(Partition)
#        overlap = (rhop1*rhop2).tr()
#        denominator = max([(rhop1**2).tr(),(rhop2**2).tr()])
#        fidelity_e = overlap/denominator
#        print('Partition ',Partition, ":", fidelity_e)
        

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

### Simulate the randomized measurements
Meas_Data1 = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings from the first device
Meas_Data2 = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings from the second device
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob1 = ObtainOutcomeProbabilities(N, qstate, unitaries[iu], p_1)
    Meas_Data1[iu,:] = Sampling_Meas(prob1,N,NM)
    prob2 = ObtainOutcomeProbabilities(N, qstate, unitaries[iu], p_2)
    Meas_Data2[iu,:] = Sampling_Meas(prob2,N,NM)
print('Measurement data generated')


### Step 3:: Reconstruct fidelities from the measured bitstrings
N_part = len(Partitions)
RM_fidelity = np.zeros(N_part)
TracedSystems =  [ [x for x in range(N) if x not in p ] for p in Partitions]
X_overlap = np.zeros((Nu,N_part))
X_1 = np.zeros((Nu,N_part))
X_2 = np.zeros((Nu,N_part))
for iu in range(Nu):
    print('PostProcessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    probe1 = get_prob(Meas_Data1[iu,:],N)
    probe2 = get_prob(Meas_Data2[iu,:],N)
    for i_part in range(N_part):
        prob_subsystem1 = reduce_prob(probe1,N,TracedSystems[i_part])
        prob_subsystem2 = reduce_prob(probe2,N,TracedSystems[i_part])
        X_overlap[iu, i_part] = get_X_overlap(prob_subsystem1, prob_subsystem2, len(Partitions[i_part]))
        X_1[iu, i_part] = get_X(prob_subsystem1, len(Partitions[i_part]))
        X_2[iu, i_part] = get_X(prob_subsystem2, len(Partitions[i_part]))
        X_1[iu,i_part] = unbias(X_1[iu,i_part], len(Partitions[i_part]), NM)
        X_2[iu,i_part] = unbias(X_2[iu,i_part], len(Partitions[i_part]), NM)
RM_fidelity = np.mean(X_overlap,0)/np.max([np.mean(X_1,0),np.mean(X_2,0)],0)

print("Measured Fidelities")
for i_part in range(N_part):
    print('Partition ',Partitions[i_part], ":", np.round(RM_fidelity[i_part],4))
