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
N = 3 # Number of qubits to analyze
Nu = 1000 # Number of random unitaries to be used
NM = 5000 # Number of projective measurements (shots) per random unitary
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
    
## depolarisation noise strength for the two devices
p1 = 0. ## noise in the first device
p2 = 0. ## noise in the second device

### Or a qutip random density matrix combined with a pure product state qubit
#rho = qutip.tensor(qutip.rand_dm(2**(N-1),pure=True),qutip.basis(2,0)*qutip.basis(2,0).dag()).data.todense()

## Optional: calculate exact purities with Qutip partial trace function
import qutip
psiQ = qutip.Qobj(psi,dims=[ [2]*N,[1]*N] )
rho1 = (1-p1)*psiQ*psiQ.dag()+p1/2**N
rho2 = (1-p2)*psiQ*psiQ.dag()+p2/2**N

print('Exact Fidelities')
for Partition in Partitions:
        rhop1 = rho1.ptrace(Partition)
        rhop2 = rho2.ptrace(Partition)
        overlap = (rhop1*rhop2).tr()
        denominator = max([(rhop1**2).tr(),(rhop2**2).tr()])
        fidelity_e = overlap#/denominator
        print('Partition ',Partition, ":", fidelity_e)

### Step 2:: Perform randomized measurements
psi = np.reshape(psi,[2]*N)
Meas_Data1 = np.zeros((Nu,NM),dtype='int64')
Meas_Data2 = np.zeros((Nu,NM),dtype='int64')
u = [0]*N
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    for i in range(N):
        u[i] = SingleQubitRotation(random_gen,mode)
    prob1 = Simulate_Meas_pseudopure(N, psi, p1, u)
    Meas_Data1[iu,:] = Sampling_Meas(prob1,N,NM)
    prob2 = Simulate_Meas_pseudopure(N, psi, p2, u)
    Meas_Data2[iu,:] = Sampling_Meas(prob2,N,NM)
    #Meas_Data[iu,:] = Simulate_Meas_mixed(N, rho, NM, u)
print('Measurement data generated \n')

### Step 3:: Reconstruct purities from measured bitstrings
N_part = len(Partitions)
X_overlap = np.zeros((Nu,N_part))
X_1 = np.zeros((Nu,N_part))
X_2 = np.zeros((Nu,N_part))
for iu in range(Nu):
    print('PostProcessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    probe1 = get_prob(Meas_Data1[iu,:],N)
    probe2 = get_prob(Meas_Data2[iu,:],N)
    for i_part in range(N_part):
        #print(Partitions[i_part])
        prob_subsystem1 = reduce_prob(probe1,N,TracedSystems[i_part])
        prob_subsystem2 = reduce_prob(probe2,N,TracedSystems[i_part])
        X_overlap[iu, i_part] = get_X_overlap(prob_subsystem1, prob_subsystem2, len(Partitions[i_part]))
        X_1[iu, i_part] = get_X(prob_subsystem1, len(Partitions[i_part]))
        X_2[iu, i_part] = get_X(prob_subsystem2, len(Partitions[i_part]))
        X_1[iu,i_part] = unbias(X_1[iu,i_part], len(Partitions[i_part]), NM)
        X_2[iu,i_part] = unbias(X_2[iu,i_part], len(Partitions[i_part]), NM)
RM_fidelity = np.mean(X_overlap,0)#/np.max([np.mean(X_1,0),np.mean(X_2,0)],0)
print('RM Fidelities')
for i_part in range(N_part):
    print('Partition ',Partitions[i_part], ":", RM_fidelity[i_part])
