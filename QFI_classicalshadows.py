#   Copyright 2021 Benoit Vermersch, Andreas Elben, Aniket Rath
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


import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *
from src.PreprocessingImportanceSampling import *

### This script estimates the lower bounds of the Quantum fisher information of a noisy GHZ realized in the experiment using classical shadows


## Parameters
N = 5 # Number of qubits to analyze
d = 2**N
Nu = 10000 # Number of random unitaries to be used
NM = 1 # Number of projective measurements (shots) per random unitary
mode = 'CUE'


### Step 1:: Create a quantum state

# The quantum state qstate is stored as numpy.array of type numpy.complex_

# qstate can be
# - a pure state |psi> represented by a numpy array of shape (2**N,)
# - a mixed state rho reprresented by a numpy array of shape (2**N, 2**N)

# An additional parameter p can be specified to admix the identity
#  - |psi><psi| ->  (1-p)*|psi><psi| + p*1/2**N or
#  - rho ->  (1-p)*rho + p*1/2**N

## An ideal GHZ state
qstate = np.zeros(2**N,dtype=np.complex_)
qstate[0] = 1./np.sqrt(2)
qstate[-1] = 1./np.sqrt(2)

### A random mixed state
#import qutip
#qstate = qutip.rand_dm(2**N).full()
#p_depo = 0.1

# Consider realizing a noisy version of the GHZ state experimentally. Noise given by depolarization noise strength p_depo
p_depo = 0

### Defining the collective spin operator along a given direction specified by spin:
def operator(spin):
    col_spin = 0*rand_dm(d).full()
    for ii in range(N):
        col_spin += tensor([qeye(2**ii),spin,qeye(2**(N-ii-1))]).full()
        
    return col_spin

## Hermitian operator 'A' wrt which the lower bounds are defined:
A = 0.5*Qobj(operator(sigmaz()), dims = [[2]*N, [2]*N])

## Theoretical estimations of the QFI and its lower bounds for noisy GHZ states
qfi = ((((1-p_depo)**2)*2**(N-1))/(p_depo*(1-2**(N-1))+ 2**(N-1)))*N**2
F0 = (1-p_depo)**2*N**2
F1 = (1-p_depo)**2*(1+ p_depo - p_depo/(2**(N - 1)))*N**2


### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Perform Randomized measurements 
print('Randomized measurements with Nu = '+str(Nu)+' and NM = '+str(NM) +' on a '+str(N)+' qubit state')

### Generate Random Unitaries
unitaries=np.zeros((Nu,N,2,2),dtype=np.complex_)
for iu in range(Nu):
    for i in range(N):
        unitaries[iu,i,:,:]=SingleQubitRotation(random_gen,mode)
print('Random unitaries generated and stored')

### Simulate the randomized measurements
Meas_Data = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = ObtainOutcomeProbabilities(N, qstate, unitaries[iu] , p_depo)
    Meas_Data[iu,:] = Sampling_Meas(prob,N,NM)
print('Measurement data generated and stored \n')


### classical post-processing to reconstruct classical shadows

rho1s = np.zeros([d,d], dtype = complex)
rho2s = np.zeros([d,d], dtype = complex)
rho2As = np.zeros([d,d], dtype = complex)
rho3s = np.zeros([d,d], dtype = complex)
rho3As = np.zeros([d,d], dtype = complex)


def get_shadow(Meas, Unitary,N):
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    bit_string_count = np.bincount(Meas, minlength = 2**N)
    bit_string_ordered = np.arange(0,2**N,1)
    
    rho_shadows = np.zeros([2**N,2**N], dtype = complex)
    for inm in range(2**N):
        rhoj = 1
        s_r = get_bin(bit_string_ordered[inm],N)
        for j in range(N):
            sj = int(s_r[j],2)
            if sj == 0:
                proj = np.array([[1,0],[0,0]])
            else:
                proj = np.array([[0,0],[0,1]])
            
            rhoj = np.kron(rhoj,3*np.einsum('ab,bc,cd', np.transpose(np.conjugate(Unitary[j,:,:])),proj,Unitary[j,:,:]) - np.array([[1,0],[0,1]]))
        
        rho_shadows += bit_string_count[inm]*rhoj/len(Meas)#np.kron([r for r in rhoj])#(rhoj)/len(Meas)
    return rho_shadows

### computing the lower bound by constructing classical shadows
for iu in range(Nu):

    rho1 = 0*np.zeros([d,d])
    rho2 = 0*np.zeros([d,d])
    rho3 = 0*np.zeros([d,d])
    
    rho1 = get_shadow(Meas_Data[iu,:], unitaries[iu,:,:,:], N)
    rho2 = np.einsum('ab,bc',rho1, rho1)
    rho3 = np.einsum('ab,bc,cd',rho1, rho1, rho1)
    
    A_rho = np.einsum('ab,bc',A,rho1)
    
    rho1s += rho1
    rho2s += rho2
    rho3s += rho3
    
    rho2As += np.einsum('ab,bc,cd',rho1,A_rho,A)
    rho3As += np.einsum('ab,bc,cd',rho2,A_rho,A)
    print('Progress bar {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    

A2 = (A**2).full()


### Unbiasing the estimates of the lower bounds using U-statistics:
F0e_1 = (np.einsum('ab,bc,ca', rho1s,rho1s,A2) - np.einsum('ab,ba', rho2s, A2))/(Nu*(Nu-1))
F0e_2 = (np.einsum('ab,bc,cd,da',rho1s, A,rho1s,A) - np.einsum('aa', rho2As))/(Nu*(Nu-1))

F0e = np.real(4*(F0e_1 - F0e_2))

F1e_1 = (np.einsum('ab,bc,cd,da', rho1s, rho1s,rho1s,A2) - 3*np.einsum('ab,bc,ca', rho1s,rho2s, A2) + 2*np.einsum('ab,ba', rho3s, A2))/(Nu*(Nu-1)*(Nu-2))
F1e_2 = (np.einsum('ab,bc,cd,de,ea',rho1s,rho1s,A,rho1s,A) - 3*np.einsum('ab,bc,cd,da', rho2s, A,rho1s,A) + 2*np.einsum('aa', rho3As))/(Nu*(Nu-1)*(Nu-2))
F1e = np.real(2*F0e - 4*(F1e_1 - F1e_2))

print('True value of F0 : ', F0)
print('F0 estimated using classical shadows: ', F0e)
print('True value of F1 : ', F1)
print('F1 estimated using classical shadows: ', F1e)




