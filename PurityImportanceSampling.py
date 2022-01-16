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

# Code to reconstruct the purity via Importance Sampling

import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *
from src.PreprocessingImportanceSampling import *

### This script estimates the purity of a noisy GHZ realized in the experiment using uniform sampling and importance sampling from an ideal pure GHZ state
### Capable of simulating noisy GHZ state till N = 25 qubits !!!
### Importance sampling provides best performances for Nu ~ O(N) and NM ~O(2^N) !!


## Parameters
N = 15 # Number of qubits to analyze
d = 2**N
Nu = 50 # Number of random unitaries to be used
NM = 2**13 # Number of projective measurements (shots) per random unitary
mode = 'CUE'
burn_in = 1 # determines the number of samples to be rejected during metropolis: (nu*burn_in) 



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
p_depo = 0.2

## Theoretical estimations:
p2_exp = (1-p_depo)**2 + (1-(1-p_depo)**2)/d ## purity of the realized noisy GHZ state
p2_theory = 1 ## Purity of the ideal pure GHZ state
fid = (1-p_depo) + p_depo/d ## Fidelity between the ideal and the experimenetal GHZ state


### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Perform Randomized measurements 
print('Randomized measurements using uniform sampling with Nu = '+str(Nu)+' and NM = '+str(NM))

### Generate Random Unitaries
unitaries=np.zeros((Nu,N,2,2),dtype=np.complex_)
for iu in range(Nu):
    for i in range(N):
        unitaries[iu,i]=SingleQubitRotation(random_gen,mode)
print('Random unitaries generated using uniform sampling')

### Simulate the randomized measurements
Meas_Data_uni = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = ObtainOutcomeProbabilities(N, qstate, unitaries[iu] , p_depo)
    Meas_Data_uni[iu,:] = Sampling_Meas(prob,N,NM)
print('Measurement data generated for uniform sampling \n')


## Estimate the uniform sampled purity
X_uni = np.zeros(Nu)
for iu in range(Nu):
    print('Postprocessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    probe = get_prob(Meas_Data_uni[iu,:], N)
    X_uni[iu] = get_X(probe,N)

p2_uni = 0 # purity given by uniform sampling
p2_uni = unbias(np.mean(X_uni),N,NM)

print('Randomized measurements using importance sampling with Nu = '+str(Nu)+' and NM = '+str(NM))

### Step 1: Preprocessing step for importance sampling. Sample Y and Z rotation angles (2N angles for each unitary u)  
# Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from an ideal GHZ state
theta_is, phi_is, n_r, N_s, p_IS = MetropolisSampling_pure(N, qstate,Nu, burn_in) 

### Step: Randomized measurements

## Step 2a: Perform the actual experiment on your quantum machine
# Store angles   theta_is, phi_is on the hard drive
# np.savetxt('theta_is.txt',theta_is) ## text file with Nu rows and N columns containing angles
# np.savetxt('phi_is.txt',phi_is) ## text file with Nu rows and N columns containing angles
# >>>> Run your quantum machine <<<<
# Load measurement results from hard drive as an array of shape (Nu,NM) containing integers
#Meas_Data_IS = np.load('MeasurementResults.npy',dtype='int64')

## Step 2b: Simulate randomized measurements with the generated importance sampled unitaries

### Generate the local importance sampled Random Unitaries
unitaries=np.zeros((Nu,N,2,2),dtype=np.complex_)
for iu in range(Nu):
    for i in range(N):
        unitaries[iu,i]=SingleQubitRotationIS(theta_is[i,iu],phi_is[i,iu])
print('Importance sampled random unitaries generated')

### Simulate the randomized measurements
Meas_Data_IS = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob = ObtainOutcomeProbabilities(N, qstate, unitaries[iu] , p_depo)
    Meas_Data_IS[iu,:] = Sampling_Meas(prob,N,NM)
print('Measurement data generated for importance sampling')


## Step 3: Estimation of the purity given by importance sampling
X_imp = np.zeros(Nu)
for iu in range(Nu):
    print('Postprocessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    probe = get_prob(Meas_Data_IS[iu,:], N)
    X_imp[iu] = unbias(get_X(probe,N),N,NM)

p2_IS = 0 # purity given by importance sampling
for iu in range(Nu):
    p2_IS += X_imp[iu]*n_r[iu]/p_IS[iu,0]/N_s
    

### some performance illustrations
print('Fidelity of the importance sampler: ', np.round(100*fid,2), '%')
print('p2 (True value) = ', p2_exp)
print('p2 (uniform sampling) = ', p2_uni)
print('p2 (Importance sampling) = ', p2_IS)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni-p2_exp)/p2_exp),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS-p2_exp)/p2_exp),2), '%')
