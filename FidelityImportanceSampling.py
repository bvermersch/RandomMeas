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

# Codes to reconstruct the purity via Importance Sampling

import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
import sys
sys.path.append("src")
from ObtainMeasurements import *
from AnalyzeMeasurements import *
from PreprocessingImportanceSampling import *

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### This script estimates the fidelity between two noisy GHZ states realized in two separate devices using uniform sampling and importance sampling from an ideal pure GHZ state
### Capable of simulatingtill N = 25 qubits !!!

N = 15 ## Number of qubits of the GHZ state
d = 2**N ## Hilbert space dimension

# Consider realizing the two noisy versions of the GHZ state experimentally. Noise given by depolarization noise strength p_depo
p_depo1 = 0.2
p_depo2 = 0.5


## Theoretical purity and fidelity esitmates:
p2_exp1 = (1-p_depo1)**2 + (1-(1-p_depo1)**2)/d ## purity of first realized noisy GHZ state
p2_exp2 = (1-p_depo2)**2 + (1-(1-p_depo2)**2)/d ## purity of second realized noisy GHZ state
p2_theory = 1 ## Purity of the ideal pure GHZ state
fidelity_theory = (1-p_depo1)*(1-p_depo2) + (p_depo1*p_depo2)/d**2 + (1-p_depo1)*p_depo2/d + (1-p_depo2)*p_depo1/d ## Fidelity between the ideal and the experimenetal GHZ state
fidelity_theory /= max(p2_exp1, p2_exp2)

### Creating the ideal pure GHZ state:
GHZ = np.zeros(d)
GHZ[0] = (1/2)**(0.5)
GHZ[-1] = (1/2)**(0.5)
GHZ_state = np.reshape(GHZ, [2]*N)


### Importance sampling provides best performances for Nu ~ O(N) and NM ~O(2^N) !!
Nu = 20 # number of unitaries to be used
NM = d*5 # number of measurements to be performed for each unitary
burn_in = 1 # determines the number of samples to be rejected during metropolis: (nu*burn_in) 
mode = 'CUE'

### Step 2: Perform Randomized measurements classically to get bit string data 
## This step can be replaced by the actual experimentally recorded bit strings for the applied unitaries

print('Randomized measurements using uniform sampling')
Meas_Data_uni_1 = np.zeros((Nu,NM),dtype='int64')
Meas_Data_uni_2 = np.zeros((Nu,NM),dtype='int64')

## Perform randomized measurements using same the uniformly sampled unitaries on both the setups
u = [0]*N
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    for iq in range(N):
        u[iq] = SingleQubitRotation(random_gen,mode)
    Prob1 = Simulate_Meas_pseudopure(N, GHZ_state, p_depo1, u)
    Prob2 = Simulate_Meas_pseudopure(N, GHZ_state,p_depo2,u)
    Meas_Data_uni_1[iu,:] = Sampling_Meas(Prob1, N, NM) ## bit string data from the first device
    Meas_Data_uni_2[iu,:] = Sampling_Meas(Prob2, N, NM) ## bitstring data from the second device
    #Meas_Data[iu,:] = Simulate_Meas_mixed(N, rho, NM, u)
print('Measurement data generated for uniform sampling \n')

X_uni_1 = np.zeros(Nu)
X_uni_2 = np.zeros(Nu)
X_uni_fidelity = np.zeros(Nu)
for iu in range(Nu):
    print('Postprocessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob1e = get_prob(Meas_Data_uni_1[iu,:], N)
    X_uni_1[iu] = get_X(prob1e,N)
    prob2e = get_prob(Meas_Data_uni_2[iu,:], N)
    X_uni_2[iu] = get_X(prob2e,N)
    X_uni_fidelity[iu] = get_X_overlap(prob1e,prob2e,N)

p2_uni_1 = 0
p2_uni_2 = 0 
RM_fidelity_uni = 0

p2_uni_1 = unbias(np.mean(X_uni_1), N, NM)
p2_uni_2 = unbias(np.mean(X_uni_2),N,NM)
RM_fidelity_uni = np.mean(X_uni_fidelity)
RM_fidelity_uni /= max(p2_uni_1, p2_uni_2)

### Step 1: Preprocessing step for importance sampling Sample Y and Z rotation angles (2N angles for each unitary u)  
print('Randomized measurements using importance sampling ')
# Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from a pure GHZ state
theta_is, phi_is, n_r, N_s, p_IS = MetropolisSampling_pure(N, GHZ_state,Nu, burn_in) 

## Step 2: Perform randomized measurements with the generated the importance sampled unitaries in both the setups
Meas_Data_IS_1 = np.zeros((Nu,NM),dtype='int64')
Meas_Data_IS_2 = np.zeros((Nu,NM),dtype='int64')
u = [0]*N
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    for iq in range(N):
        u[iq] = SingleQubitRotationIS(theta_is[iq,iu],phi_is[iq,iu])
    Prob1 = Simulate_Meas_pseudopure(N, GHZ_state, p_depo1, u)
    Prob2 = Simulate_Meas_pseudopure(N, GHZ_state,p_depo2,u)
    Meas_Data_IS_1[iu,:] = Sampling_Meas(Prob1, N, NM) ## bitstring data from the first device
    Meas_Data_IS_2[iu,:] = Sampling_Meas(Prob2, N, NM) ## bitstring data from the second device
print('Measurement data generated for importance sampling \n')

## Step 3: Estimation of the purities p2_uni and p2_IS

X_imp_1 = np.zeros(Nu)
X_imp_2 = np.zeros(Nu)
X_imp_fidelity = np.zeros(Nu)

for iu in range(Nu):    
    print('Postprocessing {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob1e = get_prob(Meas_Data_IS_1[iu,:], N)
    X_imp_1[iu] = unbias(get_X(prob1e,N), N, NM)
    prob2e = get_prob(Meas_Data_IS_2[iu,:], N)
    X_imp_2[iu] = unbias(get_X(prob2e,N), N, NM)
    X_imp_fidelity[iu] = get_X_overlap(prob1e,prob2e,N)

p2_IS_1 = 0 # purity given by importance sampling
p2_IS_2 = 0 # purity given by importance sampling
RM_fidelity_IS = 0 # purity given by importance sampling

for iu in range(Nu):
    p2_IS_1 += X_imp_1[iu]*n_r[iu]/p_IS[iu,0]/N_s
    p2_IS_2 += X_imp_2[iu]*n_r[iu]/p_IS[iu,0]/N_s
    RM_fidelity_IS += X_imp_fidelity[iu]*n_r[iu]/p_IS[iu,0]/N_s

RM_fidelity_IS /= max(p2_IS_1,p2_IS_2)

## results of the first device
print('p2 (True value) of the first state = ', p2_exp1)
print('p2 (uniform sampling) of the first state = ', p2_uni_1)
print('p2 (Importance sampling) = ', p2_IS_1)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni_1-p2_exp1)/p2_exp1),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS_1-p2_exp1)/p2_exp1),2), '% \n')

## results of the second device
print('p2 (True value) of the second state = ', p2_exp2)
print('p2 (uniform sampling) of the second state = ', p2_uni_2)
print('p2 (Importance sampling) = ', p2_IS_2)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni_2-p2_exp2)/p2_exp2),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS_2-p2_exp2)/p2_exp2),2), '% \n')

print('True value of fidelity between the two states: ', fidelity_theory)
print('Fidelity (uniform sampling) = ', RM_fidelity_uni)
print('Fidelity (Importance sampling) = ', RM_fidelity_IS)
print ('Error fidelity uniform: ', np.round(100*(np.abs(RM_fidelity_uni-fidelity_theory)/fidelity_theory),2), '%')
print ('Error fidelity IS: ', np.round(100*(np.abs(RM_fidelity_IS-fidelity_theory)/fidelity_theory),2), '%')
