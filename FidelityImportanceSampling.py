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

# Code to reconstruct the cross-platform fidelity and purities of two quantum devices via Importance Sampling

import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *
from src.PreprocessingImportanceSampling import *

## Parameters
N = 15 # Number of qubits to analyze
d = 2**N
Nu = 20 # Number of random unitaries to be used
NM = d*2 # Number of projective measurements (shots) per random unitary
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
#p_depo1 = 0.1
#p_depo2 = 0.1



## Consider realizing the two noisy versions of the GHZ state experimentally given by the depolarization noise
p_depo1 = 0.2
p_depo2 = 0.4


## Theoretical purity and fidelity esitmates:
p2_exp1 = (1-p_depo1)**2 + (1-(1-p_depo1)**2)/d ## purity of first realized noisy GHZ state
p2_exp2 = (1-p_depo2)**2 + (1-(1-p_depo2)**2)/d ## purity of second realized noisy GHZ state
p2_theory = 1 ## Purity of the ideal pure GHZ state
fidelity_theory = (1-p_depo1)*(1-p_depo2) + (p_depo1*p_depo2)/d**2 + (1-p_depo1)*p_depo2/d + (1-p_depo2)*p_depo1/d 
fidelity_theory /= max(p2_exp1, p2_exp2) ## Fidelity between the 2 realized states

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)


### Perform Randomized measurements 
print('Randomized measurements using uniform sampling on each device')

### Generate Random Unitaries
unitaries=np.zeros((Nu,N,2,2),dtype=np.complex_)
for iu in range(Nu):
    for i in range(N):
        unitaries[iu,i]=SingleQubitRotation(random_gen,mode)
print('Random unitaries generated using uniform sampling')

### Simulate the randomized measurements
Meas_Data_uni_1 = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings of the first device
Meas_Data_uni_2 = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings of the second device
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    prob1 = ObtainOutcomeProbabilities_pseudopure(N, qstate, unitaries[iu] , p_depo1)
    prob2 = ObtainOutcomeProbabilities_pseudopure(N, qstate, unitaries[iu] , p_depo2)
    Meas_Data_uni_1[iu,:] = Sampling_Meas(prob1,N,NM)
    Meas_Data_uni_2[iu,:] = Sampling_Meas(prob2,N,NM)
print('Measurement data generated for uniform sampling \n')

## Estimate the fidelity and purities of each device with uniform sampling
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
RM_fidelity_uni = 0 #Fidelity given by uniform sampling

p2_uni_1 = unbias(np.mean(X_uni_1), N, NM)
p2_uni_2 = unbias(np.mean(X_uni_2),N,NM)
RM_fidelity_uni = np.mean(X_uni_fidelity)
RM_fidelity_uni /= max(p2_uni_1, p2_uni_2)


### Step 1: Preprocessing step for importance sampling. Sample Y and Z rotation angles (2N angles for each unitary u)  
print('Randomized measurements using importance sampling on each device')
# Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from an ideal GHZ state
theta_is, phi_is, n_r, N_s, p_IS = MetropolisSampling_pure(N, qstate,Nu, burn_in) 

### Step: Randomized measurements

## Step 2a: Perform the actual experiment on your quantum machine
# Store angles   theta_is, phi_is on the hard drive
# np.savetxt('theta_is.txt',theta_is) ## text file with Nu rows and N columns containing angles
# np.savetxt('phi_is.txt',phi_is) ## text file with Nu rows and N columns containing angles
# >>>> Run the two quantum machines <<<<
# Load measurement results from hard drive as an array of shape (Nu,NM) containing integers from each device
#Meas_Data_IS_1 = np.load('MeasurementResults_Device1.npy',dtype='int64')
#Meas_Data_IS_2 = np.load('MeasurementResults_Device2.npy',dtype='int64')

## Step 2b: Simulate randomized measurements with the generated importance sampled unitaries

### Generate the local importance sampled Random Unitaries
unitaries=np.zeros((Nu,N,2,2),dtype=np.complex_)
for iu in range(Nu):
    for i in range(N):
        unitaries[iu,i]=SingleQubitRotationIS(theta_is[i,iu],phi_is[i,iu])
print('Importance sampled random unitaries generated')

### Simulate the randomized measurements
Meas_Data_IS_1 = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings from the first device
Meas_Data_IS_2 = np.zeros((Nu,NM),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings from the second device
for iu in range(Nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
    Prob1 = ObtainOutcomeProbabilities_pseudopure(N, qstate, unitaries[iu], p_depo1)
    Prob2 = ObtainOutcomeProbabilities_pseudopure(N, qstate, unitaries[iu], p_depo2)
    Meas_Data_IS_1[iu,:] = Sampling_Meas(Prob1, N, NM) 
    Meas_Data_IS_2[iu,:] = Sampling_Meas(Prob2, N, NM) 
print('Measurement data generated for importance sampling \n')

## Step 3: Estimation of the fidelity and purities of each quantum device using importance sampling
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

p2_IS_1 = 0 
p2_IS_2 = 0 
RM_fidelity_IS = 0 # Fidelity given by importance sampling

for iu in range(Nu):
    p2_IS_1 += X_imp_1[iu]*n_r[iu]/p_IS[iu,0]/N_s
    p2_IS_2 += X_imp_2[iu]*n_r[iu]/p_IS[iu,0]/N_s
    RM_fidelity_IS += X_imp_fidelity[iu]*n_r[iu]/p_IS[iu,0]/N_s

RM_fidelity_IS /= max(p2_IS_1,p2_IS_2)


## Purity results of the first device
print('p2 (True value) of the first state = ', p2_exp1)
print('p2 (uniform sampling) of the first state = ', p2_uni_1)
print('p2 (Importance sampling) = ', p2_IS_1)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni_1-p2_exp1)/p2_exp1),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS_1-p2_exp1)/p2_exp1),2), '% \n')

## Purity results of the second device
print('p2 (True value) of the second state = ', p2_exp2)
print('p2 (uniform sampling) of the second state = ', p2_uni_2)
print('p2 (Importance sampling) = ', p2_IS_2)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni_2-p2_exp2)/p2_exp2),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS_2-p2_exp2)/p2_exp2),2), '% \n')

## Results of fidelity estimation between the two devices
print('True value of fidelity between the two states: ', fidelity_theory)
print('Fidelity (uniform sampling) = ', RM_fidelity_uni)
print('Fidelity (Importance sampling) = ', RM_fidelity_IS)
print ('Error fidelity uniform: ', np.round(100*(np.abs(RM_fidelity_uni-fidelity_theory)/fidelity_theory),2), '%')
print ('Error fidelity IS: ', np.round(100*(np.abs(RM_fidelity_IS-fidelity_theory)/fidelity_theory),2), '%')
