import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
from Functions_IS import *

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### This script estimates the purity of a noisy GHZ realized in the experiment using uniform sampling and importance sampling from an ideal pure GHZ state
### Capable of simulating noisy GHZ state till N = 25 qubits !!!

N = 15 ## Number of qubits of the GHZ state
d = 2**N ## Hilbert space dimension

# Consider realizing a noisy version of the GHZ state experimentally. Noise given by depolarization noise strength p_depo
p_depo = 0.75

## Theoretical purity esitmates:
p2_exp = (1-p_depo)**2 + (1-(1-p_depo)**2)/d ## purity of the realized noisy GHZ state
p2_theory = 1 ## Purity of the ideal pure GHZ state
fid = (1-p_depo) + p_depo/d ## Fidelity between the ideal and the experimenetal GHZ state

### Creating the ideal pure GHZ state:
GHZ = np.zeros(d)
GHZ[0] = (1/2)**(0.5)
GHZ[-1] = (1/2)**(0.5)
GHZ_state = np.reshape(GHZ, [2]*N)
#mode = 'pure'

nu = 50 # number of unitaries to be used
nm = d*10 # number of measurements to be performed for each unitary
burn_in = 1 # determines the number of samples to be rejected during metropolis: (nu*burn_in) 

### Step 1: Sample Y and Z rotation angles (2N angles for each unitary u)  

# uniformly sampled angles for the Y rotation (theta_uni) and Z rotation (phi_uni)
theta_uni, phi_uni = Get_angles_uniform(N, nu) ## gives uniformly sampled angles 

# Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from a pure GHZ state
theta_is, phi_is, n_r, N_s, p_IS = Get_angles_IS(N, GHZ_state,nu, burn_in) 

### Step 2: Perform Randomized measurements classically to get bit string data 
## This step can be replaced by the actual experimentally recorded bit strings for the applied unitaries
Meas_Data_uni = np.zeros((nu,nm),dtype='int64')
Meas_Data_IS = np.zeros((nu,nm),dtype='int64')
for iu in range(nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(nu))),end = "",flush=True)
    Meas_Data_uni[iu,:] = Random_Meas(N, GHZ_state,p_depo,nm, theta_uni[:,iu], phi_uni[:,iu])
    Meas_Data_IS[iu,:] = Random_Meas(N, GHZ_state,p_depo,nm, theta_is[:,iu], phi_is[:,iu])

print('\n Measurement data loaded\n ')

X_uni = np.zeros(nu)
X_imp = np.zeros(nu)
for iu in range(nu):
    print('Postprocessing {:d} % \r'.format(int(100*iu/(nu))),end = "",flush=True)
    X_uni[iu] = get_X_unbiased(Meas_Data_uni[iu,:],N,nm)
    X_imp[iu] = get_X_unbiased(Meas_Data_IS[iu,:],N,nm)


## Step 3: Estimation of the purities p2_uni and p2_IS
p2_uni = 0 # purity given by uniform sampling
p2_uni = np.mean(X_uni)
p2_IS = 0 # purity given by importance sampling
for iu in range(nu):
    p2_IS += X_imp[iu]*n_r[iu]/p_IS[iu,0]/N_s

print('Fidelity of the sampler: ', np.round(100*fid,2), '%')
print('p2 (True value) = ', p2_exp)
print('p2 (uniform sampling) = ', p2_uni)
print('p2 (Importance sampling) = ', p2_IS)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni-p2_exp)/p2_exp),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS-p2_exp)/p2_exp),2), '%')
