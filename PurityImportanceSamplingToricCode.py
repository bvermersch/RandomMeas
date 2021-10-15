import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
from Functions_IS_mixed import *

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### This script estimates the topological entanglement entropy S_topo of a state using importance sampling and uniform sampling.

N = 9 ## total number of qubits of the state in study
d = 2**N ## Hilbert space dimension


## returns the state rho for the considered partition 
def sub_system(part):
    rho = np.load("N_9_sites_7_11_17_12_16_20_15_21_25.npy") 
    rho = Qobj(rho, [[2]*N,[2]*N])
    if (len(part) == 9):
        rho_traced = rho
    else:
        rho_traced = ptrace(rho,part)
    return rho_traced

## Partitions considered indexed by the qubit numbers of the 9 qubit sub-system
qubit_partitions = [[0,1,5],[2,3,7],[3,6,8],[0,1,2,3,5,7],[0,1,4,5,6,8],[2,3,4,6,7,8],[0,1,2,3,4,5,6,7,8]] ## indexed by thhe respective qubit numbers 
num_partitions = len(qubit_partitions) ## total number of partitions


nu_uni = 1000 ## number of unitaries used for uniform sampling
nm_uni = 10000 ## number of measurements performed for each uniformly sampled unitary

## could tune these values for each partition of the state. Partitions ordered as given in 'qubit_partitions'
nu_IS = [5]*6 +[35] ## number of unitaries used for importance sampling for each partition
nm_IS = [100000]*6 + [100000] ## number of measurements done for each importance sampled unitary for each partition
burn_in = 1

# Could consider realizing a noisy version ofthe state experimentally. Noise given by depolarization noise strength p_depo
p_depo = 0

print('Evalaution of S_topo using importance sampling in the regime N_M >> N_u \n ')

## storing purities for each partitions
p2_subsystems_IS = np.zeros(num_partitions)
p2_subsystems_uni = np.zeros(num_partitions)
p2_theory = np.zeros(num_partitions)
p2_exp = np.zeros(num_partitions)


for iparts in range(num_partitions):
    print('Evaluating purity of the sub-system indexed by the qubit numbers' + str(qubit_partitions[iparts]))
    
    N_subsystem = len(qubit_partitions[iparts]) ## number of qubits of the sub-system under study
    d_subsystem = 2**N_subsystem ## Hilbert space dimension
    rho_theory = np.array(sub_system(qubit_partitions[iparts])) ## theoretical state modeling the state realized in the experiment
    rho_exp = (1-p_depo)*rho_theory + p_depo*np.ones((d_subsystem,d_subsystem)) ## could be a noisy version of the state
    
    ## Theoretical purity esitmates for each concerned partition:
    p2_theory[iparts] = np.real(np.trace(np.dot(rho_theory,rho_theory)))
    p2_exp[iparts] = np.real(np.trace(np.dot(rho_exp,rho_exp)))

Meas_Data_uni = np.zeros((nu_uni,nm_uni),dtype='int64')
theta_uni, phi_uni = Get_angles_uniform(N_subsystem, nu_uni) ## gives uniformly sampled angles 
for iu in range(nu_uni):
        print('Data acquisition {:d} % \r'.format(int(100*iu/(nu_uni))),end = "",flush=True)
        # uniformly sampled angles for the Y rotation (theta_uni) and Z rotation (phi_uni)
        Meas_Data_uni[iu,:] = Random_Meas(N, rho_exp,nm_uni, theta_uni[:,iu], phi_uni[:,iu])

X_uni = np.zeros((nu_uni,num_partitions))
for iu in range(nu_uni):
    print('Postprocessing partition {:d} {:d} % \r'.format(iparts,int(100*iu/(nu_uni))),end = "",flush=True)
    Meas_iu  = np.array(Meas_Data_uni[iu,:],dtype=">i2").view(np.uint8)
    #Meas_unpacked = np.unpackbits(Meas_iu).reshape((nm_uni,16))[:,-N:]
    for iparts in range(num_partitions):
        partition = sum([2**i for i in qubit_partitions[iparts]])
        #N_subsystem = len(qubit_partitions[iparts]) ## number of qubits of the sub-system under study
        #Meas_SubSystem_unpacked = Meas_unpacked[:,qubit_partitions[iparts]]
        Meas_SubSystem = np.bitwise_and(Meas_iu,partition)
        X_uni[iu,iparts] = get_X_unbiased(Meas_SubSystem,N_subsystem,nm_uni)
p2_subsystems_uni = np.real(np.mean(X_uni,0))


for iparts in range(num_partitions):
    
    ### Step 1: Sample Y and Z rotation angles   
    
    
    # Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from a pure GHZ state
    theta_is, phi_is, n_r, N_s, p_IS = Get_angles_IS(N_subsystem, rho_theory,nu_IS[iparts], burn_in) 
    
    ### Step 2: Perform Randomized measurements classically to get bit string data 
    
    ## This step can be replaced by the actual experimentally recorded bit strings for the applied unitaries
    Meas_Data_IS = np.zeros((nu_IS[iparts],nm_IS[iparts]),dtype='int64')
    
        
    for iu in range(nu_IS[iparts]):
        Meas_Data_IS[iu,:] = Random_Meas(N_subsystem, rho_exp,nm_IS[iparts], theta_is[:,iu], phi_is[:,iu])
    
    print('\n Measurement data loaded\n ')
    
    ## Step 3: Estimation of the purities p2_uni and p2_IS
    
    X_imp = np.zeros(nu_IS[iparts])
        
    for iu in range(nu_IS[iparts]):
        X_imp[iu] = get_X_unbiased(Meas_Data_IS[iu,:],N_subsystem,nm_IS[iparts])
    
    p2_IS = 0 # purity given by importance sampling
    for iu in range(nu_IS[iparts]):
        p2_IS += X_imp[iu]*n_r[iu]/p_IS[iu,0]/N_s
    ## Purities stored for each concerned sub-systems
    p2_subsystems_IS[iparts] = np.real(p2_IS)

## Evaluating S_topo for both the sampling methods
S_topo_uni = -1*(np.sum(np.log2(p2_subsystems_uni)[0:3])-(np.sum(np.log2(p2_subsystems_uni)[3:6])) + np.log2(p2_subsystems_uni[6]))
S_topo_IS = -1*(np.sum(np.log2(p2_subsystems_IS)[0:3])-(np.sum(np.log2(p2_subsystems_IS)[3:6])) + np.log2(p2_subsystems_IS[6]))

## some performace summaries and results
## total number of meaurements that gives the number of times the concerned state was prepared in the experiment
 
print('Total number of uniform measurements used: ', nu_uni*nm_uni)

## total number of importance sampling measurements is given by the sum of 4 different runs of the experiment 
## run(1): Evaluates the purity of partition[0] and its complement
## run(2): Evaluates the purity of partition[1] and its complement
## run(3): Evalautes the purity of partition[2] and its complement
## run(4): Evalautes the purity of the whole state (9 qubits)
print('Total number of IS measurements used: ', 3*nu_IS[3]*nm_IS[3] + nu_IS[6]*nm_IS[6])

print('True value of S_topo: ', -1)
print('S_topo (uniform sampling) = ', S_topo_uni)
print('S_topo (Importance sampling) = ', S_topo_IS)
print ('Error uniform: ', np.round(100*(np.abs(S_topo_uni+1)),2), '%')
print ('Error IS: ', np.round(100*(np.abs(S_topo_IS+1)),2), '%')

