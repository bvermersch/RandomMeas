import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg
from src.ObtainMeasurements import *
from src.AnalyzeMeasurements import *
from src.PreprocessingImportanceSampling import *

### This script estimates the topological entanglement entropy S_topo of a state using importance sampling and uniform sampling.

N = 9 ## total number of qubits of the state in study
d = 2**N ## Hilbert space dimension

## load the toric code state:
rho = np.load("src/N_9_sites_7_11_17_12_16_20_15_21_25.npy") 

## returns the state rho_traced for the considered partition 
def sub_system(rho,part):
    rho = Qobj(rho, [[2]*N,[2]*N])
    if (len(part) == 9):
        rho_traced = rho
    else:
        rho_traced = ptrace(rho,part)
    return rho_traced

## Partitions considered indexed by the qubit numbers of the 9 qubit sub-system
qubit_partitions = [[0,1,5],[2,3,7],[3,6,8],[0,1,2,3,5,7],[0,1,4,5,6,8],[2,3,4,6,7,8],[0,1,2,3,4,5,6,7,8]] ## indexed by thhe respective qubit numbers 
Traced_systems = [[2,3,4,6,7,8],[0,1,2,4,5,7],[0,1,2,4,5,7],[4,6,8],[2,3,7],[0,1,5],[]] ## qubit indices that are traced out for each sub-system
num_partitions = len(qubit_partitions) ## total number of partitions

## Parameters
Nu_uni = 1000 ## number of unitaries used for uniform sampling
NM_uni = 10000 ## number of measurements performed for each uniformly sampled unitary

## could tune these values for each partition of the state. Partitions ordered as given in 'qubit_partitions'
Nu_IS = [5]*6 +[35] ## number of unitaries used for importance sampling for each partition
NM_IS = [100000]*6 + [100000] ## number of measurements done for each importance sampled unitary for each partition
burn_in = 1
mode = 'CUE'

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

print('Evalaution of S_topo using uniform sampling with Nu = '+str(Nu_uni)+' and NM = '+str(NM_uni)+' \n ')

### Perform randomized measurements with uniform sampling

Meas_Data_uni = np.zeros((Nu_uni,NM_uni),dtype='int64')
u = [0]*N
for iu in range(Nu_uni):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu_uni))),end = "",flush=True)
    for iq in range(N):
        u[iq] = SingleQubitRotation(random_gen,mode)
    Prob = ObtainOutcomeProbabilities_mixed(N, rho, u, p =0)
    Meas_Data_uni[iu,:] = Sampling_Meas(Prob, N, NM_uni)
print('Measurement data generated for uniform sampling')

### Reconstruct purities from measured bitstrings

X = np.zeros((Nu_uni,len(qubit_partitions)))
for iu in range(Nu_uni):
    print('PostProcessing {:d} % \r'.format(int(100*iu/(Nu_uni))),end = "",flush=True)
    probe = get_prob(Meas_Data_uni[iu,:],N)
    for i_part in range(len(qubit_partitions)):
        prob_subsystem = reduce_prob(probe,N,Traced_systems[i_part])
        X[iu,i_part] = unbias(get_X(prob_subsystem,len(qubit_partitions[i_part])), len(qubit_partitions[i_part]), NM_uni)
        
p2_subsystems_uni = np.zeros(num_partitions) ## storing purities for each partitions
p2_subsystems_uni = np.mean(X,0)

## Evalauting purities with importance sampling
p2_theory = np.zeros(num_partitions)
p2_subsystems_IS = np.zeros(num_partitions)
for iparts in range(num_partitions):
    print('Evaluating Importance sampled purity of the sub-system ' + str(qubit_partitions[iparts])+ ' with Nu = '+str(Nu_IS[iparts])+' and NM = '+str(NM_IS[iparts])+' \n ')
    
    N_subsystem = len(qubit_partitions[iparts]) ## number of qubits of the sub-system under study
    d_subsystem = 2**N_subsystem ## Hilbert space dimension
    rho_subsystem = np.array(sub_system(rho,qubit_partitions[iparts])) ## theoretical state modeling the state realized in the experiment
    

    ## Theoretical purity estimates for each concerned partition:
    p2_theory[iparts] = np.real(np.trace(np.dot(rho_subsystem,rho_subsystem)))
    
    ### Step 1: Preprocessing step for importance sampling. Sample Y and Z rotation angles (N_subsystem angles for each unitary u)  
    # Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from the concerned subsystem of the toric code ground state
    theta_is, phi_is, n_r, N_s, p_IS = MetropolisSampling_mixed(N_subsystem, rho_subsystem,Nu_IS[iparts], burn_in) 
    
    ### Step: Randomized measurements
    
    ## Step 2a: Perform the actual experiment on your quantum machine
    # Store angles   theta_is, phi_is on the hard drive for the specific subsystem
    # np.savetxt('theta_is.txt',theta_is) ## text file with Nu_IS[iparts] rows and N_subsystem columns containing angles
    # np.savetxt('phi_is.txt',phi_is) ## text file with Nu_IS[iparts] rows and N_subsystem columns containing angles
    # >>>> Run your quantum machine <<<<
    # Load measurement results from hard drive of each specific subsystem as an array of shape (Nu_IS[iparts],NM_IS[iparts]) containing integers
    #Meas_Data_IS = np.load('MeasurementResults_Partition'+str(iparts)+'.npy',dtype='int64')

    ## Step 2b: Simulate randomized measurements with the generated importance sampled unitaries for the specific subsystem
    
    ### Generate the local importance sampled Random Unitaries
    unitaries=np.zeros((Nu_IS[iparts],N_subsystem,2,2),dtype=np.complex_)
    for iu in range(Nu_IS[iparts]):
        for i in range(N_subsystem):
            unitaries[iu,i]=SingleQubitRotationIS(theta_is[i,iu],phi_is[i,iu])
    print('Importance sampled random unitaries generated')
    
    ### Simulate the randomized measurements
    Meas_Data_IS = np.zeros((Nu_IS[iparts],NM_IS[iparts]),dtype='int64') ## array to store the measurement results as integers representing the measured bitstrings of the specific chosen partition
    for iu in range(Nu_IS[iparts]):
        print('Data acquisition {:d} % \r'.format(int(100*iu/(Nu_IS[iparts]))),end = "",flush=True)
        prob = ObtainOutcomeProbabilities_mixed(N_subsystem, rho_subsystem, unitaries[iu] , p = 0)
        Meas_Data_IS[iu,:] = Sampling_Meas(prob,N_subsystem,NM_IS[iparts])
    print('Measurement data generated for importance sampling')

    ## Estimation of the purity for each subsystem using the concerned importance sampled unitaries
    X_imp = np.zeros(Nu_IS[iparts])
    for iu in range(Nu_IS[iparts]):
        print('Postprocessing {:d} % \r'.format(int(100*iu/(Nu_IS[iparts]))),end = "",flush=True)
        probe = get_prob(Meas_Data_IS[iu,:], N_subsystem)
        X_imp[iu] = unbias(get_X(probe,N_subsystem), N_subsystem, NM_IS[iparts])
    
    p2_IS = 0 # purity given by importance sampling
    for iu in range(Nu_IS[iparts]):
        p2_IS += X_imp[iu]*n_r[iu]/p_IS[iu,0]/N_s
    p2_subsystems_IS[iparts] = np.real(p2_IS)


## Evaluating S_topo for both the sampling methods
S_topo_uni = -1*(np.sum(np.log2(p2_subsystems_uni)[0:3])-(np.sum(np.log2(p2_subsystems_uni)[3:6])) + np.log2(p2_subsystems_uni[6]))
S_topo_IS = -1*(np.sum(np.log2(p2_subsystems_IS)[0:3])-(np.sum(np.log2(p2_subsystems_IS)[3:6])) + np.log2(p2_subsystems_IS[6]))

## some performance summaries and results
## total number of meaurements that gives the number of times the concerned state was prepared in the experiment
 
print('Total number of uniform measurements used: ', Nu_uni*NM_uni)

## total number of importance sampling measurements is given by the sum of 4 different runs of the experiment 
## run(1): Evaluates the purity of partition[0] and its complement
## run(2): Evaluates the purity of partition[1] and its complement
## run(3): Evalautes the purity of partition[2] and its complement
## run(4): Evalautes the purity of the whole state (9 qubits)
print('Total number of IS measurements used: ', 3*Nu_IS[0]*NM_IS[0] + Nu_IS[6]*NM_IS[6])

print('True value of S_topo: ', -1)
print('S_topo (uniform sampling) = ', S_topo_uni)
print('S_topo (Importance sampling) = ', S_topo_IS)
print ('Error uniform: ', np.round(100*(np.abs(S_topo_uni+1)),2), '%')
print ('Error IS: ', np.round(100*(np.abs(S_topo_IS+1)),2), '%')

