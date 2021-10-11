import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

## Rotation along the Y axis of an angle theta
def RY(angle1):
    return Qobj([[np.cos(angle1 / 2), -np.sin(angle1 / 2)],
                     [np.sin(angle1 / 2), np.cos(angle1 / 2)]], dims = [[2],[2]])

## Rotation along Z axis of an angle phi
def RZ(angle2):
    return Qobj([[np.exp(-1j * angle2 / 2), 0],
                     [0, np.exp(1j * angle2 / 2)]], dims = [[2],[2]])


N = 16 ## Number of qubits
d = 2**N ## Hilbert space dimension

### Initializing the pure GHZ state 
psi = np.zeros(d)
psi[0] = (1/2)**(0.5)
psi[-1] = (1/2)**(0.5)
psi_tens = np.reshape(psi, [2]*N)

# Consider realizing a noisy version of the GHZ state experimentally. Noise given by depolarization noise strength p
p_depo = 0.15

## Theoritical purity esitmates:
p2_exp = (1-p_depo)**2 + (1-(1-p_depo)**2)/d ## purity of the realized noisy GHZ state
p2_theory = 1 ## Purity of the ideal pure GHZ state
fid = (1-p_depo) + p_depo/d ## Fidelity between the ideal and the experimenetal GHZ state

### Initialozations for the einsum function
alphabet = "abcdefghijklmnopqsrtuvwxyz"
alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ein_command = alphabet[0:N]
for ii in range(N):
    ein_command += ','
    ein_command += alphabet[ii]+alphabet_cap[ii]
ein_command += ','+ alphabet_cap[0:N]#[0:N]

Co = ''
for i in range(N):
    Co += alphabet[i]
    Co += alphabet_cap[i]
    Co += ','
Co += alphabet_cap[:N]
Co += '->'
Co += alphabet[:N]

## Hamming array for a single qubit
## Randomized meaurement routine without shot noise
## Goal is to generate the ideal function X(u) for any given state. See Eq.(1) of https://arxiv.org/pdf/2102.13524.pdf
hamming_array = np.array([[1,-0.5],[-0.5,1]])
def RM_wosn(NN, num_nu, var_theta, var_phi):
    XX = np.zeros((1,num_nu))
    for iu in range(num_nu):
        uff = [0]*NN
        for iq in range(NN):
            uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq,iu]))*2)*RZ(2*math.pi*var_phi[iq,iu])).full()
        Listt = uff+[psi_tens]
        psiu = np.einsum(Co,*Listt,optimize = 'greedy').flatten()
        probb = np.abs(psiu)**2
        probb /= sum(probb)
        prob_tens = np.zeros([2]*NN) 
        prob_tens = np.reshape(probb, [2]*NN)
        List = [prob_tens] + [hamming_array]*N + [prob_tens]
        XX[0,iu] = np.einsum(ein_command, *List, optimize = True)*2**NN
    return XX[0,:]

## provides uniformly sampled rotation angles for RY and RZ
def Get_angles_uniform(NN, num_nu):
    return np.array(random_gen.rand(NN,num_nu)), np.array(random_gen.rand(NN,num_nu))

## provides importance sampled rotation angles along RY and RZ
def Get_angles_IS(NN, num_nu, burn_in):
    ### step 1: to perform a metropolis sampling
    count = 0
    
    ## to store the importance sampled angles for each qubit 
    samples = np.empty((0, NN*2))
    ## initialising the 
    X_theory = []
    
    accept = 0
    
    
    ## initializing the first set of 2N angles (theta & phi) sampled uniformly
    angles_0 = random_gen.uniform(0,1,(2*NN,1))
    
    ## constructing the X function of the ideal state with input angles of angles_0 
    X0 = RM_wosn(NN,1,angles_0[0:NN,:],angles_0[NN:2*NN,:])
    
    while (accept < num_nu+num_nu*burn_in):
        count += 1
        print('metropolis sampling {:d} % \r'.format(int(100*accept/(num_nu+num_nu*burn_in))),end = "",flush=True)
        
        ## generating 2N angles (theta & phi) of the candidate unitary by sampling uniformly
        angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        
        ## constructing the X function of the ideal state with input angles of angles_candidate
        X_cand = RM_wosn(NN,1,angles_candidate[0:NN,:],angles_candidate[NN:2*NN,:])
        
        ## choose a random number \beta uniformly in [0, 1]
        beta = 0
        beta = random_gen.uniform(0,1,1)
        alpha =  min(1,(X_cand/X0))
        
        ## the acceptance or rejection of candidiate angles following:
        if(beta <= alpha):
            accept += 1
            angles_0 = angles_candidate
            X0 = X_cand
        elif((count == 1) and (alpha < beta)):
            accept += 1
    
        ## append the obtained unitary angles and the corresponding weight function X_is for each u
        samples = np.append(samples, np.array([angles_0[:,0]]), axis = 0)        
        X_theory = np.append(X_theory, np.array(X0), axis = 0)
    print('\n')
    

    nu_tot = num_nu + num_nu*burn_in # total number of unitaries (including burn_in)

    ### Get the angles without burn_in and the p_is while taking into account the repeatitions to build Eq:15
    new_samples = np.zeros((nu_tot, 2*NN))  
    
    ## Define the importance sampling weights for each of the sampled u; X_weights = X_IS 
    X_weight = np.zeros((nu_tot, 1)) 
    
    ### In the following steps, we would like obtain samples of u and their occurence after excluding the burn-in samples
    ## Finding the number of occurance of each sampled unitary u
    for ii in range(2*NN):
        indexes = np.unique(samples[:,ii], return_index=True)[1]
        new_samples[:,ii] = np.array([samples[:,ii][index] for index in sorted(indexes)])
    indexes = np.unique(X_theory, return_index=True)[1]
    X_weight[:,0] = np.array([X_theory[index] for index in sorted(indexes)])   
             
    ## counts the number of occurence of each sample generated by metropolis
    counts = [list(samples[:,0]).count(i) for i in new_samples[:,0]]
    
    ## we remove the burn-in samples and its corresponding count
    new_samples = np.delete(new_samples, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    counts = np.delete(counts, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    n_s = np.sum(counts)
    
    ## imprtance sampling weights after removing the burn-in samples
    X_weight = np.delete(X_weight, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    new_samples = np.transpose(new_samples)
    
    ## We define the importance sampling probability distribution: p_is = X_theory/p2_theory with p2_theory = 1
    p_is = X_weight/p2_theory
    
    return new_samples[0:NN,:], new_samples[NN:2*NN,:], counts, n_s, p_is

def Random_Meas(NN, num_nm, var_theta, var_phi):
    d = 2**NN
    bit_strings = np.zeros(nm, dtype = 'int64')

    #print(iu)
    uff = [0]*NN
    #print(var_theta)
    for iq in range(NN):
        uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq]))*2)*RZ(2*math.pi*var_phi[iq])).full()
    Listt = uff+[psi_tens]
    psiu = np.einsum(Co,*Listt,optimize = 'greedy').flatten()
    probb = np.abs(psiu)**2*(1-p_depo) + p_depo/d
    #probb /= sum(probb)
    prob_tens = np.zeros([2]*NN) 
    prob_tens = np.reshape(probb, [2]*NN)
    List = [prob_tens] + [hamming_array]*NN + [prob_tens]
     
    bit_strings = random_gen.choice(range(2**NN), size = num_nm, p = probb) 
        
    return bit_strings 
     
def get_X_unbiased(meas_data, NN, num_nm):
    probbe = np.bincount(meas_data,minlength=2**NN)
    probe_tens = np.reshape(probbe, [2]*NN)
    Liste = [probe_tens] + [hamming_array]*N + [probe_tens]
    XX_e = np.einsum(ein_command, *Liste, optimize = True)*2**NN/(num_nm*(num_nm-1)) - 2**NN/(num_nm-1)
    return XX_e

### step 2: Sampling of the unitaries from the ideal state 
## initialize randomized measurment parameters
nu = 100  ## number of unitaries to be used 
nm = 10*d ## number of readout measurements per unitary
burn_in = 1 # number of burn_in samples: nu*burn_in 
nu_tot = nu + nu*burn_in # total number of unitaries (including burn_in)


### Step 1: Sample Y and Z rotation angles (2N angles for each unitary u)  

# uniformly sampled angles for the Y rotation (theta_uni) and Z rotation (phi_uni)
theta_uni, phi_uni = Get_angles_uniform(N, nu) ## gives uniformly sampled angles 

# Importance sampling of the angles (theta_is) and (phi_is) using metropolis algorithm from a pure GHZ state
theta_is, phi_is, n_r, N_s, p_IS = Get_angles_IS(N, nu, burn_in) 

### Step 2: Perform Randomized measurements classically to get bit string data 
## This step can be replaced by the actual experimentally recorded bit strings for the applied unitaries

Meas_Data_uni = np.zeros((nu,nm),dtype='int64')
Meas_Data_IS = np.zeros((nu,nm),dtype='int64')
for iu in range(nu):
    print('Data acquisition {:d} % \r'.format(int(100*iu/(nu))),end = "",flush=True)
    Meas_Data_uni[iu,:] = Random_Meas(N, nm, theta_uni[:,iu], phi_uni[:,iu])
    Meas_Data_IS[iu,:] = Random_Meas(N, nm, theta_is[:,iu], phi_is[:,iu])

print('\n loaded\n ')

X_uni = np.zeros(nu)
X_is = np.zeros(nu)
for iu in range(nu):
    print('Postprocessing {:d} % \r'.format(int(100*iu/(nu))),end = "",flush=True)
    X_uni[iu] = get_X_unbiased(Meas_Data_uni[iu,:],N,nm)
    X_is[iu] = get_X_unbiased(Meas_Data_IS[iu,:],N,nm)
    
print("Measurement data loaded")    

## Step 3: Estimation of the purities p2_uni and p2_IS
p2_uni = 0 # purity given by uniform sampling
p2_uni = np.mean(X_uni)
p2_IS = 0 # purity given by importance sampling
for iu in range(nu):
    p2_IS += X_is[iu]*n_r[iu]/p_IS[iu,0]/N_s

print('Fidelity of the sampler: ', fid)
print('p2 (True value) = ', p2_exp)
print('p2 (uniform sampling) = ', p2_uni)
print('p2 (Importance sampling) = ', p2_IS)
print ('Error uniform: ', np.round(100*(np.abs(p2_uni-p2_exp)/p2_exp),2), '%')
print ('Error IS: ', np.round(100*(np.abs(p2_IS-p2_exp)/p2_exp),2), '%')


"""
nexp = 1
Error_is = np.zeros(nexp)
Error_uni = np.zeros(nexp)
for exps in range(nexp):
    ## calling the metropolis function to obtain the samples
    ##      the importance sampled angles for each qubit are stored in: angles
    ##      the number of occurence of each unitary: counter   
    angles, counter, X_t = metropolis_sampling(N,nu, burn_in)
    
    ## To contain importance sampled angles for each qubit excluding the burn-in 
    new_samples = np.zeros((nu_tot, 2*N))  
    
    ## Define the importance sampling weights for each of the sampled u; X_weights = X_IS 
    X_weight = np.zeros((nu_tot, 1)) 
    
    ### In the following steps, we would like obtain samples of u and their occurence after excluding the burn-in samples
    
    ## Finding the number of occurance of each sampled unitary u
    for ii in range(2*N):
        indexes = np.unique(angles[:,ii], return_index=True)[1]
        new_samples[:,ii] = np.array([angles[:,ii][index] for index in sorted(indexes)])
        
    indexes = np.unique(X_t, return_index=True)[1]
    X_weight[:,0] = np.array([X_t[index] for index in sorted(indexes)])   
             
    ## counts the number of occurence of each sample generated by metropolis
    counts = [list(angles[:,0]).count(i) for i in new_samples[:,0]]
    
    ## we remove the burn-in samples and its corresponding count
    new_samples = np.delete(new_samples, list(np.arange(0,int(burn_in*nu),1)), axis = 0)
    counts = np.delete(counts, list(np.arange(0,int(burn_in*nu),1)), axis = 0)
    
    ## imprtance sampling weights after removing the burn-in samples
    X_weight = np.delete(X_weight, list(np.arange(0,int(burn_in*nu),1)), axis = 0)
    
    new_samples = np.transpose(new_samples)
    
    ## We define the importance sampling probability distribution: p_is = X_theory/p2_theory
    p_is = X_weight/p2_theory
    
    ### Step 3: perform randomized measurements for both importance sampling and uniform sampling: 
        
    ## randomized measurements performed using angles importance sampled from the theory state
    X_e_is = RM_sn(N,nu,nm,new_samples[0:N,:],new_samples[N:2*N,:])

    ## Computing p2_IS as given according to 
    p2_IS = 0

    for ii in range(nu):
        p2_IS += X_e_is[ii]*counts[ii]/p_is[ii,0]/np.sum(counts)        
    
    ### Protocol with uniform sampling for comparison
    
    ## Initializing an uniform sampling of the angles for comparision
    theta_uni = np.array(random_gen.rand(N,nu)) # rotation along y
    phi_uni = np.array(random_gen.rand(N,nu)) # this is the first rotation along z that is applied
    
    ## RM using uniform sampling
    X_e_uni = RM_sn(N,nu,nm,theta_uni,phi_uni)
    #X_testtt = RM_test(N,rho_exp,nu,theta_uni,phi_uni)
    ## computing uniform sampled purity: p2_uni 
    p2_uni = 0 
    p2_uni = np.mean(X_e_uni)
    Error_is[exps] = np.abs(p2_IS-p2_exp)/p2_exp
    Error_uni[exps] = np.abs(p2_uni-p2_exp)/p2_exp

print(np.mean(Error_is))
print(np.mean(Error_uni))
print('purity IS:', p2_IS)
print('purity uni:', p2_uni)

def metropolis_sampling(NN, n_samples, burn_in):
    count = 0
    
    ## to store the importance sampled angles for each qubit 
    samples = np.empty((0, NN*2))
    ## initialising the 
    X_theory = []
    
    accept = 0
    
    ## initializing the first set of 2N angles (theta & phi) sampled uniformly
    angles_0 = random_gen.uniform(0,1,(2*NN,1))
    
    ## constructing the X function of the ideal state with input angles of angles_0 
    X0 = RM_wosn(NN,1,angles_0[0:NN,:],angles_0[NN:2*NN,:])
    
    while (accept < n_samples+n_samples*burn_in):
        #print('Metropolis sampling {:d} % \r'.format(int(100*accept/(n_samples+n_samples*burn_in))),end = "",flush=True)
        count += 1
        
        ## generating 2N angles (theta & phi) of the candidate unitary by sampling uniformly
        angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        
        ## constructing the X function of the ideal state with input angles of angles_candidate
        X_cand = RM_wosn(NN,1,angles_candidate[0:NN,:],angles_candidate[NN:2*NN,:])
        
        ## choose a random number \beta uniformly in [0, 1]
        beta = 0
        beta = random_gen.uniform(0,1,1)
        alpha =  min(1,(X_cand/X0))
        
        ## the acceptance or rejection of candidiate angles following:
        if(beta <= alpha):
            accept += 1
            angles_0 = angles_candidate
            X0 = X_cand
        elif((count == 1) and (alpha < beta)):
            accept += 1
    
        ## append the obtained unitary angles and the corresponding weight function X_is for each u
        samples = np.append(samples, np.array([angles_0[:,0]]), axis = 0)        
        X_theory = np.append(X_theory, np.array(X0), axis = 0)
    
    return samples, count, X_theory

def RM_sn(NN,num_nu, num_nm, var_theta, var_phi):
    d = 2**NN
    XX_e = np.zeros((1,num_nu))
    bit_strings = np.zeros(nm, dtype = 'int64')
    for iu in range(num_nu):
        #print(iu)
        uff = [0]*NN
        for iq in range(NN):
            uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq,iu]))*2)*RZ(2*math.pi*var_phi[iq,iu])).full()
        Listt = uff+[psi_tens]
        psiu = np.einsum(Co,*Listt,optimize = 'greedy').flatten()
        probb = np.abs(psiu)**2*(1-p) + p/d
        #probb /= sum(probb)
        prob_tens = np.zeros([2]*NN) 
        prob_tens = np.reshape(probb, [2]*NN)
        List = [prob_tens] + [hamming_array]*NN + [prob_tens]
         
        bit_strings = random_gen.choice(range(2**NN), size = num_nm, p = probb) 
        probbe = np.bincount(bit_strings,minlength=2**NN)

        probe_tens = np.reshape(probbe, [2]*NN)
        #hamming_array = np.array([[1,-0.5],[-0.5,1]])
        Liste = [probe_tens] + [hamming_array]*N + [probe_tens]
        XX_e[0,iu] = np.einsum(ein_command, *Liste, optimize = True)*2**NN/(num_nm*(num_nm-1)) - 2**NN/(num_nm-1)
    return XX_e[0,:]
"""
