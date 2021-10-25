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
def RY(theta):
    return Qobj([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]], dims = [[2],[2]])

## Rotation along Z axis of an angle phi
def RZ(phi):
    return Qobj([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]], dims = [[2],[2]])

## Prepare single qubit unitary u_i = RY_i*RZ_i
def single_u(ry,rz):
    return np.array(ry*rz)

## Function that counts the number of bits for a number n
def  countSetBits(n): 
    count = 0
    while (n): 
        count += n & 1
        n >>= 1
    return count 

## Randomized meaurement routine without shot noise
## Goal is to generate the ideal function X(u) for any given state. See Eq.(1) of https://arxiv.org/pdf/2102.13524.pdf
## Input parameters:
##      NN: Number of qubits, rho_rm: input state to perform RM, num_nu: number of random unitaries, 
##      var_theta: Y rotations for each qubit, var_phi: Z rotations for each qubit
def RM(NN,rho_rm, num_nu, var_theta, var_phi, mode):   
    
    bitcount = np.zeros((1,2**NN))
    for ii in range(2**NN):
        bitcount[:,ii] = (-2.)**(-countSetBits(ii))
    
    ## Initializing the angles in their respective intervals
    var_theta = 2*math.pi * var_theta
    var_phi  = 2*math.pi* var_phi
    
    ## Vector to strore values of the function X for each u
    X = np.zeros((NN, num_nu))
    
    #partition_string_system = ['1'*x +'0'*(NN-x) for x in range(1,NN+1)]
    #partition_system = np.array([int(p,2) for p in partition_string_system])
    
    for iu in range(num_nu):
        
        probb = np.zeros((1, 2**NN))
        uf = [1]
        u_temp = 1j*(np.zeros((NN,2,2)))
        
        for qubits in range(NN):
            u_temp[qubits,:,:] = np.array(RY(var_theta[qubits, iu])*RZ(var_phi[qubits, iu]))
            uf = (np.kron(uf,u_temp[qubits,:,:]))
        #print(uf)
        ## probabilities obtained by performing the randomized measurements on rho_rm
        probbe = np.real(np.einsum('ab,bc,ac->a', uf, rho_rm, np.conjugate(rho_rm), optimize = 'greedy'))
        probb = np.real(np.diag(uf.dot(rho_rm).dot(np.conj(np.transpose(uf)))))
        
        ## Constructing the X function for each unitary
        #qubits = NN # number of qubits, # qubit is the variable in this loop
        probbe = probb
        
        ## dimension of the Hilbert space
        d = 2**NN
        
        M = np.zeros((d, d))
        M = np.dot(probbe[:,None], probbe[None,:])
    
        xor_mat = np.zeros((d, d))
        xor_mat = np.bitwise_xor(np.arange(d)[:,None],np.arange(d)[None,:])

        temp = xor_mat
        for ii in range(d-1,0,-1):
            xor_matt = np.where(temp == ii, bitcount[0,ii], temp)
            temp = xor_matt
        xor_matt =  np.where(xor_matt == 0, 1, xor_matt)   

        X_temp = M*xor_matt
        
        ## final vector of the X function without shot noise
        X[0,iu] = d*np.sum(X_temp.flatten())

            
    return X[0,:]

## provides uniformly sampled rotation angles for RY and RZ
def Get_angles_uniform(NN, num_nu):
    return np.array(random_gen.rand(NN,num_nu)), np.array(random_gen.rand(NN,num_nu))
    
## provides importance sampled rotation angles along RY and RZ
def Get_angles_IS(NN, rho_theory, num_nu, burn_in):
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
    X0 = RM(NN,rho_theory,1,angles_0[0:NN,:],angles_0[NN:2*NN,:])
    
    while (accept < num_nu+num_nu*burn_in):
        count += 1
        
        ## generating 2N angles (theta & phi) of the candidate unitary by sampling uniformly
        angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        
        ## constructing the X function of the ideal state with input angles of angles_candidate
        X_cand = RM(NN,rho_theory,1,angles_candidate[0:NN,:],angles_candidate[NN:2*NN,:])
        
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
    

    nu_tot = num_nu + num_nu*burn_in # total number of unitaries (including burn_in)

    ### Step 2: To get the angles without burn_in and the p_is while taking into account the repeatitions to build Eq:15
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
    p2_ideal = np.trace(np.dot(rho_theory,rho_theory))
    ## We define the importance sampling probability distribution: p_is = X_theory/p2_theory
    p_is = X_weight/p2_ideal
    
    return new_samples[0:NN,:], new_samples[NN:2*NN,:], counts, n_s, p_is

def Random_Meas(NN, rho_rm, num_nm, var_theta, var_phi):

    ## Initializing the angles in their respective intervals
    var_theta = 2*math.pi * var_theta
    var_phi  = 2*math.pi* var_phi
    
    ## Vector to strore values of the function X for each u
    #X_e = np.zeros((NN, num_nu))
    
    #partition_string_system = ['1'*x +'0'*(NN-x) for x in range(1,NN+1)]
    #partition_system = np.array([int(p,2) for p in partition_string_system])
    
    #for iu in range(num_nu):
    ## initializing the array of bitstrings measured for each applied u
    bit_strings = np.zeros((1,num_nm), dtype = int)
    probb = np.zeros((1, 2**NN))
    uf = [1]
    u_temp = 1j*(np.zeros((NN,2,2)))
    
    for qubits in range(NN):
        u_temp[qubits,:,:] = single_u(RY(var_theta[qubits]), RZ(var_phi[qubits]))
        uf = 1j*(np.kron(uf,u_temp[qubits,:,:]))
    #print(uf)
    ## probabilities obtained by performing the randomized measurements on rho_rm
    probb = np.real(np.diag(uf.dot(rho_rm).dot(np.conj(np.transpose(uf)))))  
    
    ## Hilbert space dimension
    d = 2**NN            
    ## loading the bitstrings measured during the experiment
    bit_strings[0,:] = random_gen.choice(range(2**NN), size = num_nm, p = probb) # bit strings measured as in an experiment for each random unitary applied to the qubits

    return bit_strings

def Build_X_unbiased(meas_data, NN, num_nm):
    
    bitcount = np.zeros((1,2**NN))
    for ii in range(2**NN):
        bitcount[:,ii] = (-2.)**(-countSetBits(ii))
        
    #bit_strings = meas_data
    d = 2**NN
    histto = np.zeros(d).astype(float)
    for ii in range(d):
        histto[ii] = np.count_nonzero(meas_data == ii)               
                
    M = np.zeros((d, d))
    M = np.dot(histto[:,None], histto[None,:])

    xor_mat = np.zeros((d, d))
    xor_mat = np.bitwise_xor(np.arange(2**NN)[:,None],np.arange(2**NN)[None,:])

    temp = xor_mat
    for ii in range(d-1,0,-1):
        xor_matt = np.where(temp == ii, bitcount[0,ii], temp)
        temp = xor_matt
    xor_matt =  np.where(xor_matt == 0, 1, xor_matt)   
    X_temp = np.sum(M*xor_matt)
    
    ## unbiasing the estimation
    X_e = (X_temp*2**(NN))/(num_nm*(num_nm-1))-2**(NN)/(num_nm-1)
        
    return X_e
