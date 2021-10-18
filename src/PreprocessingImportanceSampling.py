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

def SingleQubitRotationIS(theta,phi):
    U = np.array(RY(theta)*RZ(phi))
    return U

### Initialozations for the einsum function
alphabet = "abcdefghijklmnopqsrtuvwxyz"
alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

## provides importance sampled rotation angles along RY and RZ
def MetropolisSampling_pure(NN, psi, num_nu, burn_in):
    ### step 1: to perform a metropolis sampling
    count = 0
    p2_theory = 1
    ## to store the importance sampled angles for each qubit 
    samples = np.empty((0, NN*2))

    X_theory = []
    
    accept = 0
    
    ## initializing the first set of 2N angles (theta & phi) sampled uniformly
    angles_0 = random_gen.uniform(0,1,(2*NN,1))
    
    ## constructing the X function of the ideal state with input angles of angles_0 
    u_0 = [0]*NN
    for iq in range(NN):
        u_0[iq] = SingleQubitRotationIS(np.arcsin(np.sqrt(angles_0[iq,0]))*2, 2*math.pi*angles_0[iq+NN,0])
    prob0 = Simulate_Meas_pseudopure(NN,psi,0,u_0)
    prob0 = np.reshape(prob0, [2]*NN)
    X0 = get_X(prob0,NN)

    
    #X00 = get_X_ideal_pure(NN,psi,angles_0[0:NN,:],angles_0[NN:2*NN,:])
    while (accept < num_nu+num_nu*burn_in):
        
        count += 1
        print('Metropolis sampling {:d} % \r'.format(int(100*accept/(num_nu+num_nu*burn_in))),end = "",flush=True)
        
        ## generating 2N angles (theta & phi) of the candidate unitary by sampling uniformly
        angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        u_cand = [0]*NN
        for iq in range(NN):
            u_cand[iq] = SingleQubitRotationIS(np.arcsin(np.sqrt(angles_candidate[iq,0]))*2, 2*math.pi*angles_candidate[iq+NN,0])
        prob_cand = Simulate_Meas_pseudopure(NN,psi,0,u_cand)
        prob_cand = np.reshape(prob_cand, [2]*NN)
        X_cand = get_X(prob_cand,NN)    
        ## constructing the X function of the ideal state with input angles of angles_candidate
        #X_cand = get_X_ideal_pure(NN,psi,angles_candidate[0:NN,:],angles_candidate[NN:2*NN,:])
        
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
        X_theory = np.append(X_theory, np.array([X0]), axis = 0)
        
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
    
    return 2*np.arcsin(np.sqrt(new_samples[0:NN,:])), 2*math.pi*new_samples[NN:2*NN,:], counts, n_s, p_is

## provides importance sampled rotation angles along RY and RZ
def MetropolisSampling_mixed(NN, rho, num_nu, burn_in):
    ### step 1: to perform a metropolis sampling
    count = 0
    p2_theory = np.trace(np.dot(rho,rho))
    ## to store the importance sampled angles for each qubit 
    samples = np.empty((0, NN*2))
    ## initialising the 
    X_theory = []
    
    accept = 0
    
    ## initializing the first set of 2N angles (theta & phi) sampled uniformly
    angles_0 = random_gen.uniform(0,1,(2*NN,1))
    
    ## constructing the X function of the ideal state with input angles of angles_0 
    u_0 = [0]*NN
    for iq in range(NN):
        u_0[iq] = SingleQubitRotationIS(np.arcsin(np.sqrt(angles_0[iq,0]))*2, 2*math.pi*angles_0[iq+NN,0])
    prob0 = Simulate_Meas_mixed(NN,rho,u_0)
    prob0 = np.reshape(prob0, [2]*NN)
    X0 = get_X(prob0,NN)
    
    while (accept < num_nu+num_nu*burn_in):
        
        count += 1
        print('Metropolis sampling {:d} % \r'.format(int(100*accept/(num_nu+num_nu*burn_in))),end = "",flush=True)
        
        ## generating 2N angles (theta & phi) of the candidate unitary by sampling uniformly
        #angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        u_cand = [0]*NN
        for iq in range(NN):
            u_cand[iq] = SingleQubitRotationIS(np.arcsin(np.sqrt(angles_candidate[iq,0]))*2, 2*math.pi*angles_candidate[iq+NN,0])
        prob_cand = Simulate_Meas_mixed(NN,rho,u_cand)
        prob_cand = np.reshape(prob_cand, [2]*NN)
        X_cand = get_X(prob_cand,NN)    
        ## constructing the X function of the ideal state with input angles of angles_candidate
        #X_cand = get_X_ideal_mixed(NN,rho,angles_candidate[0:NN,:],angles_candidate[NN:2*NN,:])
        
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
        X_theory = np.append(X_theory, np.array([X0]), axis = 0)
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
    if (round(p2_theory,6) == 2**(-NN)):
        X_weight = 2**(-NN)*np.ones((nu_tot,1))
    else:    
        X_weight[:,0] = np.array([X_theory[index] for index in sorted(indexes)])            
    # counts the number of occurence of each sample generated by metropolis
    counts = [list(samples[:,0]).count(i) for i in new_samples[:,0]]
             
    ## counts the number of occurence of each sample generated by metropolis
    #counts = [list(samples[:,0]).count(i) for i in new_samples[:,0]]
    
    ## we remove the burn-in samples and its corresponding count
    new_samples = np.delete(new_samples, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    counts = np.delete(counts, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    n_s = np.sum(counts)
    
    ## imprtance sampling weights after removing the burn-in samples
    X_weight = np.delete(X_weight, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    new_samples = np.transpose(new_samples)
    
    ## We define the importance sampling probability distribution: p_is = X_theory/p2_theory with p2_theory = 1
    p_is = X_weight/p2_theory
    
    return 2*np.arcsin(np.sqrt(new_samples[0:NN,:])), 2*math.pi*new_samples[NN:2*NN,:], counts, n_s, p_is