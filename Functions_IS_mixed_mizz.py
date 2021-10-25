import numpy as np
import math
import cmath
from qutip import *
import random 
from scipy import linalg

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)


hamming_array = np.array([[1,-0.5],[-0.5,1]]) ## Hamming array for a single qubit
#p2_theory = 1 ## Purity of the ideal pure GHZ state


## Rotation along the Y axis of an angle theta
def RY(theta):
    return Qobj([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]], dims = [[2],[2]])

## Rotation along Z axis of an angle phi
def RZ(phi):
    return Qobj([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]], dims = [[2],[2]])

def SingleQubitRotation():
    U = 1j*np.zeros((2,2))

    #Generate a 2x2 CUE matrix (ref: http://www.ams.org/notices/200705/fea-mezzadri-web.pdf)
    U = (random_gen.randn(2,2)+1j*random_gen.randn(2,2))/np.sqrt(2)
    q,r = linalg.qr(U)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    U = np.multiply(q,ph,q)
    return U

### Initialozations for the einsum function
alphabet = "abcdefghijklmnopqsrtuvwxyz"
alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


## Randomized meaurement routine without shot noise
## Goal is to generate the ideal function X(u) for any given state. See Eq.(1) of https://arxiv.org/pdf/2102.13524.pdf
## Input parameters:
##      NN: Number of qubits, rho_rm: input state to perform RM, num_nu: number of random unitaries, 
##      var_theta: Y rotations for each qubit, var_phi: Z rotations for each qubit
def RM_wosn(NN, state, Unitary):
    XX = np.zeros((1,1))
    Co = ''
    for i in range(NN):
        Co += alphabet[i]
        Co += alphabet_cap[i]
        Co += ','
    Co += alphabet_cap[:NN]
    Co += '->'
    Co += alphabet[:NN]
    ein_command = alphabet[0:NN]
    for ii in range(NN):
        ein_command += ','
        ein_command += alphabet[ii]+alphabet_cap[ii]
    ein_command += ','+ alphabet_cap[0:NN]

    #uff = [0]*NN
    uff = [1]
    for iq in range(NN):
        #uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq,iu]))*2)*RZ(2*math.pi*var_phi[iq,iu])).full()
        #Unitary = np.kron(Unitary, uff[iq])
        uff = np.kron(uff, Unitary[iq,:,:])
    probb = np.real(np.einsum('ab,bc,ac->a',uff,state,np.conj(uff),optimize='greedy'))
    probb /= sum(probb)
        
    prob_tens = np.zeros([2]*NN) 
    prob_tens = np.reshape(probb, [2]*NN)
    List = [prob_tens] + [hamming_array]*NN + [prob_tens]
    XX[0,0] = np.einsum(ein_command, *List, optimize = True)*2**NN
        
    return XX[0,:]

## provides uniformly sampled rotation angles for RY and RZ
def Get_angles_uniform(NN, num_nu):
    return np.array(random_gen.rand(NN,num_nu)), np.array(random_gen.rand(NN,num_nu))
    
## provides importance sampled rotation angles along RY and RZ
def Get_angles_IS(NN, state, num_nu, burn_in):
    ### step 1: to perform a metropolis sampling
    count = 0
    p2_theory = np.trace(np.dot(state,state))
    ## to store the importance sampled angles for each qubit 
    ## initialising the 
    X_theory = []
    nu_tot = num_nu + num_nu*burn_in # total number of unitaries (including burn_in)
    u_samples = np.empty((nu_tot,NN,2,2))#np.empty((0, NN*2))
    accept = 0
    
    ## initializing the first set of N local unitaries sampled uniformly
    u_00  = np.zeros((NN,2,2))
    for iq in range(NN):
        u_00[iq,:,:] = SingleQubitRotation()
    #angles_0 = random_gen.uniform(0,1,(2*NN,1))
    print(u_00)
    
    ## constructing the X function of the ideal state with input angles of angles_0 
    X0 = RM_wosn(NN,state,u_00)
    
    while (accept < nu_tot):
        
        count += 1
        print('Metropolis sampling {:d} % \r'.format(int(100*accept/(num_nu+num_nu*burn_in))),end = "",flush=True)
        
        ## generating 2N angles (theta & phi) of the candidate unitary by sampling uniformly
        angles_candidate = random_gen.uniform(0,1,(2*NN,1))
        u_candidate  = np.zeros((NN,2,2))
        for iq in range(NN):
            u_candidate[iq,:,:] = SingleQubitRotation()
            
        ## constructing the X function of the ideal state with input angles of angles_candidate
        X_cand = RM_wosn(NN,state,u_candidate)
        
        ## choose a random number \beta uniformly in [0, 1]
        beta = 0
        beta = random_gen.uniform(0,1,1)
        alpha =  min(1,(X_cand/X0))
        
        ## the acceptance or rejection of candidiate angles following:
        if(beta <= alpha):
            accept += 1
            u0_0 = u_candidate
            X0 = X_cand
        elif((count == 1) and (alpha < beta)):
            accept += 1
    
        ## append the obtained unitary angles and the corresponding weight function X_is for each u
        u_samples = np.append(u_samples, np.array([u_00[:,:,:]]), axis = 0)        
        X_theory = np.append(X_theory, np.array(X0), axis = 0)
    print('\n')
    
    #print(u_samples)
    
    #return(X_theory)

    ### Get the angles without burn_in and the p_is while taking into account the repeatitions to build Eq:15
    new_samples = np.zeros((nu_tot, NN,2,2))  
    
    ## Define the importance sampling weights for each of the sampled u; X_weights = X_IS 
    X_weight = np.zeros((nu_tot, 1)) 
    
    indexes = np.unique(X_theory, return_index=True)[1]
    if (round(p2_theory,6) == 2**(-NN)):
        X_weight = 2**(-NN)*np.ones((nu_tot,1))
    else:    
        X_weight[:,0] = np.array([X_theory[index] for index in sorted(indexes)])         
    
    ### In the following steps, we would like obtain samples of u and their occurence after excluding the burn-in samples
    ## Finding the number of occurance of each sampled unitary u
    for ii in range(NN):
        #print(ii)
        #indexes = np.unique(u_samples[:,ii,:,:], axis = 0, return_index=True)[1]
        #print(indexes)
        #print(len(indexes))
        new_samples[:,ii,:,:] = np.array([u_samples[:,ii,:,:][index] for index in sorted(indexes)])
    #indexes = np.unique(X_theory, return_index=True)[1]
    #X_weight[:,0] = np.array([X_theory[index] for index in sorted(indexes)])   
    #return(u_samples)
    #indexes = np.unique(X_theory, return_index=True)[1]
    #if (round(p2_theory,6) == 2**(-NN)):
    #    X_weight = 2**(-NN)*np.ones((nu_tot,1))
    #else:    
    #    X_weight[:,0] = np.array([X_theory[index] for index in sorted(indexes)])            
    # counts the number of occurence of each sample generated by metropolis
    counts = [list(X_weight[:,0]).count(i) for i in X_weight[:,0]]
             
    ## counts the number of occurence of each sample generated by metropolis
    #counts = [list(samples[:,0]).count(i) for i in new_samples[:,0]]
    
    ## we remove the burn-in samples and its corresponding count
    new_samples = np.delete(new_samples, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    counts = np.delete(counts, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    n_s = np.sum(counts)
    
    ## imprtance sampling weights after removing the burn-in samples
    X_weight = np.delete(X_weight, list(np.arange(0,int(burn_in*num_nu),1)), axis = 0)
    #new_samples = np.transpose(new_samples)
    
    ## We define the importance sampling probability distribution: p_is = X_theory/p2_theory with p2_theory = 1
    p_is = X_weight/p2_theory
    
    return new_samples, counts, n_s, p_is


N = 2
rho = np.zeros((2**N,2**N))
rho[0,0] = 0.5
rho[-1,-1] = 0.5
rho[-1,0] = 0.5
rho[0,-1] = 0.5
xx = Get_angles_IS(N, rho, 10, 1)    

def metropolis_sampling(N_index, nu_index):
    
    

def Random_Meas(NN, state, num_nm, var_theta, var_phi):
    d = 2**NN
    bit_strings = np.zeros(num_nm, dtype = 'int64')
    Co = ''
    for i in range(NN):
        Co += alphabet[i]
        Co += alphabet_cap[i]
        Co += ','
    Co += alphabet_cap[:NN]
    Co += '->'
    Co += alphabet[:NN]
    #print(iu)
    uff = [0]*NN
    Unitary = [1]
    for iq in range(NN):
        uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq]))*2)*RZ(2*math.pi*var_phi[iq])).full()
        Unitary = np.kron(Unitary, uff[iq])
    
    probb = np.real(np.einsum('ab,bc,ac->a',Unitary,state,np.conj(Unitary),optimize='greedy')) 
    probb /= sum(probb)
    #print(probb)
    #for iq in range(NN):
    #    uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq]))*2)*RZ(2*math.pi*var_phi[iq])).full()
    #Listt = uff+[psi_tens]
    #psiu = np.einsum(Co,*Listt,optimize = True).flatten()
    #probb = np.abs(psiu)**2*(1-p_depo) + p_depo/d ## makes the probabilities noisy by adding white noise
    #probb /= sum(probb)
    prob_tens = np.zeros([2]*NN) 
    prob_tens = np.reshape(probb, [2]*NN) ## noisy prob tensor constructed from the noisy probabilities
    bit_strings = random_gen.choice(range(2**NN), size = num_nm, p = probb) 
        
    return bit_strings 
     
def get_X_unbiased(meas_data, NN, num_nm):
    ein_command = alphabet[0:NN]
    for ii in range(NN):
        ein_command += ','
        ein_command += alphabet[ii]+alphabet_cap[ii]
    ein_command += ','+ alphabet_cap[0:NN]
    
    probbe = np.bincount(meas_data,minlength=2**NN)
    probe_tens = np.reshape(probbe, [2]*NN)
    Liste = [probe_tens] + [hamming_array]*NN + [probe_tens]
    XX_e = np.einsum(ein_command, *Liste, optimize = True)*2**NN/(num_nm*(num_nm-1)) - 2**NN/(num_nm-1)
    return XX_e