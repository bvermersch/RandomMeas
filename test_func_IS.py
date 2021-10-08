#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 20:50:53 2021

@author: aniket
"""
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

N = 16  
d = 2**N
p2 = 1
p = 0

"""
rho_theory = np.zeros((2**N, 2**N))

rho_theory[0,0] = 0.5
rho_theory[0,-1] = 0.5
rho_theory[-1,0] = 0.5
rho_theory[-1,-1] = 0.5
    
# Consider realizing a noisy version of the GHZ state experimentally. Noise given by depolarization noise strength p
p = 0
rho_exp = np.zeros((2**N, 2**N))
rho_exp = (1-p)*rho_theory + p*np.eye(2**N)/2**N

## true value of the purity to be obtained in the experiment
p2 = 0
p2 = np.trace(np.dot(rho_exp,rho_exp))

## Purity of the ideal pure state
p2_ideal = 1

## fidelity of the ideal state wrt the prepared noisy state 
fid = np.trace(np.dot(rho_theory,rho_exp))/max(p2_ideal,p2)
"""

psi = np.zeros(d)
psi[0] = (1/2)**(0.5)
psi[-1] = (1/2)**(0.5)
psi_tens = np.reshape(psi, [2]*N)

alphabet = "abcdefghijklmnopqsrtuvwxyz"
alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ein_command = alphabet[0:N]
for ii in range(N):
    ein_command += ','
    ein_command += alphabet[ii]+alphabet_cap[ii]
ein_command += ','+ alphabet_cap[0:N]#[0:N]

hamming_array = np.array([[1,-0.5],[-0.5,1]])
Co = ''
for i in range(N):
    Co += alphabet[i]
    Co += alphabet_cap[i]
    Co += ','
Co += alphabet_cap[:N]
Co += '->'
Co += alphabet[:N]

def RM_wosn(NN, num_nu, var_theta, var_phi):
    XX = np.zeros((1,num_nu))
    for iu in range(num_nu):
        uff = [0]*NN
        for iq in range(NN):
            uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq,iu]))*2)*RZ(2*math.pi*var_phi[iq,iu])).full()
        Listt = uff+[psi_tens]
        psiu = np.einsum(Co,*Listt,optimize = 'greedy').flatten()
        probb = np.abs(psiu)**2#*(1-p) + p/d
        probb /= sum(probb)
        
        prob_tens = np.zeros([2]*NN) 
        prob_tens = np.reshape(probb, [2]*NN)
        List = [prob_tens] + [hamming_array]*N + [prob_tens]
        XX[0,iu] = np.einsum(ein_command, *List, optimize = True)*2**NN
    return XX[0,:]
        
def RM_sn (NN,num_nu, num_nm, var_theta, var_phi):
    d = 2**NN
    XX_e = np.zeros((1,num_nu))
    bit_strings = np.zeros(nm, dtype = 'int64')
    for iu in range(num_nu):
        print(iu)
        uff = [0]*NN
        for iq in range(NN):
            uff[iq] = (RY(np.arcsin(np.sqrt(var_theta[iq,iu]))*2)*RZ(2*math.pi*var_phi[iq,iu])).full()
        Listt = uff+[psi_tens]
        psiu = np.einsum(Co,*Listt,optimize = 'greedy').flatten()
        probb = np.abs(psiu)**2#*(1-p) + p/d
        probb /= sum(probb)
        
        prob_tens = np.zeros([2]*NN) 
        prob_tens = np.reshape(probb, [2]*NN)
        List = [prob_tens] + [hamming_array]*NN + [prob_tens]
         
        bit_strings = random_gen.choice(range(2**NN), size = num_nm, p = probb) 
        pseudo_string = np.append(bit_strings, [d-1]) ## added an extra measurement for (d-1)
        probbe = np.bincount(pseudo_string)
        probbe[-1] = probbe[-1] - 1 
        probe_tens = np.reshape(probbe, [2]*NN)
        #hamming_array = np.array([[1,-0.5],[-0.5,1]])
        Liste = [probe_tens] + [hamming_array]*N + [probe_tens]
        XX_e[0,iu] = np.einsum(ein_command, *Liste, optimize = True)*2**NN/(num_nm*(num_nm-1)) - 2**NN/(num_nm-1)
    return XX_e[0,:]


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
        print('metropolis count', count)
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


### step 2: Sampling of the unitaries from the ideal state 
## initialize randomized measurment parameters
nu = 50 #number of unitaries to be used 
nm = 100000# number of readout measurements per unitary
burn_in = 1 # number of burn_in samples: nu*burn_in 
nu_tot = nu + nu*burn_in # total number of unitaries (including burn_in)

exp = 20
Error_is = np.zeros(exp)
Error_uni = np.zeros(exp)
for exps in range(exp):
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
    p_is = X_weight/p2
    
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
    Error_is[exps] = np.abs(p2_IS-p2)/p2
    Error_uni[exps] = np.abs(p2_uni-p2)/p2

print(np.mean(Error_is))
print(np.mean(Error_uni))



"""   
for iexp in range(nexp):
theta = np.array(random_gen.rand(N,nu))
phi  = np.array(random_gen.rand(N,nu))

#rho_rm = Qobj(rho_exp, [[2]*N,[2]*N])
#bitcount = np.zeros((1,2**N))
#for ii in range(2**N):
#    bitcount[:,ii] = (-2.)**(-countSetBits(ii))
X = np.zeros((1,nu))
XX = np.zeros((1,nu))
XX_e = np.zeros((1,nu))
p2_e = 0
p2_wosn = 0

#def einsum_charecters(N):
alphabet = "abcdefghijklmnopqsrtuvwxyz"
alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ein_command = alphabet[0:N]
for ii in range(N):
ein_command += ','
ein_command += alphabet[ii]+alphabet_cap[ii]
ein_command += ','+ alphabet_cap[0:N]#[0:N]

hamming_array = np.array([[1,-0.5],[-0.5,1]])
Co = ''
for i in range(N):
Co += alphabet[i]
Co += alphabet_cap[i]
Co += ','
Co += alphabet_cap[:N]
Co += '->'
Co += alphabet[:N]
for iu in range(nu):
print(iu)

bit_strings = np.zeros(nm, dtype = 'int64')


xi_b = 1
psi_b = 2*math.pi
alpha_b = 2*math.pi
var_xi= xi_b*(theta_2) # parametrising phi and theta_2 to xi
var_theta_3 = theta_3*psi_b
# interval of integration for each angle
var_xi= xi_b*(theta_2) # parametrising phi and theta_2 to xi
X_e = np.zeros((1, nu))
n_qubits = N
bitcount = np.zeros((1,2**N))
for ii in range(2**N):
    bitcount[:,ii] = (-2.)**(-countSetBits(ii))

bit_strings = np.zeros((nu,nm), dtype = int)
partition_string_system = ['1'*x +'0'*(N-x) for x in range(1,N+1)]
partition_system = np.array([int(p,2) for p in partition_string_system])

#for iu in range(nu):
probb = np.zeros((1, 2**n_qubits))
uf = [1]
u_temp = 1j*(np.zeros((n_qubits,2,2)))

for qubits in range(n_qubits):
    alpha = 0
    psi = 0
    alpha = (-theta_3[qubits, iu])*alpha_b
    psi = (theta_3[qubits,iu])*psi_b
    u_temp[qubits,:,:] = (np.array([[math.sqrt(1-var_xi[qubits,iu])*cmath.exp(1j*alpha), math.sqrt(var_xi[qubits, iu])*cmath.exp(1j*psi)] , [-math.sqrt(var_xi[qubits, iu])*cmath.exp(-1j*psi), math.sqrt(1-var_xi[qubits, iu])*cmath.exp(-1j*alpha)]], dtype = complex))

    uf = (np.kron(uf,u_temp[qubits,:,:]))
#NN = np.dot(uf,np.dot(rho_rm,np.conj(np.transpose(uf))))
#LL = np.dot(np.conj(np.transpose(uf)),np.dot(rho_rm,uf))
#print(NN+LL)
probb = np.real(np.diag(uf.dot(rho_rm).dot(np.conj(np.transpose(uf)))))

#phi = 2*math.pi*phi
uff = [0]*N
for iq in range(N):
    uff[iq] = (RY(np.arcsin(np.sqrt(theta[iq,iu]))*2)*RZ(2*math.pi*phi[iq,iu])).full()
Listt = uff+[psi_tens]
psiu = np.einsum(Co,*Listt,optimize = 'greedy').flatten()
probb_test = np.abs(psiu)**2#*(1-p) + p/d
probb_test /= sum(probb_test)

#theta_2 = 2*math.pi*theta_2

u = [0]*N
for iq in range(N):
    u[iq] = RY(np.arcsin(np.sqrt(theta[iq, iu]))*2)*RZ(phi[iq, iu])
uf = tensor(u)
rho_u = uf*rho_rm*uf.dag()
probb = np.real(np.diag(rho_u.full()))
#print(probb)

prob_tens = np.zeros([2]*N) 
prob_tens = np.reshape(probb_test, [2]*N)
List = [prob_tens] + [hamming_array]*N + [prob_tens]
XX[0,iu] = np.einsum(ein_command, *List, optimize = True)*2**N

bit_strings = random_gen.choice(range(2**N), size = nm, p = probb_test) 
pseudo_string = np.append(bit_strings, [d-1]) ## added an extra measurement for (d-1)
probbe = np.bincount(pseudo_string)
probbe[-1] = probbe[-1] - 1 
probe_tens = np.reshape(probbe, [2]*N)
#hamming_array = np.array([[1,-0.5],[-0.5,1]])
Liste = [probe_tens] + [hamming_array]*N + [probe_tens]
XX_e[0,iu] = np.einsum(ein_command, *Liste, optimize = True)*2**N/(nm*(nm-1)) - 2**N/(nm-1)
p2_e += XX_e[0,iu]/nu
p2_wosn += XX[0,iu]/nu
Ep2_e += np.abs(p2_e -p2)/p2/nexp
Ep2_wosn += np.abs(p2_wosn -p2)/p2/nexp    


print(p2)
print(Ep2_e)
print(Ep2_wosn)
print(p2_e)
print(p2_wosn)
"""
"""
## dimension of the Hilbert space
d = 2**N
M = np.zeros((d, d))
M = np.dot(probb[:,None], probb[None,:])

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

    u = [0]*N
    for iq in range(N):
        u[iq] = RY(var_theta[iq, iu])*RZ(var_phi[iq, iu])
    uf = tensor(u)
    rho_u = uf*rho_rm*uf.dag()
    probb = np.real(np.diag(rho_u.full()))

xi_b = 1
psi_b = 2*math.pi
alpha_b = 2*math.pi
var_xi= xi_b*(theta_2) # parametrising phi and theta_2 to xi
var_theta_3 = theta_3*psi_b
# interval of integration for each angle
var_xi= xi_b*(theta_2) # parametrising phi and theta_2 to xi
X_e = np.zeros((1, nu))
n_qubits = N
bitcount = np.zeros((1,2**N))
for ii in range(2**N):
    bitcount[:,ii] = (-2.)**(-countSetBits(ii))

bit_strings = np.zeros((nu,nm), dtype = int)
partition_string_system = ['1'*x +'0'*(N-x) for x in range(1,N+1)]
partition_system = np.array([int(p,2) for p in partition_string_system])

#for iu in range(nu):
probb = np.zeros((1, 2**n_qubits))
uf = [1]
u_temp = 1j*(np.zeros((n_qubits,2,2)))

for qubits in range(n_qubits):
    alpha = 0
    psi = 0
    alpha = (-theta_3[qubits, iu])*alpha_b
    psi = (theta_3[qubits,iu])*psi_b
    u_temp[qubits,:,:] = (np.array([[math.sqrt(1-var_xi[qubits,iu])*cmath.exp(1j*alpha), math.sqrt(var_xi[qubits, iu])*cmath.exp(1j*psi)] , [-math.sqrt(var_xi[qubits, iu])*cmath.exp(-1j*psi), math.sqrt(1-var_xi[qubits, iu])*cmath.exp(-1j*alpha)]], dtype = complex))

    uf = (np.kron(uf,u_temp[qubits,:,:]))
#NN = np.dot(uf,np.dot(rho_rm,np.conj(np.transpose(uf))))
#LL = np.dot(np.conj(np.transpose(uf)),np.dot(rho_rm,uf))
#print(NN+LL)
probb = np.real(np.diag(uf.dot(rho_rm).dot(np.conj(np.transpose(uf)))))

"""
