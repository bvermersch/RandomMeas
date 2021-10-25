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
### step 2: Sampling of the unitaries from the ideal state 
## initialize randomized measurment parameters
nu = 1 # number of unitaries to be used 
nm = 2000# number of readout measurements per unitary
nexp = 1
Ep2_e = 0
Ep2_wosn = 0
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
        
        """
        bit_strings = np.zeros(nm)
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
        phi = 2*math.pi*phi
        uff = [0]*N
        for iq in range(N):
            uff[iq] = (RY(np.arcsin(np.sqrt(theta[iq,iu]))*2)*RZ(phi[iq,iu])).full()
        Listt = uff+[psi_tens]
        psiu = np.einsum(Co,*Listt).flatten()
        probb_test = np.abs(psiu)**2*(1-p) + p/d
        """
        #theta_2 = 2*math.pi*theta_2
        
        u = [0]*N
        for iq in range(N):
            u[iq] = RY(np.arcsin(np.sqrt(theta[iq, iu]))*2)*RZ(phi[iq, iu])
        uf = tensor(u)
        rho_u = uf*rho_rm*uf.dag()
        probb = np.real(np.diag(rho_u.full()))
        #print(probb)
        """
        prob_tens = np.zeros([2]*N) 
        prob_tens = np.reshape(probb_test, [2]*N)
        List = [prob_tens] + [hamming_array]*N + [prob_tens]
        XX[0,iu] = np.einsum(ein_command, *List)*2**N
        
        bit_strings = random_gen.choice(range(2**N), size = nm, p = probb_test) 
        pseudo_string = np.append(bit_strings, [d-1]) ## added an extra measurement for (d-1)
        probbe = np.bincount(pseudo_string)
        probbe[-1] = probbe[-1] - 1 
        probe_tens = np.reshape(probbe, [2]*N)
        #hamming_array = np.array([[1,-0.5],[-0.5,1]])
        Liste = [probe_tens] + [hamming_array]*N + [probe_tens]
        XX_e[0,iu] = np.einsum(ein_command, *Liste)*2**N/(nm*(nm-1)) - 2**N/(nm-1)
        p2_e += XX_e[0,iu]/nu
        p2_wosn += XX[0,iu]/nu
    Ep2_e += np.abs(p2_e -p2)/p2/nexp
    Ep2_wosn += np.abs(p2_wosn -p2)/p2/nexp    
        

print(p2)
print(Ep2_e)
print(Ep2_wosn)
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
"""    

"""
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
