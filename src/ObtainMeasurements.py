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

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

def SingleQubitRotation(random_gen,mode):
    U = 1j*np.zeros((2,2))
    if mode=='CUE':
        #Generate a 2x2 CUE matrix (ref: http://www.ams.org/notices/200705/fea-mezzadri-web.pdf)
        U = (random_gen.randn(2,2)+1j*random_gen.randn(2,2))/np.sqrt(2)
        q,r = linalg.qr(U)
        d = np.diagonal(r)
        ph = d/np.absolute(d)
        U = np.multiply(q,ph,q)
    else:
        ## Equivalently we can measure randomly in the x,y, or z basis:
        pick = random_gen.randint(3)
        if pick==0: #Measurement setting y: we apply pi/4 along x before measurement
            U[0,0] = 1./np.sqrt(2)
            U[1,1] = 1./np.sqrt(2)
            U[0,1] = -1j*1./np.sqrt(2)
            U[1,0] = -1j*1./np.sqrt(2)
        elif pick==1: #Measurement setting x: we apply pi/4 along y before measurement
            U[0,0] = 1./np.sqrt(2)
            U[1,1] = 1./np.sqrt(2)
            U[0,1] = -1./np.sqrt(2)
            U[1,0] = 1./np.sqrt(2)
        else:
            U = np.eye(2)
    return U

def Simulate_Meas_pseudopure(NN, psi, p, u):
    alphabet = "abcdefghijklmnopqsrtuvwxyz"
    alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    d = 2**NN
    Co = ''
    for i in range(NN):
        Co += alphabet[i]
        Co += alphabet_cap[i]
        Co += ','
    Co += alphabet_cap[:NN]
    Co += '->'
    Co += alphabet[:NN]
    #print(iu)
    Listt = u+[psi]
    psiu = np.einsum(Co,*Listt,optimize = True).flatten()
    probb = np.abs(psiu)**2*(1-p) + p/d ## makes the probabilities noisy by adding white noise
    probb /= sum(probb)
    #probb = np.reshape(probb, [2]*NN)
    #bit_strings = random_gen.choice(range(2**NN), size = NM, p = probb) 
    return probb

def Simulate_Meas_mixed(NN, rho,  u):
        #Unitary = u[0]
        #for i in range(1,NN):
        #    Unitary = np.kron(Unitary,u[i])
        #Prob = np.real(np.einsum('ab,bc,ac->a',Unitary,rho,np.conj(Unitary),optimize='greedy'))
        prob_tensor = rho.reshape(tuple([2] * (2*NN)),order='C')
        for n in range(NN):
            prob_tensor = np.einsum(u[n], [2*NN, n], prob_tensor, list(range(NN))+list(range(n+NN,2*NN)), np.conjugate(u[n]), [2*NN,NN+n], list(range(n)) + [2*NN] + list(range(n + 1, NN)) + list(range(NN+n+1,2*NN)))
        probb= np.real(prob_tensor.reshape(2**NN))
        probb /= sum(probb)
        #Prob /= sum(Prob)
        #Sample NM measurements according to the bitstrings probabilities
        #bit_strings = random_gen.choice(range(2**N),p=Prob,size=NM)
        return probb

def Sampling_Meas(prob,N,NM):
        return random_gen.choice(range(2**N),p=prob,size=NM)