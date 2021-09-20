#   Copyright 2020 Benoit Vermersch, Andreas Elben
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

## Code to reconstruct the purity of artifically generated randomized measurement data of a N qubit system
## Protocol: Elben, Vermersch, et al: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.050406
##           Apply Nu random unitaries made of single qubit rotations, measure for each unitary Nm bitstrings k_{s=1,..,NM}
## Estimator used in the present code: https://science.sciencemag.org/content/364/6437/260 and https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.052323
##           purity = Average(sum_{s,s'} 2^(Number of qubits)(-2)^(-Hamming_Distance[s,s']) ) ( - statistical bias )

import random
import numpy as np
from scipy import linalg
#from qutip import *

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

## Parameters
N = 3 # Number of qubits to analyze (necessary <= 16 for the present code due to the use of the function np.unpackbits)
if (N>16):
    print('Please reduce N (or adapt the call to np.unpackbits)')
Nu = 2000 # Number of random unitaries to be used
NM = 200 # Number of projective measurements (shots) per random unitary
mode = 'xyz'
Partition_string = ['1'*x +'0'*(N-x) for x in range(1,N+1)] ## List of partitions for which we want to extract the purity (ex: '100000..' only the first spin)
Partition_string = [ '0'*x+'1'+'0'*(N-x-1) for x in range(N)]
Partition_string += [ '1'*N ]

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Step 1:: Create a quantum state
## A GHZ state
rho = np.zeros((2**N,2**N))
rho[0,0] = 1.
#rho[-1,-1] = 1./2
#rho[-1,0] = 1./2
#rho[0,-1] = 1./2
### Or a qutip random density matrix combined with a pure product state qubit
import qutip
rho = qutip.tensor(qutip.rand_dm(2**(N-1),pure=True),qutip.basis(2,0)*qutip.basis(2,0).dag()).data.todense()

## Optional: calculate exact purities with Qutip partial trace function
rhoQ = qutip.Qobj(rho,dims=[ [2]*N,[2]*N] )
print('Exact Purities')
for p in Partition_string:
    if p=='1'*N:
        print('Partition ',p, ":", (rhoQ*rhoQ).tr())
    else:
        rhop = rhoQ.ptrace([i for i in range(N) if p[i]=='1'])
        print('Partition ',p, ":", (rhop*rhop).tr())

### Step 2:: Perform randomized measurements
path_info = np.einsum_path('ab,bc,ac->a',rho,rho,rho,optimize='greedy')
Meas_Data = np.zeros((Nu,NM),dtype=int)  # Meas_data will store the measured bit strings (NM per random unitary)
for iu in range(Nu):
        #Construct random unitary as a product of single qubit rotations
        Unitary = SingleQubitRotation(random_gen,mode)
        for i in range(1,N):
            Unitary = np.kron(Unitary,SingleQubitRotation(random_gen,mode))

        #Apply on state and compute bitstrings probabilities
        Prob_bitstrings = np.real(np.einsum('ab,bc,ac->a',Unitary,rho,np.conj(Unitary),optimize=path_info[0]))
        Prob_bitstrings /= sum(Prob_bitstrings)

        #Sample NM measurements according to the bitstrings probabilities
        Meas_Data[iu,:] = random_gen.choice(range(2**N),p=Prob_bitstrings,size=NM)
print('Measurement data generated')

### Step 3:: Reconstruct purities from measured bitstrings
N_Partition = np.array([p.count('1') for p in Partition_string]) # Number of spins in each partition
Partition = np.array([int(p,2) for p in Partition_string])
nbp = len(Partition) #Number of partitions to analyze
# Convert each partition in 'qubit' representation
Partition = np.unpackbits(np.array(Partition, dtype=">i2").view(np.uint8)).reshape((nbp,16))[:,-N:]
Purity = np.zeros(nbp)
Hist_Hamming_Distances = np.zeros((N+1,nbp)) #Array use for storage

path_info = np.einsum_path('ac,c->a',np.zeros((NM,N)), np.zeros(N),optimize='greedy')
#Process the result proced by each unitary
for iu in range(Nu):
    #Compare each pair of bit string
    Pairs = np.bitwise_xor(Meas_Data[iu][:,None],Meas_Data[iu][None,:])
    # Calculate the Hamming distance for each qubit and each pair of bit string
    Hamming_Array_Full = np.unpackbits(np.array(Pairs, dtype=">i2").view(np.uint8)).reshape((NM**2,16))[:,-N:]
    # Loop over partitions to extract the relevant total Hamming distances
    for i in range(nbp):
        # Get the total Hamming distance
        Hamming_Distances = np.array(np.einsum('ac,c->a',Hamming_Array_Full,Partition[i,:],optimize=path_info[0]),dtype=int)
        # Count the number of each Hamming Distance when summing over each pair
        Counts = np.bincount(Hamming_Distances)
        Hist_Hamming_Distances[:len(Counts),i]  += Counts

#Extraction of the purity based on averaging histograms of Hamming distances
for i in range(nbp):
    #They key formula for randomized measurements
    Purity[i] = 2**N_Partition[i]/Nu*sum(Hist_Hamming_Distances[:N_Partition[i]+1,i]*(-2.)**(-np.arange(N_Partition[i]+1)))
    #Remove statistical bias
    Purity[i] = Purity[i]/(NM*(NM-1))-2**N_Partition[i]/(NM-1)

print('Reconstructed Purities')
for i,p in enumerate(Partition_string):
    print('Partition ',p, ":", Purity[i], '2nd Renyi Entropy : ',-np.log2(Purity[i]))
