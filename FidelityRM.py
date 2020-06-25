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

# Code to reconstruct the quantum state fidelity between two quantum devices (or one quantum device and a classical simulator)
# Based on https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.010504

import random
import numpy as np
from scipy import linalg

def SingleQubitRotation_CUE(random_gen): #Generate a 2x2 CUE matrix (ref: http://www.ams.org/notices/200705/fea-mezzadri-web.pdf)
    U = (random_gen.randn(2,2)+1j*random_gen.randn(2,2))/np.sqrt(2)
    q,r = linalg.qr(U)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    U = np.multiply(q,ph,q)
    return U

def get_Overlap(Meas_Data_1,Meas_Data_2,Partition_string,N,bias=True): #Function to get the overlap tr(rho_1 rho_2) of two quantum states based on randomized measurements
    Nu,NM = np.shape(Meas_Data_1)
    N_Partition = np.array([p.count('1') for p in Partition_string]) # Number of spins in each partition
    Partition = np.array([int(p,2) for p in Partition_string])
    nbp = len(Partition) #Number of partitions to analyze
# Convert each partition in 'qubit' representation
    Partition = np.unpackbits(np.array(Partition, dtype=">i2").view(np.uint8)).reshape((nbp,16))[:,-N:]
    Overlap = np.zeros(N)
    Hist_Hamming_Distances = np.zeros((N+1,nbp)) #Array use for storage

    path_info = np.einsum_path('ac,c->a',np.zeros((NM,N)), np.zeros(N),optimize='greedy')
#Process the result proced by each unitary
    for iu in range(Nu):
        #Compare each pair of bit string
        Pairs = np.bitwise_xor(Meas_Data_1[iu][:,None],Meas_Data_2[iu][None,:])
        # Calculate the Hamming distance for each qubit and each pair of bit string
        Hamming_Array_Full = np.unpackbits(np.array(Pairs, dtype=">i2").view(np.uint8)).reshape((NM**2,16))[:,-N:]
        # Loop over partitions to extract the relevant total Hamming distances
        for i in range(nbp):
            # Get the total Hamming distance
            Hamming_Distances = np.array(np.einsum('ac,c->a',Hamming_Array_Full,Partition[i,:],optimize=path_info[0]),dtype=int)
            # Count the number of each Hamming Distance when summing over each pair
            Counts = np.bincount(Hamming_Distances)
            Hist_Hamming_Distances[:len(Counts),i]  += Counts

#Extraction of the overlap based on averaging histograms of Hamming distances
    for i in range(nbp):
        #They key formula for randomized measurements
        Overlap[i] = 2**N_Partition[i]/Nu*sum(Hist_Hamming_Distances[:N_Partition[i]+1,i]*(-2.)**(-np.arange(N_Partition[i]+1)))
        #Remove statistical bias
        if bias:
            Overlap[i] = Overlap[i]/(NM*(NM-1))-2**N_Partition[i]/(NM-1)
        else:
            Overlap[i] = Overlap[i]/NM**2
    return Overlap

## Parameters
N = 8 # Number of qubits to analyze (necessary <= 16 for the present code due to the use of the function np.unpackbits)
if (N>16):
    print('Please reduce N (or adapt the call to np.unpackbits)')
Nu = 1000 # Number of random unitaries to be used
NM = 300 # Number of projective measurements (shots) per random unitary
Partition_string = ['1'*x +'0'*(N-x) for x in range(1,N+1)] ## List of partitions for which we want to extract the purity (ex: '100000..' only the first spin)
p_noise = 0.1#Noise strength related to the second device

### Initiate Random Generator
a = random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
random_gen = np.random.RandomState(a)

### Step 1:: Create a first quantum state on one device
## A GHZ state
rho_1 = np.zeros((2**N,2**N))
rho_1[0,0] = 1./2
rho_1[-1,-1] = 1./2
rho_1[-1,0] = 1./2
rho_1[0,-1] = 1./2

###Create the same quantum state on a second device
## and simulate additional noise on this device by adding depolarization
rho_2 = (1.-p_noise)*rho_1 + p_noise/2**N*np.eye(2**N)

### Step 2:: Perform randomized measurements on both devices
path_info = np.einsum_path('ab,bc,ac->a',rho_1,rho_1,rho_1,optimize='greedy')
Meas_Data_1 = np.zeros((Nu,NM),dtype=int)  # Meas_data_1 will store the measured bit strings (NM per random unitary) on the first device
Meas_Data_2 = np.zeros((Nu,NM),dtype=int)  # second device
for iu in range(Nu):
        #Construct random unitary as a product of single qubit rotations
        Unitary = SingleQubitRotation_CUE(random_gen)
        for i in range(1,N):
            Unitary = np.kron(Unitary,SingleQubitRotation_CUE(random_gen))

        #Apply on state and compute bitstrings probabilities
        Prob_bitstrings_1 = np.real(np.einsum('ab,bc,ac->a',Unitary,rho_1,np.conj(Unitary),optimize=path_info[0]))
        Prob_bitstrings_1 /= sum(Prob_bitstrings_1)

        Prob_bitstrings_2 = np.real(np.einsum('ab,bc,ac->a',Unitary,rho_2,np.conj(Unitary),optimize=path_info[0]))
        Prob_bitstrings_2 /= sum(Prob_bitstrings_2)

        #Sample NM measurements according to the bitstrings probabilities
        Meas_Data_1[iu,:] = random_gen.choice(range(2**N),p=Prob_bitstrings_1,size=NM)
        Meas_Data_2[iu,:] = random_gen.choice(range(2**N),p=Prob_bitstrings_2,size=NM)
print('Measurement data generated')

### Step 3:: Reconstruct fidelities from measured bitstrings
Purity1 = get_Overlap(Meas_Data_1,Meas_Data_1,Partition_string,N,bias=True)
Purity2 = get_Overlap(Meas_Data_2,Meas_Data_2,Partition_string,N,bias=True)
Overlap = get_Overlap(Meas_Data_1,Meas_Data_2,Partition_string,N,bias=False)
Fidelity = Overlap/np.maximum(Purity1,Purity2)
Fidelity = 1*(Fidelity>1) + Fidelity*(Fidelity<=1) #Remove potential unphysical value due to shot noise

print('Reconstructed Purities')
for i,p in enumerate(Partition_string):
    print('Partition ',p, ":", Fidelity[i])
print('Estimated depolarization noise ',1 - Fidelity[-1])
