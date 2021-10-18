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

    
def get_prob(meas_data,NN):
    NM = len(meas_data)
    prob = np.bincount(meas_data,minlength=2**NN)/NM
    prob = np.reshape(prob, [2]*NN)
    return prob

def reduce_prob(prob,NN,tracedsystem):
   return np.sum(prob,tuple(tracedsystem))
     
def get_X(prob, NN):
    alphabet = "abcdefghijklmnopqsrtuvwxyz"
    alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    Hamming_matrix = np.array([[1,-0.5],[-0.5,1]]) ## Hamming matrix for a single qubit
    ein_command = alphabet[0:NN]
    for ii in range(NN):
        ein_command += ','
        ein_command += alphabet[ii]+alphabet_cap[ii]
    ein_command += ','+ alphabet_cap[0:NN]
    Liste = [prob] + [Hamming_matrix]*NN + [prob]
    XX_e = np.einsum(ein_command, *Liste, optimize = True)*2**NN
    return XX_e
    
def unbias(X,NN,NM):
    return X*NM**2/(NM*(NM-1)) - 2**NN/(NM-1)

def get_X_overlap(prob1,prob2, NN,NM):
    alphabet = "abcdefghijklmnopqsrtuvwxyz"
    alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    Hamming_matrix = np.array([[1,-0.5],[-0.5,1]]) ## Hamming matrix for a single qubit
    ein_command = alphabet[0:NN]
    for ii in range(NN):
        ein_command += ','
        ein_command += alphabet[ii]+alphabet_cap[ii]
    ein_command += ','+ alphabet_cap[0:NN]
    Liste = [prob1] + [Hamming_matrix]*NN + [prob2]
    XX_e = np.einsum(ein_command, *Liste, optimize = True)*2**NN
    return XX_e
