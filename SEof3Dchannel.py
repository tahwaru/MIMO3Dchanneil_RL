# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:22:14 2020

@author: tahawaru
"""
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import math


#Spectral efficiency with 3-D channel model

tilt=np.array([np.pi/4, np.pi/6, np.pi/12])

######3D channel H
#------------------------System Parameters---------------------------------
Num_BS_Antennas=4; # BS antennas
BSAntennas_Index=np.linspace(0,Num_BS_Antennas-1); # Indices of the BS Antennas


Num_MS_Antennas=2; # MS antennas
MSAntennas_Index=np.linspace(0,Num_MS_Antennas-1); # Indices of the MS Antennas



# ---------------------Channel Parameters ---------------------------------
Num_paths=3; # Number of channel paths


# Channel Generation 

# Channel parameters (angles of arrival and departure and path gains)
AoD=2*np.pi*np.random.rand(1,Num_paths);
AoA=2*np.pi*np.random.rand(1,Num_paths);
alpha=(np.sqrt(1/2)*np.sqrt(1/Num_paths)*(np.random.randint(1,size=Num_paths)+1j*np.random.randint(1,size=Num_paths)));

Abh=np.zeros((Num_paths,Num_BS_Antennas), dtype=complex)
Amh=np.zeros((Num_paths,Num_MS_Antennas), dtype=complex)
# Channel construction
Channel=np.zeros((Num_MS_Antennas,Num_BS_Antennas), dtype=complex);
for l in range(Num_paths):
    for k in range(Num_BS_Antennas):
        Abh[l,k]=np.sqrt(1/Num_BS_Antennas)*np.exp(1j*BSAntennas_Index[k]*AoD[l]);
    for i in range(Num_MS_Antennas):
        Amh[l,i]=np.sqrt(1/Num_MS_Antennas)*np.exp(1j*MSAntennas_Index[i]*AoA[l]);
    Channel=Channel+np.sqrt(Num_BS_Antennas*Num_MS_Antennas)*alpha[l]*Amh[l]*np.conj(Abh[l]);
