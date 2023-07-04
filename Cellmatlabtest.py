# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 00:28:45 2020

@author: tahawaru
"""

import numpy as np
import matplotlib.pyplot as plt

def mat(nr,nt):
    return np.zeros((nr,nt), dtype=complex)
    
# A=[]
# Pt=30
# N=4
# Nrow=3
# Ncol=4
# for i in range(N):
#     v=np.array(mat(Nrow,Ncol))
#     for i in range(Nrow):
#         for j in range(Ncol):
#             if (i==j):
#                 v[i,j]=Pt/Ncol + np.random.rand(1) 
    
#     A.append(v)

# vdiag=np.diag((Pt/N)*np.ones(N,dtype=complex))

a=2
b=4
f=6
if((a==b) and (b<f)):
    print(f)
else:
    print("no")

Npjmax=10000000 #np.linspace(2, 100, 100)
FixVal=1000
Nnt=5 #np.linspace(2, 100, 100)
Po=np.linspace(0.005,0.95, 20)


pout1=1 + Po*(-1 + 1/Npjmax + 1/FixVal)
pout2=1 + Po*(-1 + 1/FixVal + 1/Nnt)

plt.figure(1)
plt.xlabel('SE probability of outage')
plt.ylabel('Markov State Transition Probability')
 
plt.plot(Po,pout1, marker='^',label="Pout, Nnt=1000")
plt.plot(Po,pout2, marker='+',label="Pout, Npjmax=1000")
plt.legend(loc='upper right')
plt.savefig('D2DTransProb.png')
plt.show()
