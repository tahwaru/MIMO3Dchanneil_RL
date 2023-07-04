# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:11:59 2020

@author: tahawaru
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:40:19 2020

@author: tahawaru
"""

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import math


W=1.5e3 #bandwidth considering narrow band
Pjmax=60 # D2D link transmit maximum power
Pcirc=5 # circuitry power consumption
#nt=[8,16,32,64,128] # Number of transmit nodes
nt=64
sum_n=int(math.modf(Pcirc*(nt**2)/Pjmax)[1])
x = np.arange(1, sum_n)
xU, xL = x + 0.5, x - 0.5 


N=100000
Not_AvSE=np.zeros((N,sum_n))
Not_AvEE=np.zeros((N,sum_n))
# Generating random number for the SE values
rng = default_rng()
vals = rng.standard_normal(sum_n)
for i in range(N):
    prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
    prob = prob / prob.sum() #normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size = sum_n, p = prob)
    n_sum=nums/np.sum(nums)
    factor=0.15  # =0.5 for n_t=16, Pcirc=5
    Not_AvSE[i]= Not_AvSE[i] + 0.05*(np.linspace(1,sum_n,sum_n) + rng.standard_normal(sum_n))  
    Not_AvEE[i]=W*Not_AvSE[i]/(2*(1e6)*(Pjmax/(nt*n_sum)+nt*Pcirc)) # Non optimal Energy efficiency
    
#Optimality 
A=Pcirc*n_sum;
B=Pjmax/(nt**2)

SE=np.sum(Not_AvSE,axis=0)/N
# dividing by 1e6 to present EE in Mbps/Joule  

EE=np.sum(Not_AvEE,axis=0)/N
EEopt=W*SE/(4*(1e6)*nt*Pcirc) #Optimal energy efficiency

# Add title and axis names
#plt.title('My title')
plt.xlabel('SE (bits/s/Hz)')
plt.ylabel('EE (Mbps/Joule)')
 

plt.plot(SE,EE, marker='*',label="Non-Optimal EE_j, n_t=64")
plt.plot(SE,EEopt, marker='+',label="Optimal EE_j, n_t=64")
plt.legend(loc='upper left')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('EEoptNt.png')
plt.show()
