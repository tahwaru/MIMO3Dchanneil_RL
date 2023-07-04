# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:25:56 2020

@author: tahawaru
"""
import scipy.stats as ss
import numpy as np
#import matplotlib.pyplot as plt
import math

Pjmax=30 # D2D link transmit maximum power
Pcirc=9 # circuitry power consumption
nt=10
sum_n=int(math.modf(Pcirc*(nt**2)/Pjmax)[1])
x = np.arange(1, sum_n)
xU, xL = x + 0.5, x - 0.5 
prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
prob = prob / prob.sum() #normalize the probabilities so their sum is 1
nums = np.random.choice(x, size = sum_n, p = prob)
n_sum=nums/np.sum(nums)
#plt.hist(nums, bins = len(x))
N=5
s=np.zeros((N,sum_n))
for k in range(N):
    s[k]=n_sum
    
Rsum=np.sum(s,axis=0)