# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:14:04 2020

@author: tahawaru
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:36:39 2020

@author: tahawaru
"""
#import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import math


#Computing spectral efficiency with 3D MIMO channel
Nt=16 # Number of transmit antenna ports
Nr=16  # Number of receive antenna ports
F=3.5e9 # Carrier Frequency is 3.5 GHz(CBRS) 6GHz
W=150e3  # Bandwidth is 150 KHz
Lambda=3e8/F # Wavelength
K=2*np.pi/Lambda  # Wave number


P_BS =30 #(W)  Max transmit power
m=100
Pt= np.linspace( (P_BS)/m, (P_BS), m)
N=6  # Number of paths in a multipath propagation

Theta=np.pi/np.linspace(20,50,N) 
Phi=np.pi/np.linspace(20,70,N) 

VarTheta=np.pi/np.linspace(20,70,N) 
VarPhi=np.pi/np.linspace(20,80,N)

Tilt=np.pi/np.linspace(20,80,Nt) 
dr=Lambda/2  # separation between antenna ports at the receiver
dt=Lambda/2 # separation between antenna ports at the transmitter
# function transmit array 
def at(s,l):  # l: multipath index, s= antenna node
    return np.exp(-1j*(s-1)*dt*np.sin(Theta[l])*np.sin(Phi[l]))
# function receive array 
def ar(r,l):  # l: multipath index, r= antenna node
    return np.exp(-1j*(r-1)*dr*np.sin(VarTheta[l])*np.sin(VarPhi[l]))

def Ah(phi):
    return -min(12*(phi/(70*np.pi/180))**2,20)  # in dB unit

def Av(Ttheta,Ttilt):
    return -min(12*((Ttheta-Ttilt)/(15*np.pi/180))**2,20)  # in dB unit


def Sqrg(s,l):
    return np.sqrt(np.power(10,(17-min(-(Ah(Phi[l])+Av(Theta[l],Tilt[s])),20))/10))
     
#u=at(2,3)
#v=ar(1,3)
#rs=Sqrg(2,3)

# Average pathloss
n_pathloss=3 # Pathloss constant
Tx_Rx_dist=0.02 # Distance between the pair of equipment forming the D2D
def ro(dist):
    return ((Lambda/(4*np.pi*5))**2)*np.power((5/dist),n_pathloss) # Pathloss
#Pt_avg=np.power(10,0.7); # Average total transmitted power
#Pr_avg=Pt_avg*ro(Tx_Rx_dist); # Average received power
 
#Generating the alpha vector  
def iid():
    U1=np.random.rand(1)
    U2=np.random.rand(1)
    X=np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
    Y=np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)
    return X+Y*1j  

def MIMO3Dchannel(nr,nt,n,dist): # nr x nt MIMO channel with n paths for multipath propagation
    # Deterministic matrix A of dimension Nt x N
    # A is a schur(Hadamard) product of A1 and A2 mutiplied by a factor
    A1=np.zeros((nt,n), dtype=complex)
    A2=np.zeros((nt,n), dtype=complex)
    for i in range(nt):
        for k in range(n):
            A1[i,k]=A1[i,k] + at(i,k)
            A2[i,k]=A2[i,k] + Sqrg(i,k)
    A=np.multiply(A1,A2)

    # Deterministic matrix B of dimension Nr x N
    B=np.zeros((nr,n), dtype=complex)
    for i in range(nr):
        for k in range(n):
            B[i,k]=B[i,k]+ar(i,k)
    Alpha=np.zeros((n,n), dtype= complex)
    for i in range(n):
        Alpha[i,i]=np.sqrt(nr*nt/ro(dist))*Alpha[i,i]+ iid()
        
    G=np.zeros((nr,nt), dtype=complex)  # MIMO 3D channel
    G=np.sqrt(1/n)*np.matmul(np.matmul(B,Alpha),A.conj().T)
    return G

#H=MIMO3Dchannel(Nr,Nt,n,Tx_Rx_dist)

def capa(nr,nt,n,d,Ptx):
    #Omega1 = 1;  #Average channel gain between transmitter 1 and relay
    # Interference----
    NumInt=3 #Number of interfering users
    #Hinter=ChanMem(NumInt); % Generate the MIMO channel for each of the Numinter interferers 
    epsd=0.001
    #d=0.02;
    dInter=0.195
    PL=ro(d + epsd)
    PLInter=ro(dInter + epsd)
    noiseFigure = np.power(10,0.2) # (dB) Thermal Noise Density 
    N0 =  np.power(10,(-20.4)) #W/Hz = dBm/Hz 
    Qinter=[]
    for i in range(NumInt):
        v=np.diag((Ptx/nt)*np.ones(nt, dtype=complex))
        Qinter.append(v)
    
    InterP=np.zeros((nr,nr), dtype=complex) #sum(HQH')  
    for jj in range (NumInt):
       Hint=MIMO3Dchannel(nr,nt,n,(dInter + epsd))
       Qint=Qinter[jj]
       InterP=InterP+np.matmul((1/PLInter)*np.matmul(Hint,Qint),Hint.conj().T)
    
      #---------
    No=W * noiseFigure * N0
    myu =np.array([0.08, 0.12, 0.8]) #Probability of the realization of each channel; 
    tempSum=0 
    Q=np.diag((Ptx/nt)*np.ones(nt, dtype=complex))
    H=[]
    for ii in range(myu.shape[0]):
        H.append(MIMO3Dchannel(nr,nt,n,(d + epsd)))
        Inr=np.diag(np.ones(nr,dtype=complex))
        tempSum=tempSum + myu[ii]*np.log2(np.linalg.det(No*Inr+InterP+np.matmul((1/PL)*np.matmul(H[ii],Q),H[ii].conj().T)))
    
      
    Capacity=tempSum - np.log2(np.linalg.det(No*np.diag(np.ones(nr, dtype=complex))+InterP))
    return abs(Capacity)
# Generating the AP coefficient


n_sum=1/np.linspace(2,Nt+1,Nt+1) # Generation of n_sum components
#Spectral efficiency and Energy Efficiency of 3D channel
Pcirc=5 # circuitry power consumption
ITER=20 #Pt.shape[0]
SE=np.zeros(ITER, dtype=float) # (Compute spectral efficiency )Achievable rate with perfect channel knowledge and computed Q
EEopt=np.zeros(ITER, dtype=float)
EEopt2=np.zeros(ITER, dtype=float)
EE=np.zeros(ITER, dtype=float)# (Compute Energy efficiency from spectral efficiency ) 
TPC=np.zeros(ITER, dtype=float)# Total power consumption
NumChan=1000   # Wireless channel iteration number
Pa=(Pcirc*Nt**2)/P_BS  # amplifier power
mu=0
Ntparam=[]
distD2D=1/np.linspace(5,2000,ITER)
for l in range(Nt):
     mu=mu + 1/n_sum[l]
     
for iter in range(ITER):
    sum_n=int(math.modf(Pcirc*(Nt**2)/Pt[iter])[1])
    Ntparam.append(np.sqrt(mu*Pt[iter]/Pcirc))    
    for chan in range(NumChan):
        SE[iter]=SE[iter] + capa(Nr,Nt,N,distD2D[iter],P_BS)
        
    SE[iter]=SE[iter]/NumChan
    
    EE[iter]=W*SE[iter]/(2*(1e6)*(mu*P_BS/Nt + Nt*Pcirc)) # Non optimal Energy efficiency    
    #Optimal energy efficiency; # dividing by 1e6 to present EE in Mbps/Joule 
    EEopt2[iter]=np.sqrt(Pa*P_BS/Pcirc)*W*SE[iter]/(4*(1e6)*(P_BS*Pa))
#Optimality 
#Aval=Pcirc*n_sum
#Bval=Pt/(Nt**2)


plt.figure(1)
plt.xlabel('Normalized distance (d/do)')
plt.ylabel('SE (bps/Hz)')
 
plt.plot(distD2D,SE, marker='^',label="Optimal SE_j, n_t=16, n_r=16")
plt.savefig('SEopt.png')
plt.show()


plt.figure(2)
# Add title and axis names
#plt.title('My title')
plt.xlabel('SE (bits/s/Hz)')
plt.ylabel('EE (Mbps/Joule)')
 

plt.plot(SE,EE, marker='*',label="Non-Optimal EE_j, n_t=16, n_r=16")
plt.plot(SE,EEopt2, marker='+',label="Optimal EE_j, n_t=16, n_r=16")
plt.legend(loc='upper left')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('EEoptNt2.png')
plt.show()