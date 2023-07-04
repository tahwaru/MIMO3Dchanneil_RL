# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:36:39 2020

@author: tahawaru
"""
#import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
#import math


#Computing spectral efficiency with 3D MIMO channel
#Nt=16 # Number of transmit antenna ports
Nr=16  # Number of receive antenna ports
F=3.5e9 # Carrier Frequency is 3.5 GHz(CBRS) 6GHz
W=150e3  # Bandwidth is 150 KHz
Lambda=3e8/F # Wavelength
K=2*np.pi/Lambda  # Wave number


N=20  # Number of paths in a multipath propagation

Theta=np.pi/np.linspace(20,50,N) 
Phi=np.pi/np.linspace(20,70,N) 

VarTheta=np.pi/np.linspace(20,70,N) 
VarPhi=np.pi/np.linspace(20,80,N)

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


def Sqrg(s,l, tilt):
    return np.sqrt(np.power(10,(17-min(-(Ah(Phi[l])+Av(Theta[l],tilt)),20))/10))
     
#u=at(2,3)
#v=ar(1,3)
#rs=Sqrg(2,3)

# Average pathloss
n_pathloss=3 # Pathloss constant
Tx_Rx_dist=0.06 # Distance between the pair of equipment forming the D2D
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

def MIMO3Dchannel(nr,nt,n,dist,tilt): # nr x nt MIMO channel with n paths for multipath propagation
    # Deterministic matrix A of dimension Nt x N
    # A is a schur(Hadamard) product of A1 and A2 mutiplied by a factor
    A1=np.zeros((nt,n), dtype=complex)
    A2=np.zeros((nt,n), dtype=complex)
    for i in range(nt):
        for k in range(n):
            A1[i,k]=A1[i,k] + at(i,k)
            A2[i,k]=A2[i,k] + Sqrg(i,k,tilt)
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

def capa(nr,nt,n,d,Ptx, tilt):
    #Omega1 = 1;  #Average channel gain between transmitter 1 and relay
    # Interference----
    NumInt=3 #Number of interfering users
    #Hinter=ChanMem(NumInt); % Generate the MIMO channel for each of the Numinter interferers 
    epsd=0.001
    #d=0.02;
    dInter=0.290
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
       Hint=MIMO3Dchannel(nr,nt,n,(dInter + epsd),tilt)
       Qint=Qinter[jj]
       InterP=InterP+np.matmul((PLInter)*np.matmul(Hint,Qint),Hint.conj().T)
    
      #---------
    No=W * noiseFigure * N0
    myu =np.array([0.08, 0.12, 0.8]) #Probability of the realization of each channel; 
    tempSum=0 
    Q=np.diag((Ptx/nt)*np.ones(nt, dtype=complex))
    H=[]
    for ii in range(myu.shape[0]):
        H.append(MIMO3Dchannel(nr,nt,n,(d + epsd),tilt))
        Inr=np.diag(np.ones(nr,dtype=complex))
        tempSum=tempSum + myu[ii]*np.log2(np.linalg.det(No*Inr+InterP+np.matmul((PL)*np.matmul(H[ii],Q),H[ii].conj().T)))
    
      
    Capacity=tempSum - np.log2(np.linalg.det(No*np.diag(np.ones(nr, dtype=complex))+InterP))
    return abs(Capacity)

Nlink=10  # number of D2D links
Ptot=30
Ncg=15
#RL algorithm
#def ReinLearn(Pjmax, tilt, nt, nr, n, d, SEtilde): # SE for optimal Pjmax, tilt, nt
   
            
ProbSE=0.8  # probability for the SE in the reward function
ProbEE=0.2# probability for the EE in the reward function

distD2D=0.25   
   
PTx=np.linspace(20,30,10)  # Tx power is 20 Watt 
Npjmax=PTx.shape[0]
Pcirc=5 # circuitry power consumption
nt=[2**n for n in range(3,7)] # Nt=8,16,32,64
Nnt=len(nt)
Tilt=np.pi/np.linspace(10,80,20)

Po=np.exp(-3*np.random.rand(Nlink))
SEtilde=2  # SE threshold for outage
SE_RL=np.zeros((Nlink,Ncg), dtype=float)  # SE for all D2D links
SE=np.zeros((Nlink,Ncg), dtype=float)  # SE for all D2D links

EE_RL=np.zeros((Nlink,Ncg), dtype=float)  # EE for all D2D links
EE=np.zeros((Nlink,Ncg), dtype=float)  # EE for all D2D links
R1_D2D=np.zeros((Ncg), dtype=float)  # Centralized reward function
R2_D2D=np.zeros((Ncg), dtype=float)  # Centralized reward function
R3_D2D=np.zeros((Ncg), dtype=float)  # Centralized reward function
CumR1_D2D=np.zeros((Ncg), dtype=float)  # Centralized reward function
CumR2_D2D=np.zeros((Ncg), dtype=float)  # Centralized reward function
CumR3_D2D=np.zeros((Ncg), dtype=float)  # Centralized reward function

TPC=np.zeros((Nlink,Ncg), dtype=float)
TPC_RL=np.zeros((Nlink,Ncg), dtype=float)

R=np.zeros((Ncg), dtype=float)
EEug=np.zeros((Ncg), dtype=float)
EEugRL=np.zeros((Ncg), dtype=float)
#initial state
Pjmax=Ptot/Nlink
IndexS=[4,5,3] # Index of the state Ptx, Tilt and nt values at t, should be smaller than their cardinalities

#Initialization for all links
for j in range(Nlink):
    print("Initial state")
    print("nt=",nt[3])
    print("Pt=",PTx[4])
    print("tilt=",Tilt[5])
    print("        ")
    SE[j,0]=capa(Nr,nt[IndexS[2]],N,distD2D,PTx[IndexS[0]], Tilt[IndexS[1]])
    SE_RL[j,0]=SE[j,0]
    n_sum=1/np.linspace(2,nt[IndexS[2]]+1,nt[IndexS[2]]+1) # Generation of n_sum components
    mu=0
    for l in range(nt[IndexS[2]]):
        mu=mu + 1/n_sum[l]
        
    EE[j,0]=W*SE[j,0]/(2*(1e6)*(mu*PTx[IndexS[0]]/nt[IndexS[2]] + nt[IndexS[2]]*Pcirc))
    EE_RL[j,0]=EE[j,0]
    TPC[j,0]=mu*PTx[IndexS[0]]/nt[IndexS[2]]+nt[IndexS[2]]*Pcirc
    TPC_RL[j,0]=TPC[j,0]
OutageIndic=0  # Indicator of SE outage
TiltSE=np.zeros(Tilt.shape[0])
for t in range(1,Ncg):    
    for j in range(Nlink): # We can average over the number of Ncg
         pout=1 + Po[j]*(-1 + 1/Npjmax + 1/Nnt) #Compute the probability of transition 
         SE[j,t]=capa(Nr,nt[IndexS[2]],N,distD2D,PTx[IndexS[0]],Tilt[IndexS[1]])
         n_sum=1/np.linspace(2,nt[IndexS[2]]+1,nt[IndexS[2]]+1)
         mu=0
         for l in range(nt[IndexS[2]]):
             mu=mu + 1/n_sum[l]
         EE[j,t]=W*SE[j,t]/(2*(1e6)*(mu*PTx[IndexS[0]]/nt[IndexS[2]] + nt[IndexS[2]]*Pcirc))
         TPC[j,t]=mu*PTx[IndexS[0]]/nt[IndexS[2]]+nt[IndexS[2]]*Pcirc
         if (SEtilde > SE[j,t-1]):  # SE outage occurrence
             print("SE outage")
             if (IndexS[2]< Nnt - 1):
                 IndexS[2]=   IndexS[2] + 1 
             if (Pjmax < Ptot):
                 Pjmax=Pjmax + Ptot/Nlink
                 if (IndexS[0] < Npjmax - 1):
                     IndexS[0]=IndexS[0] + 1             
             #Pa=(Pcirc*nt[IndexS[2]]**2)/PTx[IndexS[0]]  # amplifier power
            
         else: # There's no SE outage
                 OutageIndic=1
                 if (EE[j,t-1]> EE[j,t]):
                       print("EE(t+1)< EE(t)")
                       if (IndexS[2]>0):
                            IndexS[2]=   IndexS[2] - 1 
                       if (Pjmax > Ptot/Nlink):
                            Pjmax=Pjmax - Ptot/Nlink
                            if (IndexS[0]>0):
                                IndexS[0]=IndexS[0] - 1    
                            
         Pa=(Pcirc*nt[IndexS[2]]**2)/PTx[IndexS[0]]  # amplifier power
           # Search for tilt*
         for k in range(Tilt.shape[0]):
              TiltSE[k]=capa(Nr,nt[IndexS[2]],N,distD2D,PTx[IndexS[0]],Tilt[k])
         
         SE_RL[j,t]=max(TiltSE) 
         IndexS[1]=np.argmax(TiltSE)
         
         EE_RL[j,t]=np.sqrt(Pa*PTx[IndexS[0]]/Pcirc)*W*SE_RL[j,t]/(4*(1e6)*(PTx[IndexS[0]]*Pa))             
         if (OutageIndic==1):  
             R[t]=R[t]+EE_RL[j,t]
         TPC_RL[j,t]=Pa*PTx[IndexS[0]]/nt[IndexS[2]] + nt[IndexS[2]]*Pcirc
         print(" state")
         print("nt=",nt[IndexS[2]])
         print("Pt=",PTx[IndexS[0]])
         print("tilt=",Tilt[IndexS[1]])
         print("        ")
    
    EEug[t]= sum(SE[:,t])/sum(TPC[:,t])
    EEugRL[t]=sum(SE_RL[:,t])/sum(TPC_RL[:,t])    
    R1_D2D[t]=0.15*sum(SE_RL[:,t]) + 0.85*R[t]
    R2_D2D[t]=0.55*sum(SE_RL[:,t]) + 0.45*R[t]
    R3_D2D[t]=0.85*sum(SE_RL[:,t]) + 0.15*R[t]

CumR1_D2D=np.cumsum(R1_D2D, dtype=float)
CumR2_D2D=np.cumsum(R2_D2D, dtype=float)
CumR3_D2D=np.cumsum(R3_D2D, dtype=float)
# averaging over the number of links         
AvSE=np.sum(SE, axis=0)/Nlink
AvSE_RL=np.sum(SE_RL, axis=0)/Nlink

AvEE=np.sum(EE, axis=0)/Nlink
AvEE_RL=np.sum(EE_RL, axis=0)/Nlink


AvTPC=np.sum(TPC,axis=0)/Nlink
AvTPC_RL=np.sum(TPC_RL,axis=0)/Nlink

Time=np.arange(Ncg)


plt.figure(1)
plt.xlabel('Contiguous time slots')
plt.ylabel('Average SE (bps/Hz)')
 
plt.plot(Time,AvSE_RL, marker='^',label="Average SE*-RL  ,  n_r=16")
plt.plot(Time,AvSE, marker='^',label="Average SE  ,  n_r=16")
plt.legend(loc='center right')
plt.savefig('SESERL.png')
plt.show()


plt.figure(2)
# Add title and axis names
#plt.title('My title')
plt.xlabel('Contiguous time slots')
plt.ylabel('Average EE (Mbps/Joule)')
 

plt.plot(Time,AvEE_RL, marker='*',label="Average EE*-RL, n_r=16")
plt.plot(Time,AvEE, marker='+',label="Average EE*,  n_r=16")
plt.yscale("log")
plt.legend(loc='center right')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('EEEERL.png')
plt.show()

plt.figure(3)
# Add title and axis names
#plt.title('My title')
plt.xlabel('Contiguous time slots')
plt.ylabel('Average TPC (Watt)')
plt.plot(Time,AvTPC_RL, marker='*',label="Average TPC-RL, n_r=16")
plt.plot(Time,AvTPC, marker='+',label="Average TPC,  n_r=16")
plt.yscale("log")
plt.legend(loc='upper right')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('TPCRL.png')
plt.show()

plt.figure(4)
# Add title and axis names
#plt.title('My title')
plt.xlabel('Contiguous time slots')
plt.ylabel('Utility function')
plt.plot(Time,R1_D2D, marker='*',label="P1=0.15, P2=0.85")
plt.plot(Time,R2_D2D, marker='+',label="P1=0.55, P2=0.45")
plt.plot(Time,R3_D2D, marker='^',label="P1=0.85, P2=0.15")
plt.yscale("log")
plt.legend(loc='lower right')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('RewardRL.png')
plt.show()

plt.figure(5)
# Add title and axis names
#plt.title('My title')
plt.xlabel('Contiguous time slots')
plt.ylabel('Average EE_ug (Mbps/Joule)')
 

plt.plot(Time,EEugRL, marker='*',label="Average EE_ug*-RL, n_r=16")
plt.plot(Time,EEug, marker='+',label="Average EEug*,  n_r=16")
plt.legend(loc='center right')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('EEugRL.png')
plt.show()

plt.figure(6)
plt.xlabel('Contiguous time slots')
plt.ylabel('Cumulative reward')
plt.plot(Time,CumR1_D2D, marker='*',label="P1=0.15, P2=0.85")
plt.plot(Time,CumR2_D2D, marker='+',label="P1=0.55, P2=0.45")
plt.plot(Time,CumR3_D2D, marker='^',label="P1=0.85, P2=0.15")
plt.yscale("log")
plt.legend(loc='upper left')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('CumRewardRL.png')
plt.show()