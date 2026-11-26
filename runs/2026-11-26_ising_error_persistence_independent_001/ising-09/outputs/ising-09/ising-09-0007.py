
import numpy as np
from scipy.special import ellipk
N=16
def simulate(T, R, nburn, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    tab=np.zeros(9)
    for k in [-4,-2,0,2,4]: tab[k+4]=min(1.0,np.exp(-2.0*k/T))
    def sweep(s):
        for m in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=tab[(s*nn).astype(np.int16)+4]
            flip=(rng.random(s.shape)<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    E=[];M=[]
    for k in range(nmeas):
        for _ in range(every): s=sweep(s)
        nnb=(np.roll(s,1,1)*s+np.roll(s,1,2)*s).reshape(R,-1).sum(1)
        E.append(-nnb/(2*N*N)); M.append(s.reshape(R,-1).mean(1))
    return np.array(E),np.array(M)
def onsager_e(T):
    K=1/T
    k=2*np.sinh(2*K)/np.cosh(2*K)**2
    # K1(k)=ellipk(m=k^2)*(2/pi)... use scipy ellipk with parameter k^2
    K1=ellipk(k*k)
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*K1)
for T in [2.3,2.4]:
    E,M=simulate(T,200,1000,400,2,55)
    print("T",T,"sim e",E.mean(),"Onsager e",onsager_e(T))
E,M=simulate(50.0,2000,200,200,1,9)
m2=M**2;m4=M**4
print("high T U:",1-m4.mean()/(3*m2.mean()**2))
