
import numpy as np
from scipy.special import ellipk
def simulate(L,T,R,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
    ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    tab=np.zeros(9)
    for k in [-4,-2,0,2,4]: tab[k+4]=min(1.0,np.exp(-2.0*k/T))
    def sweep(s):
        for mk in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=tab[(s*nn).astype(np.int16)+4]
            flip=(rng.random(s.shape)<p)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    es=[]
    for t in range(nmeas):
        s=sweep(s)
        nnb=(np.roll(s,1,1)*s+np.roll(s,1,2)*s).reshape(R,-1).astype(np.int64).sum(1)
        es.append(-nnb/(L*L))
    es=np.array(es)
    return es.mean()
def onsager(T):
    K=1/T; k=2*np.sinh(2*K)/np.cosh(2*K)**2
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*ellipk(k*k))
T=2.4
for L in [8,16,32,64]:
    print(L, simulate(L,T,max(64,int(2e6/(L*L))),500,300,L), onsager(T))
# independent check of onsager via numerical integral of the standard formula using quad
from scipy.integrate import quad
K=1/T
kap=2*np.sinh(2*K)/np.cosh(2*K)**2
Kint=quad(lambda th: 1/np.sqrt(1-kap**2*np.sin(th)**2),0,np.pi/2)[0]
print("K(kappa) quad",Kint, "ellipk", ellipk(kap**2))
