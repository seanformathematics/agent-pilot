
import numpy as np, itertools
from scipy.special import ellipk
L=4; T=2.3
# exact enumeration
confs=np.array(list(itertools.product([-1,1],repeat=L*L)),dtype=np.int64).reshape(-1,L,L)
nb=(np.roll(confs,1,1)*confs).sum((1,2))+(np.roll(confs,1,2)*confs).sum((1,2))
E=-nb
w=np.exp(-E/T); Zs=w.sum()
e_exact=(w*E).sum()/Zs/(L*L)
m=confs.reshape(len(confs),-1).mean(1)
print("exact e per site",e_exact, "exact <m^2>",(w*m*m).sum()/Zs)
# metropolis
def simulate(T,L,R,nsw,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
    ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    tab=np.zeros(9)
    for k in [-4,-2,0,2,4]: tab[k+4]=min(1.0,np.exp(-2.0*k/T))
    es=[];m2=[]
    for t in range(nsw):
        for mk in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=tab[(s*nn).astype(np.int16)+4]
            flip=(rng.random(s.shape)<p)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        if t>500:
            nnb=(np.roll(s,1,1)*s+np.roll(s,1,2)*s).reshape(R,-1).astype(np.int64).sum(1)
            es.append(-nnb/(L*L)); m2.append(s.reshape(R,-1).mean(1)**2)
    return np.mean(es), np.mean(m2)
print("metro", simulate(T,L,200,3000,3))
# Onsager check
def onsager(T):
    K=1/T; k=2*np.sinh(2*K)/np.cosh(2*K)**2
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*ellipk(k*k))
print("onsager", onsager(2.3), onsager(2.4), onsager(1.9))
