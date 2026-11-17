import sys, time; sys.path.insert(0,'/work')
from ising import *
import numpy as np
N=4
cfg=np.arange(2**(N*N))
bits=((cfg[:,None]>>np.arange(N*N))&1)
S=(2*bits-1).reshape(-1,N,N).astype(np.float64)
m=S.sum(axis=(1,2))/N**2
E=-(np.sum(S*np.roll(S,-1,1),axis=(1,2))+np.sum(S*np.roll(S,-1,2),axis=(1,2)))
masks=make_masks(N)
T=2.0
w=np.exp(-(E-E.min())/T); w/=w.sum()
exact=1-np.sum(w*m**4)/(3*np.sum(w*m**2)**2)
rng=np.random.default_rng(90210)
pac=pacc_table(T)
R=20000
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
t0=time.time()
for _ in range(300): s=sweep(s,pac,masks,rng)
a2=np.zeros(R); a4=np.zeros(R); n=0
for k in range(6000):
    s=sweep(s,pac,masks,rng)
    mm=s.sum(axis=(1,2))/N**2
    a2+=mm**2; a4+=mm**4; n+=1
p2=a2/n; p4=a4/n
U=1-p4.mean()/(3*p2.mean()**2)
S2=p2.sum(); S4=p4.sum()
jk=np.array([1-((S4-p4[r])/(R-1))/(3*((S2-p2[r])/(R-1))**2) for r in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print(f"exact={exact:.5f} MC={U:.5f} ± {err:.5f}  diff/err={(U-exact)/err:.2f} time={time.time()-t0:.1f}")
