
import sys, math, numpy as np; sys.path.insert(0,'/work')
from ising import run
N=4; V=N*N
# exact enumeration
states=np.arange(2**V)
spins=((states[:,None]>>np.arange(V))&1)*2-1
spins=spins.reshape(-1,N,N)
E=np.zeros(len(states))
for ax in [1,2]:
    E-= (spins*np.roll(spins,-1,axis=ax)).sum(axis=(1,2))
for T in [2.3]:
    w=np.exp(-E/T); Z=w.sum()
    m=spins.sum(axis=(1,2))/V
    m2=(w*m**2).sum()/Z; m4=(w*m**4).sum()/Z
    print('exact',T,m2,m4,1-m4/(3*m2*m2), (w*E).sum()/Z/V)
    a,b=run(N,T,3,2000,400000)
    a=np.array(a);b=np.array(b)
    print('wolff',a.mean(),b.mean(),1-b.mean()/(3*a.mean()**2))
