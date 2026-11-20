import sys, numpy as np
sys.path.insert(0,"/work")
from wolff import run
N=4; V=N*N
confs=((np.arange(2**V)[:,None]>>np.arange(V))&1)*2-1
confs=confs.reshape(-1,N,N)
E=-(np.sum(confs*np.roll(confs,1,1),(1,2))+np.sum(confs*np.roll(confs,1,2),(1,2)))
M=confs.sum((1,2))
for T in [1.9,2.3]:
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    m2=(M/V)**2
    m2e=np.sum(w*m2**1); m4e=np.sum(w*m2**2)
    Ue=1-m4e/(3*m2e**2)
    Mw,_=run(N,T,7,400000,5000)
    x=(Mw/V)**2
    Uw=1-np.mean(x**2)/(3*np.mean(x)**2)
    print(T,"exact U",round(Ue,5),"wolff U",round(Uw,5))
