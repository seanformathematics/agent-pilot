
import numpy as np, sys
sys.path.insert(0,'/work')
from ising import run, binder_jk
L=4
idx=np.arange(2**(L*L))
S=(((idx[:,None]>>np.arange(L*L))&1)*2-1).reshape(-1,L,L)
E=-(np.sum(S*np.roll(S,1,1),axis=(1,2))+np.sum(S*np.roll(S,1,2),axis=(1,2)))
m=S.reshape(-1,L*L).mean(1)
for T,seed in [(2.4,201),(2.0,202)]:
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    M2=np.sum(w*m**2); M4=np.sum(w*m**4)
    Uex=1-M4/(3*M2**2)
    A2,A4,_,_=run(T,N=L,R=2000,nsweep=5000,ntherm=1000,seed=seed)
    U,err=binder_jk(A2,A4)
    print(f"T={T}: exact m2={M2:.5f} U={Uex:.5f} | MC m2={A2.mean():.5f} U={U:.5f} +- {err:.5f}")
