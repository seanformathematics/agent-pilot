
import sys; sys.path.insert(0,'/work')
import numpy as np, time
from sw import SW
N=16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=1024; nburn=300; nmeas=2000
todo=[0,1,2,3,4,5,6]
t0=time.time()
for i in todo:
    T=Ts[i]
    rng=np.random.default_rng(1000+i)
    s=SW(N,R,T,rng)
    for _ in range(nburn): s.sweep()
    M2=np.zeros((nmeas,R),dtype=np.float32); M4=np.zeros((nmeas,R),dtype=np.float32)
    for k in range(nmeas):
        s.sweep()
        m=s.mags()
        M2[k]=m**2; M4[k]=m**4
    np.savez(f'/work/prod_{i}.npz', T=T, M2=M2, M4=M4, seed=1000+i)
    print("saved",i,T,time.time()-t0, flush=True)
