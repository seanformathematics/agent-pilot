
import sys, time
import numpy as np
sys.path.insert(0,'/work')
from sw import SWReplicas

N=16; nsite=N*N
R=2000; Rh=1000; nsw=4000
for iT,T in enumerate([2.2,1.5]):
    t0=time.time()
    rng=np.random.default_rng(1230000+iT)
    sim=SWReplicas(N,R,T,rng,'hot')
    sim.s[Rh:]=1
    g_m2=np.zeros((nsw,2)); g_m4=np.zeros((nsw,2)); g_E=np.zeros((nsw,2))
    nbk=nsw//100
    B2=np.zeros((R,nbk)); B4=np.zeros((R,nbk)); BE=np.zeros((R,nbk))
    for t in range(nsw):
        sim.sweep()
        mm=sim.magnetization_sum()/nsite
        ee=sim.energy()/nsite
        m2=mm**2; m4=mm**4
        g_m2[t]=(m2[:Rh].mean(),m2[Rh:].mean()); g_m4[t]=(m4[:Rh].mean(),m4[Rh:].mean()); g_E[t]=(ee[:Rh].mean(),ee[Rh:].mean())
        b=t//100
        B2[:,b]+=m2; B4[:,b]+=m4; BE[:,b]+=ee
    np.savez(f"/work/relax_{iT}.npz",T=T,Rh=Rh,g_m2=g_m2,g_m4=g_m4,g_E=g_E,B2=B2,B4=B4,BE=BE)
    print(f"T={T} done in {time.time()-t0:.0f}s",flush=True)
