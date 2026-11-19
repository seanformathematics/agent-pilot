
import sys, time
import numpy as np
sys.path.insert(0,'/work')
from exact import exact_e_C
from sw import SWReplicas
from ana import jackknife_U

N=16; nsite=N*N
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
Rh=400; Rc=400; R=Rh+Rc
nburn=100; nmeas=600
t0=time.time()
for iT,T in enumerate(Ts):
    rng=np.random.default_rng(2024000+iT)
    sim=SWReplicas(N,R,T,rng,'hot')
    sim.s[Rh:]=1          # replicas Rh.. start cold (all +1); first Rh are hot
    trace=[]
    for t in range(nburn):
        sim.sweep()
        mm=sim.magnetization_sum()/nsite
        trace.append((np.mean(mm[:Rh]**2),np.mean(mm[Rh:]**2)))
    trace=np.array(trace)
    S2=np.zeros(R); S4=np.zeros(R); SE=np.zeros(R); SE2=np.zeros(R)
    for _ in range(nmeas):
        sim.sweep()
        mm=sim.magnetization_sum()/nsite
        S2+=mm**2; S4+=mm**4
        ee=sim.energy().astype(float); SE+=ee; SE2+=ee**2
    Uh,eh,_=jackknife_U(S2[:Rh],S4[:Rh],nmeas)
    Uc,ec,_=jackknife_U(S2[Rh:],S4[Rh:],nmeas)
    U,err,Ubc=jackknife_U(S2,S4,nmeas)
    e_ex,C_ex=exact_e_C(N,T)
    e_r=SE/nmeas/nsite
    e_mc=e_r.mean(); e_err=e_r.std(ddof=1)/np.sqrt(R)
    E_mean=SE.sum()/(R*nmeas); E2_mean=SE2.sum()/(R*nmeas)
    C_mc=(E2_mean-E_mean**2)/T**2/nsite
    early=trace[:3].mean(axis=0); late=trace[-20:].mean(axis=0)
    print(f"T={T}: U={U:.4f}+-{err:.4f} | hot {Uh:.4f}+-{eh:.4f} cold {Uc:.4f}+-{ec:.4f} (diff z={(Uh-Uc)/np.hypot(eh,ec):+.2f}) | e z={(e_mc-e_ex)/e_err:+.2f} (e={e_mc:.5f}, ex={e_ex:.5f}) | C mc={C_mc:.4f} ex={C_ex:.4f} | m2 hot/cold first3 sweeps {early[0]:.3f}/{early[1]:.3f}, last20 {late[0]:.4f}/{late[1]:.4f}  [{time.time()-t0:.0f}s]")
