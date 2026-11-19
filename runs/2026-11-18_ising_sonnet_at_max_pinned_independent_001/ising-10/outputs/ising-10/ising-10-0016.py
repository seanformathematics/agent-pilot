
import sys, time
import numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U
from metro import MetroReplicas
from sw import SWReplicas

pooled=np.load('/work/pooled12.npy')   # columns T, U, err
N=16; nsite=N*N
R=1000; nburn=500; nmeas=5000
for iT,T in enumerate([1.5,1.7,1.9]):
    t0=time.time()
    rng=np.random.default_rng(616100+iT)
    sim=MetroReplicas(N,R,T,rng)
    sim.s[:]=1                                  # cold start
    for _ in range(nburn): sim.sweep()
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(nmeas):
        sim.sweep()
        mm=sim.s.sum(axis=(1,2),dtype=np.int64)/nsite
        S2+=mm**2; S4+=mm**4
    U,err,_=jackknife_U(S2,S4,nmeas)
    row=pooled[np.argmin(np.abs(pooled[:,0]-T))]
    print(f"Metropolis(cold) T={T}: U={U:.6f}+-{err:.6f} | SW pooled U={row[1]:.6f}+-{row[2]:.6f} | diff={U-row[1]:+.6f}, z={(U-row[1])/np.hypot(err,row[2]):+.2f}  [{time.time()-t0:.0f}s]", flush=True)

print("--- SW estimate of probability weight with |m|<0.5 (sector-crossing region) ---")
for iT,T in enumerate([1.7,1.9,2.0,2.1,2.2]):
    rng=np.random.default_rng(717100+iT)
    sim=SWReplicas(N,R,T,rng,'hot')
    for _ in range(100): sim.sweep()
    cnt=0; tot=0
    for _ in range(600):
        sim.sweep()
        mm=np.abs(sim.magnetization_sum()/nsite)
        cnt+=np.sum(mm<0.5); tot+=R
    print(f"T={T}: P(|m|<0.5) = {cnt/tot:.2e}  ({cnt} of {tot})")
