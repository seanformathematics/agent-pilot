import numpy as np, time, sys, json
sys.path.insert(0,'/work')
import ising
temps=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=400; nequil=5000; nmeas=3000
t0=time.time()
for i,T in enumerate(temps):
    seed=99000+7+i
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,16,16))<0.5,1,-1).astype(np.int8)
    for k in range(nequil):
        s=ising.sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    for k in range(nmeas):
        s=ising.sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1); m2=m*m
        S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    S2/=nmeas; S4/=nmeas; S1/=nmeas
    np.savez(f'/work/val_T{T:.2f}_N16.npz',S2=S2,S4=S4,S1=S1)
    U=1-S4.mean()/(3*S2.mean()**2)
    print(f"T={T} U={U:.5f} t={time.time()-t0:.1f}", flush=True)
