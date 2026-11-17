import numpy as np, time, json, sys
sys.path.insert(0,'/work')
import ising
temps=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=400; nequil=2000; nmeas=4000
t0=time.time()
for i,T in enumerate(temps):
    seed=2024+7+i
    rng=np.random.default_rng(seed)
    if T<2.0:
        s=np.ones((R,16,16),dtype=np.int8)*np.where(rng.random((R,1,1))<0.5,1,-1).astype(np.int8)
    else:
        s=np.where(rng.random((R,16,16))<0.5,1,-1).astype(np.int8)
    eq_blocks=[]
    for k in range(nequil):
        s=ising.sweep(s,T,rng)
        if (k+1)%500==0:
            m=s.reshape(R,-1).mean(1)
            eq_blocks.append(float(np.mean(m*m)))
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    for k in range(nmeas):
        s=ising.sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1); m2=m*m
        S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    S2/=nmeas; S4/=nmeas; S1/=nmeas
    np.savez(f'/work/raw_T{T:.2f}_N16.npz',S2=S2,S4=S4,S1=S1)
    U=1-S4.mean()/(3*S2.mean()**2)
    print(f"T={T} U={U:.5f} eq<m2> blocks={[round(x,4) for x in eq_blocks]} t={time.time()-t0:.1f}", flush=True)
