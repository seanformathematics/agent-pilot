import numpy as np, time, json, os
N=16; R=1000; EQ=3000; MEAS=20000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8)
    def sweep(s):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nn
            pr=np.exp(-np.clip(dE,0,None)/T).astype(np.float32)
            acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<pr)
            s[acc & masks[p]] *= -1
    for k in range(EQ): sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    ts=np.zeros((MEAS,2))
    for k in range(MEAS):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2
        ts[k,0]=m2.mean(); ts[k,1]=(m2*m2).mean()
    return S2/MEAS, S4/MEAS, ts
def ustats(a2,a4):
    n=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    T2=a2.sum(); T4=a4.sum()
    jk=1-((T4-a4)/(n-1))/(3*((T2-a2)/(n-1))**2)
    err=np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
    return U,err
for T,seed in [(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time()
    a2,a4,ts=run(T,seed)
    U,err=ustats(a2,a4)
    h1=ustats(a2[:500],a4[:500]); h2=ustats(a2[500:],a4[500:])
    print(f"T={T} U={U:.5f} err={err:.5f} halves {h1[0]:.5f},{h2[0]:.5f} (half err {h1[1]:.5f}) time={time.time()-t:.1f}s",flush=True)
    np.savez(f"/work/res_{T}.npz",a2=a2,a4=a4,ts=ts)
