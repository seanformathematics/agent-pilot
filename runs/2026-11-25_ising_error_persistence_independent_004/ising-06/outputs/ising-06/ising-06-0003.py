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
    S2=np.zeros(R); S4=np.zeros(R); E1=np.zeros(R)
    ts=np.zeros((MEAS,2))
    for k in range(MEAS):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2
        ts[k,0]=m2.mean(); ts[k,1]=(m2*m2).mean()
    return S2/MEAS, S4/MEAS, ts
def ustats(a2,a4):
    # a2=<m^2> per replica, a4=<m^4> per replica
    U=1-a4.mean()/(3*a2.mean()**2)
    jk=np.empty(R)
    for r in range(R):
        mask=np.ones(R,bool); mask[r]=False
        jk[r]=1-a4[mask].mean()/(3*a2[mask].mean()**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err
out={}
for T,seed in [(1.9,101),(2.0,102)]:
    t=time.time()
    a2,a4,ts=run(T,seed)
    U,err=ustats(a2,a4)
    # halves check
    h1=ustats(a2[:500],a4[:500]); h2=ustats(a2[500:],a4[500:])
    out[T]=dict(U=U,err=err,n=MEAS*R)
    print(f"T={T} U={U:.5f} err={err:.5f} halves(R) {h1[0]:.5f},{h2[0]:.5f} time={time.time()-t:.1f}s",flush=True)
    np.save(f"/work/a2_{T}.npy",a2); np.save(f"/work/a4_{T}.npy",a4)
