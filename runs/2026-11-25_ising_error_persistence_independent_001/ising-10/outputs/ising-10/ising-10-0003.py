
import time, numpy as np
N=16; R=1000; NTH=3000; NMEAS=15000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
for k,T in enumerate([2.2,2.3,2.4]):
    seed=12348+k
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*h
            acc=(rng.random(s.shape)<np.exp(-dE/T))&m
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    t0=time.time()
    for _ in range(NTH): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    blocks=[]
    for it in range(NMEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        S2+=m*m; S4+=m**4; S1+=np.abs(m)
        if (it+1)%1500==0:
            n=it+1
            blocks.append(float(1-np.mean(S4/n)/(3*np.mean(S2/n)**2)))
    n=NMEAS
    m2=S2/n; m4=S4/n
    np.savez(f"/work/rep_T{T}.npz",m2=m2,m4=m4,absm=S1/n)
    U=1-m4.mean()/(3*m2.mean()**2)
    Uj=np.array([1-np.delete(m4,r).mean()/(3*np.delete(m2,r).mean()**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(f"T={T} U={U:.5f} jackknife_err={err:.5f} time={time.time()-t0:.0f}s block_U={np.round(blocks,4).tolist()} <|m|>={np.mean(S1/n):.4f}", flush=True)
