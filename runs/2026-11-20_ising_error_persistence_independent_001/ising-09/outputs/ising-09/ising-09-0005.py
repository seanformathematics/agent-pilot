
import numpy as np, time, json
N=16
def make_sweep(N,T,rng):
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep(s):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nn
            prob=np.exp(-np.maximum(dE,0)/T)
            r=rng.random(s.shape,dtype=np.float32)
            s=np.where((r<prob)&masks[p],-s,s)
        return s
    return sweep
out={}
t0=time.time()
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    seed=int(round(T*1000))+7
    rng=np.random.default_rng(seed)
    sw=make_sweep(N,T,rng)
    R=1000; nb=2000; nm=8000
    s=np.where(rng.random((R,N,N))<0.5,-1.0,1.0).astype(np.float32)
    for _ in range(nb): s=sw(s)
    M2=np.zeros((R,)); M4=np.zeros((R,))
    series=np.zeros((nm,R),dtype=np.float32)
    for k in range(nm):
        s=sw(s); m=s.mean(axis=(1,2)); series[k]=m
    m=series.astype(np.float64)
    a2=(m**2).mean(axis=0); a4=(m**4).mean(axis=0)
    tot2=a2.sum(); tot4=a4.sum()
    U=1-tot4/R/(3*(tot2/R)**2)
    Uj=np.array([1-((tot4-a4[r])/(R-1))/(3*((tot2-a2[r])/(R-1))**2) for r in range(R)])
    err=float(np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2)))
    # half-split consistency
    h=nm//2
    Ua=1-np.mean(m[:h]**4)/(3*np.mean(m[:h]**2)**2); Ub=1-np.mean(m[h:]**4)/(3*np.mean(m[h:]**2)**2)
    out[T]=dict(seed=seed,U=float(U),err=err,Ufirst=float(Ua),Usecond=float(Ub),n=R*nm)
    print(T,out[T],"t=",round(time.time()-t0,1),flush=True)
    json.dump(out,open("/work/prod.json","w"),indent=1)
