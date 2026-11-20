
import numpy as np, time, json
N=16; R=2000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        s=np.where(acc&masks[c],-s,s).astype(np.int8)
    return s
def run(T,seed,neq=3000,nmeas=8000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(neq): s=sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); S2s=[]
    for k in range(nmeas):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2/nmeas, S4/nmeas
def ujack(a2,a4):
    def U(x2,x4): return 1-x4.mean()/(3*x2.mean()**2)
    full=U(a2,a4)
    n=len(a2); jk=np.empty(n)
    tot2=a2.sum(); tot4=a4.sum()
    for r in range(n):
        jk[r]=U((tot2-a2[r])/(n-1),(tot4-a4[r])/(n-1))
    err=np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))
    return full,err
res={}
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    a2,a4=run(T,seed)
    U,e=ujack(a2,a4)
    res[T]=dict(U=U,err=e,seed=seed,m2=a2.mean(),)
    print(T,U,e,a2.mean(),time.time()-t,flush=True)
    np.savez(f"/work/raw_T{T}.npz",a2=a2,a4=a4)
