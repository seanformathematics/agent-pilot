
import time, numpy as np
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def sweep(s,beta,rng):
    for m in (mask0,mask1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-beta*np.clip(dE,0,None)))
        s=np.where(acc&m,-s,s).astype(np.int8)
    return s
def run(T,seed,neq=1000,nmeas=20000,every=2):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    beta=1/T
    for _ in range(neq): s=sweep(s,beta,rng)
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for k in range(nmeas):
        s=sweep(s,beta,rng)
        if k%every==0:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; n+=1
    return S2,S4,n
def jk(S2,S4,n):
    R=len(S2)
    tot2=S2.sum(); tot4=S4.sum()
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    a2=(tot2-S2)/((R-1)*n); a4=(tot4-S4)/((R-1)*n)
    Ui=1-a4/(3*a2**2)
    return U,np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
for T,seed in [(2.1,103),(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time()
    S2,S4,n=run(T,seed)
    np.savez(f"/work/sums_T{T}.npz",S2=S2,S4=S4,n=n)
    print(T,jk(S2,S4,n),n,round(time.time()-t),flush=True)
