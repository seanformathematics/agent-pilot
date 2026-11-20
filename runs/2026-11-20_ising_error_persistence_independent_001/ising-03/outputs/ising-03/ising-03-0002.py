
import time, numpy as np, sys, json
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
    Ucheck=[]
    for k in range(nmeas):
        s=sweep(s,beta,rng)
        if k%every==0:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; n+=1
            if k%5000==0: Ucheck.append(float(1-np.mean(S4/n)/(3*np.mean(S2/n)**2)))
    return S2,S4,n,Ucheck
out={}
for T,seed in [(1.9,101),(2.0,102)]:
    t=time.time()
    S2,S4,n,uc=run(T,seed)
    np.savez(f"/work/sums_T{T}.npz",S2=S2,S4=S4,n=n)
    U=1-np.mean(S4/n)/(3*np.mean(S2/n)**2)
    print(T,"U=",U,"n=",n,"checks",uc,"time",time.time()-t,flush=True)
