
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=2000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def run(T,seed,n_eq=1000,n_meas=8000):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8)
    def sweep():
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*h
            acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T))
            s[acc&mk]*=-1
    for _ in range(n_eq): sweep()
    S2=np.zeros(R); S4=np.zeros(R); S2b=[]; 
    blocks=[]
    nb=4; per=n_meas//nb
    for b in range(nb):
        b2=np.zeros(R); b4=np.zeros(R)
        for k in range(per):
            sweep()
            M=s.sum(axis=(1,2),dtype=np.int64).astype(np.float64)
            M2=M*M
            b2+=M2; b4+=M2*M2
        S2+=b2; S4+=b4
        blocks.append(1-(b4.sum()/per/R)/(3*(b2.sum()/per/R)**2))
    return S2,S4,n_meas,blocks
out={}
for T in [1.9,2.0,2.1]:
    t=time.time()
    S2,S4,n,blocks=run(T,seed=int(T*1000))
    np.savez(f'/work/T{T}.npz',S2=S2,S4=S4,n=n)
    U=1-(S4.sum()/n/R)/(3*(S2.sum()/n/R)**2)
    print(T,"U=",U,"blocks=",np.round(blocks,5),"time",time.time()-t,flush=True)
