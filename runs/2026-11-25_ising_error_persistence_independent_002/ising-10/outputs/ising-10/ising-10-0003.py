
import numpy as np, time, json, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, n_eq=3000, n_meas=20000, blk=1000):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(S):
        for p in (0,1):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nn
            r=rng.random(S.shape)
            flip=((dE<=0)|(r<np.exp(-dE/T)))&masks[p]
            S=np.where(flip,-S,S)
        return S
    for k in range(n_eq): S=sweep(S)
    m2s=np.zeros(R); m4s=np.zeros(R); m2b=[]; 
    b2=np.zeros(R); 
    for k in range(n_meas):
        S=sweep(S)
        m=S.sum(axis=(1,2))/(N*N)
        m2=m*m
        m2s+=m2; m4s+=m2*m2; b2+=m2
        if (k+1)%blk==0:
            m2b.append(b2/blk); b2=np.zeros(R)
    return m2s/n_meas, m4s/n_meas, np.array(m2b)  # per-replica averages
os.makedirs('/work',exist_ok=True)
out={}
for T,seed in [(1.9,101),(2.0,102)]:
    t=time.time()
    a2,a4,blocks=run(T,seed)
    np.savez(f'/work/res_T{T}.npz',a2=a2,a4=a4,blocks=blocks)
    U=1-a4.mean()/(3*a2.mean()**2)
    print(T, "U=",U, "time",time.time()-t, "block <m2> first/last:", blocks[:2].mean(), blocks[-2:].mean(), flush=True)
