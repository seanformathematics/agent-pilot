
import numpy as np, time
N=16
def run(T, R, nsweep, seed):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table indexed by dE (=2*s*h), dE in -8..8 step 4? use exp directly
    acc={dE:np.exp(-dE/T) for dE in (4,8)}
    t0=time.time()
    for sw in range(nsweep):
        for p in (0,1):
            h=(np.roll(S,1,1).astype(np.int8)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-2+3,2)) if False else None
            nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            dE=2*S*nn
            pr=np.where(dE<=0,1.0,np.exp(-np.maximum(dE,0)/T))
            flip=(rng.random((R,N,N))<pr)&masks[p]
            S=np.where(flip,-S,S)
    return time.time()-t0, S
dt,S=run(2.3,200,200,1)
print("time per sweep R=200:",dt/200)
