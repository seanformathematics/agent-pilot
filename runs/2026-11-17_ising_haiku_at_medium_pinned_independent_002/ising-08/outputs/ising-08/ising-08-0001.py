
import numpy as np, time
def run(T, R=1000, nsweep=20, seed=1):
    rng=np.random.default_rng(seed)
    N=16
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc=np.exp(-4/T*np.arange(0,1)) 
    p4=np.exp(-4/T); p8=np.exp(-8/T)
    t=time.time()
    for _ in range(nsweep):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb  # energy change when flipping
            r=rng.random(s.shape)
            flip=(dE<=0)|(r<np.where(dE==4,p4,p8))
            flip&=m[None]
            s[flip]*=-1
    return time.time()-t
print(run(2.27))
