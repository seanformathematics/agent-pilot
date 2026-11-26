
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
def setup(R_per, seed):
    rng=np.random.default_rng(seed)
    Tv=np.repeat(np.array(temps),R_per)
    R=len(Tv)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c)[None] for c in (0,1)]
    return rng,Tv,s,masks
def sweep(s,Tv,rng,masks):
    beta=(1.0/Tv).reshape(-1,1,1)
    p4=np.exp(-4*beta); p8=np.exp(-8*beta)
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn  # energy change for flipping is dE
        r=rng.random(s.shape)
        acc=(dE<=0)|((dE==4)&(r<p4))|((dE==8)&(r<p8))
        acc&=m
        s[acc]*=-1
    return s
rng,Tv,s,masks=setup(500,12345)
t=time.time()
for k in range(20): sweep(s,Tv,rng,masks)
dt=(time.time()-t)/20
print("per sweep (3000 replicas) s:",dt, " per replica-sweep us:",dt/len(Tv)*1e6)
