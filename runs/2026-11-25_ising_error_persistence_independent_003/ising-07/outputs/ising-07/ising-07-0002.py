
import numpy as np, time
L=16; R=1000
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
T=2.2
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb  # energy change for flip (values -8..8 step 4)
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(rng.random(s.shape,dtype=np.float32)<p)&m
        s[acc]*=-1
    return s
t=time.time()
for _ in range(20): s=sweep(s,T,rng)
print((time.time()-t)/20,'s per sweep R=',R)
