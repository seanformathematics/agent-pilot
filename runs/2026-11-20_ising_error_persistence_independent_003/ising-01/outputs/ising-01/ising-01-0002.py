import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def sweep(s,T,rng):
    for mk in (mask0,mask1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float64)
        dE=2*s*h
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(rng.random(s.shape)<p)&mk
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(100): s=sweep(s,2.2,rng)
print("time per sweep R=1000:", (time.time()-t)/100)
