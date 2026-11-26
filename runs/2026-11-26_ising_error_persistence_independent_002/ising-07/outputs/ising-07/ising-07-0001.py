
import numpy as np, time
N=16; R=512
rng=np.random.default_rng(1)
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
ii,jj=np.indices((N,N))
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb  # energy change for flipping
        acc=np.exp(-dE/T)
        r=rng.random(s.shape,dtype=np.float32)
        flip=(r<acc)&m
        s[flip]*=-1
    return s
t=time.time()
for k in range(100): s=sweep(s,2.27,rng)
print("time per sweep (R=512):",(time.time()-t)/100)
