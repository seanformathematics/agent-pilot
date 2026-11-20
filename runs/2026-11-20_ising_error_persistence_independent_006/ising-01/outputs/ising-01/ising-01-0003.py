import numpy as np, time
N=16; R=1000; T=2.27
rng=np.random.default_rng(12345)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nn  # int8 ok: max 8
        p = np.exp(-dE.astype(np.float64)/T) if False else None
        acc = (dE<=0) | (rng.random(s.shape,dtype=np.float32) < np.exp(-np.maximum(dE,0)/T).astype(np.float32))
        flip = acc & m
        s[flip] *= -1
    return s
t=time.time()
for k in range(100): sweep(s,T)
print("per sweep (R=1000):", (time.time()-t)/100)
