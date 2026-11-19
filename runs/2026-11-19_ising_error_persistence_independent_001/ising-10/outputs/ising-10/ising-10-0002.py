import numpy as np, time
N=16; R=2000; T=2.27
rng=np.random.default_rng(12345)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb  # energy change for flipping (E=-sum ss)
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-dE.clip(0)/T).astype(np.float32))
        flip=acc&m
        s[flip]*=-1
    return s
t=time.time()
for k in range(20): sweep(s,T,rng)
print("per sweep",(time.time()-t)/20)