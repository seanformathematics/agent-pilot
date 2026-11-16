
import numpy as np,time
N=16;R=1000;T=2.2
rng=np.random.default_rng(1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
t0=time.time()
for _ in range(20):
    for m in masks:
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE=2*S*nb
        acc=np.exp(-np.clip(dE,0,None)/T)
        r=rng.random(S.shape)
        flip=(r<acc)&m
        S[flip]*=-1
print((time.time()-t0)/20)
