import numpy as np, time, os
N=16; R=1000
rng=np.random.default_rng(1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(S,T,rng):
    acc=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)])
    for p in (0,1):
        nb=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
        dE=2*S*nb
        a=acc[(dE+8)]
        flip=masks[p]&(rng.random(S.shape)<a)
        S[flip]*=-1
    return S
t=time.time()
for _ in range(20): sweep(S,2.2,rng)
print("per sweep", (time.time()-t)/20)
