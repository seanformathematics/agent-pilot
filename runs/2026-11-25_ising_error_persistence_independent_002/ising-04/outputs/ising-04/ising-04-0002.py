import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(S,T,rng):
    for m in masks:
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nb
        acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T).astype(np.float32))
        S[acc&m]*=-1
    return S
t=time.time()
for _ in range(100): sweep(S,2.2,rng)
print("time per sweep R=1000:",(time.time()-t)/100)