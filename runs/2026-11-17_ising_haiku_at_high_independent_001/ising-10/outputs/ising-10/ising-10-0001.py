
import numpy as np, time
N=16; R=512
rng=np.random.default_rng(1)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T,rng):
    for m in masks:
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nn
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(dE<=0)|(rng.random(S.shape)<p)
        S[...]=np.where(acc&m,-S,S)
    return S
t=time.time()
for k in range(100): sweep(S,2.0,rng)
print((time.time()-t)/100,"s per sweep for R=",R)
