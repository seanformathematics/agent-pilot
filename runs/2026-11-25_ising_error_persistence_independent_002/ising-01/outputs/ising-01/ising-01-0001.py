
import numpy as np, time
N=16; R=2000
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,beta,rng):
    for m in masks:
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb  # energy change for flipping (E=-sum ss)
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-beta*np.clip(dE,0,None)))
        flip=acc&m
        s[flip]*=-1
    return s
t=time.time()
for k in range(100):
    sweep(s,1/2.2,rng)
print((time.time()-t)/100, "s per sweep for R=",R)
