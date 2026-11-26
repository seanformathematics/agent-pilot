
import numpy as np, time
rng = np.random.default_rng(12345)
N=16; R=2000
s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
def sweep(s,beta):
    for mk in masks:
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE = 2*s*nb  # energy change for flipping
        acc = (dE<=0) | (rng.random(s.shape) < np.exp(-beta*np.clip(dE,0,None)))
        flip = acc & mk
        s[flip] *= -1
    return s
t=time.time()
for _ in range(10): sweep(s,1/2.2)
print("time per sweep (R=2000):", (time.time()-t)/10)
