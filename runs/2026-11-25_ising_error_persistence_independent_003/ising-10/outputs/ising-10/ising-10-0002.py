import numpy as np, time
N=16; R=2000
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb
        p = np.exp(-np.maximum(dE,0)/T)
        acc = (dE<=0) | (rng.random(s.shape) < p)
        acc &= m[None]
        s[acc] *= -1
    return s
t=time.time()
for _ in range(100): sweep(s,2.27,rng)
print("per sweep", (time.time()-t)/100)