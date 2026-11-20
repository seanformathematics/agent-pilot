
import numpy as np, time
N=16
def make(R,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    return s,masks,rng
def sweep(s,masks,rng,acc):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn  # in {-8..8}
        r=rng.random(s.shape,dtype=np.float32)
        flip=(r<acc[dE+8]) & m
        s[flip]*=-1
T=2.3
R=1000
s,masks,rng=make(R,1)
acc=np.zeros(17,dtype=np.float32)
for d in range(-8,9):
    acc[d+8]=min(1.0,np.exp(-d/T))
t=time.time()
for k in range(50): sweep(s,masks,rng,acc)
print('time per sweep R=1000:',(time.time()-t)/50)
