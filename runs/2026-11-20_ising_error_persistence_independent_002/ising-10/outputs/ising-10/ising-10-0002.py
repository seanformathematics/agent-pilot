
import numpy as np, time
N=16
def make(R,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    return s,masks,rng
def sweep(s,masks,rng,pt):
    for mk in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        delta=(s*nn).astype(np.int8)  # in {-4,-2,0,2,4}
        r=rng.random(s.shape,dtype=np.float32)
        acc=(delta<=0)|(r<pt[delta+4])
        flip=acc&mk
        s[flip]*=-1
R=1000
s,masks,rng=make(R,1)
T=2.3
pt=np.exp(-2*np.arange(-4,5)/T).astype(np.float32)
t=time.time()
for _ in range(100): sweep(s,masks,rng,pt)
print("per sweep", (time.time()-t)/100)
