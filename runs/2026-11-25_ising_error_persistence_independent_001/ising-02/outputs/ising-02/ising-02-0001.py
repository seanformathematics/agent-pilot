
import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,beta,rng):
    for p in (0,1):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
        x=(s*nn)  # in {-4..4}
        acc=np.exp(-2*beta*np.maximum(x,0))
        u=rng.random(s.shape,dtype=np.float32)
        flip=(u<acc)&masks[p]
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(20):
    s=sweep(s,1/2.269,rng)
print((time.time()-t)/20,"s per sweep for R=",R)
