import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,T):
    for p in (0,1):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-dE.clip(0)/T).astype(np.float32))
        s[acc & masks[p]] *= -1
    return s
t=time.time()
for k in range(200): sweep(s,2.27)
print("time per sweep (R=1000):",(time.time()-t)/200)
