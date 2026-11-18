import numpy as np, time
N=16; R=1000; T=2.0
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s):
    for p in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
        flip=acc&masks[p]
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(100): s=sweep(s)
print("time per sweep (R=1000):",(time.time()-t)/100)
