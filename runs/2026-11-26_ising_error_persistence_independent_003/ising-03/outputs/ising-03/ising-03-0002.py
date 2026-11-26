import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h  # in {-8..8}
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        flip=acc&masks[c]
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(20): s=sweep(s,2.27,rng)
print("time per sweep (R=1000):",(time.time()-t)/20)