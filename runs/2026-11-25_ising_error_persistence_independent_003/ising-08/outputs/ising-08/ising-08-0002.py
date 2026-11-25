import numpy as np, time
N=16
def make(R,rng):
    return rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
rng=np.random.default_rng(1)
R=1000
s=make(R,rng)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def sweep(s,T,rng):
    for mask in (mask0,mask1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb  # in {-8,-4,0,4,8}
        acc=np.exp(-np.maximum(dE,0)/T)
        r=rng.random(s.shape)
        flip=(r<acc)&mask
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for k in range(50):
    s=sweep(s,2.27,rng)
dt=(time.time()-t)/50
print("time per sweep (R=1000):",dt)