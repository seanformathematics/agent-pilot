
import numpy as np, time, os
print(os.cpu_count(), np.__version__)
N=16; R=1000; T=2.27
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,beta,rng):
    for p in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h  # energy change for flipping
        acc=np.exp(-beta*np.clip(dE,0,None))
        u=rng.random(s.shape)
        flip=(u<acc)&masks[p]
        s[flip]*=-1
    return s
beta=1/T
t=time.time()
for k in range(100): sweep(s,beta,rng)
print("time per sweep (R=1000):",(time.time()-t)/100)
