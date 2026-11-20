
import numpy as np, time
N=16; R=1000; T=2.269
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(s,T,rng):
    for p in (0,1):
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nn
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-np.clip(dE,0,None)/T).astype(np.float32))
        s[:,masks[p]]*=np.where(acc[:,masks[p]],-1,1).astype(np.int8)
    return s
t=time.time()
for _ in range(20): sweep(s,T,rng)
print("per sweep (R=1000):",(time.time()-t)/20)
