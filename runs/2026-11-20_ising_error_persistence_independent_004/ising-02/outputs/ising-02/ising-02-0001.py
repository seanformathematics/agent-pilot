
import numpy as np, time
N=16; R=400; T=2.2; seed=12345
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
maskA=((ii+jj)%2==0); maskB=~maskA
def sweep(s,T):
    for mask in (maskA,maskB):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        p=np.exp(-dE.clip(min=0)/T)
        acc=(rng.random(s.shape)<p)&mask
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(200): s=sweep(s,T)
print("time per sweep (R=400):",(time.time()-t)/200)
