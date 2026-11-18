
import numpy as np, time
N=16; R=512; T=2.27
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for k in range(2):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        p=np.exp(-dE.astype(np.float64)/T)
        acc=(rng.random(s.shape)<p)&masks[k]
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(100): s=sweep(s,T,rng)
print("time per sweep (R=512):",(time.time()-t)/100)
