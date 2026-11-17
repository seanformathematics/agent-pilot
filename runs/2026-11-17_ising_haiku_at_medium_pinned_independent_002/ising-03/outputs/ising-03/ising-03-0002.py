import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
R,L=1000,16
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,beta):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-beta*np.clip(dE,0,None)))
        s[:,m]=np.where(acc[:,m],-s[:,m],s[:,m])
t=time.time()
for _ in range(20): sweep(s,1/2.3)
print("per sweep R=1000:",(time.time()-t)/20)
