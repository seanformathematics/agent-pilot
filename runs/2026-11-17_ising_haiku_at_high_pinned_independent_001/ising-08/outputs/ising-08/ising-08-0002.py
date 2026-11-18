
import numpy as np, time
def make_sampler(N,R,seed):
    rng=np.random.default_rng(seed)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    def sweep(s,T,ptab):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            k=(s*nb+4)
            p=ptab[k]
            u=rng.random(s.shape,dtype=np.float32)
            flip=(u<p)&m
            s=np.where(flip,-s,s)
        return s
    return rng,sweep
N=16;R=1000
rng,sweep=make_sampler(N,R,1)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
T=2.269
ptab=np.array([min(1.0,np.exp(-2*(k-4)/T)) for k in range(9)],dtype=np.float32)
t=time.time()
for _ in range(20): s=sweep(s,T,ptab)
print((time.time()-t)/20,'s per sweep')
