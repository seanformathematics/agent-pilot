
import numpy as np, time
N=16
def run(T, R, sweeps, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc=np.exp(-4.0*np.arange(-8,9)/T)  # not used directly
    # precompute acceptance for dE = 2*s*nsum, nsum in {-4,-2,0,2,4}
    p={}
    for nsum in [-4,-2,0,2,4]:
        for sv in [-1,1]:
            dE=2*sv*nsum
            p[(sv,nsum)]=np.exp(-dE/T)
    t0=time.time()
    for sw in range(sweeps):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            pr=np.exp(-np.clip(dE,0,None)/T)
            u=rng.random(s.shape)
            flip=(u<pr)&m
            s=np.where(flip,-s,s).astype(np.int8)
    return s, time.time()-t0
s,dt=run(2.269,500,100,1)
print(dt)
