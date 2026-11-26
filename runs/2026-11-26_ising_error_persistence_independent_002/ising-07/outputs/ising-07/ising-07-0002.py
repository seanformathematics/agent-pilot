
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=512; NEQ=5000; NMEAS=20000
ii,jj=np.indices((N,N))
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        acc=np.exp(-dE/T)
        r=rng.random(s.shape)
        flip=(r<acc)&m
        s[flip]*=-1
    return s
out={}
t0=time.time()
for T,seed in zip([1.9,2.0,2.1,2.2,2.3,2.4],[101,202,303,404,505,606]):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for k in range(NEQ): s=sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); S2sq=np.zeros(R)
    m2s=np.zeros((NMEAS,R),dtype=np.float32); m4s=np.zeros((NMEAS,R),dtype=np.float32)
    for k in range(NMEAS):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2s[k]=m*m; m4s[k]=m**4
    np.savez(f'/work/T{T}.npz',m2=m2s,m4=m4s,seed=seed)
    m2=m2s.astype(float); m4=m4s.astype(float)
    U=1-m4.mean()/(3*m2.mean()**2)
    print(T, 'U=',U, 'elapsed',time.time()-t0, flush=True)
