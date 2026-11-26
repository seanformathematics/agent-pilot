
import numpy as np, time, os
N=16; R=512; NEQ=5000; NMEAS=20000; BLK=100
ii,jj=np.indices((N,N))
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        acc=np.exp(-dE/T)
        r=rng.random(s.shape)
        s[(r<acc)&m]*=-1
    return s
t0=time.time()
for T,seed in zip([2.2,2.3,2.4],[404,505,606]):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for k in range(NEQ): s=sweep(s,T,rng)
    nb_=NMEAS//BLK
    B2=np.zeros((nb_,R)); B4=np.zeros((nb_,R))
    for k in range(NMEAS):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        B2[k//BLK]+=m*m; B4[k//BLK]+=m**4
    np.savez(f'/work/T{T}.npz',B2=B2/BLK,B4=B4/BLK,seed=seed,NEQ=NEQ)
    print(T,'done',time.time()-t0,flush=True)
