
import numpy as np,time
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
ii,jj=np.indices((N,N));masks=[((ii+jj)%2==k)[None] for k in (0,1)]
res={}
for T in Ts:
    s=np.ones((R,N,N),np.int8)
    def sweep():
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=rng.random(s.shape)<np.exp(-dE/T)
            s[:]=np.where(m&acc,-s,s)
    for _ in range(1000):sweep()
    m2=[];m4=[]
    for k in range(6000):
        sweep()
        if k%2==0:
            m=s.mean((1,2),dtype=float);m2.append(m**2);m4.append(m**4)
    a=np.array(m2).mean(0);b=np.array(m4).mean(0)  # per replica
    U=1-b.mean()/(3*a.mean()**2)
    J=[1-np.delete(b,r).mean()/(3*np.delete(a,r).mean()**2) for r in range(R)]
    err=np.sqrt((R-1)/R*np.sum((np.array(J)-np.mean(J))**2))
    res[T]=(U,err);print(T,U,err,flush=True)
