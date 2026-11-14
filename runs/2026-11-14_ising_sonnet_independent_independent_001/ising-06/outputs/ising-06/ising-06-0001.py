
import numpy as np, time, json
rng=np.random.default_rng(12345)
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
masks=[mask,~mask]
def sweep(s,T):
    for mk in masks:
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        acc=rng.random(s.shape)<np.exp(-dE/T)
        s[:]=np.where(mk&acc,-s,s)
res={}
t0=time.time()
for T in Ts:
    s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8) if False else rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(1000): sweep(s,T)
    m2=[];m4=[]
    for k in range(4000):
        sweep(s,T)
        if k%2==0:
            m=s.mean(axis=(1,2),dtype=float); m2.append(m**2); m4.append(m**4)
    m2=np.array(m2).mean(0); m4=np.array(m4).mean(0)  # per replica
    U=1-m4.mean()/(3*m2.mean()**2)
    bs=[]
    for _ in range(300):
        i=rng.integers(0,R,R); bs.append(1-m4[i].mean()/(3*m2[i].mean()**2))
    res[T]=(U,np.std(bs),R*2000)
    print(T,res[T],time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res.json','w'))
