
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
masks=[mask,~mask]
def run(T,seed,burn=1000,sweeps=8000,every=4):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<.5,-1,1).astype(np.int8)
    p={k:np.exp(-2*k/T) for k in (4,8)}
    m2=[];m4=[]
    for t in range(burn+sweeps):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random(s.shape)<np.exp(-dE/T))
            s=np.where(acc&mk,-s,s)
        if t>=burn and t%every==0:
            m=s.reshape(R,-1).mean(1,dtype=np.float64)
            m2.append(m**2);m4.append(m**4)
    a=np.mean(m2,0);b=np.mean(m4,0)  # per replica
    U=lambda a,b:1-b.mean()/(3*a.mean()**2)
    u=U(a,b)
    jk=np.array([U(np.delete(a,i),np.delete(b,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    return u,err,len(m2)*R
t0=time.time();res={}
for i,T in enumerate(Ts):
    res[T]=run(T,1000+i,sweeps=3000 if True else 0);print(T,res[T],time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/r.json','w'))
