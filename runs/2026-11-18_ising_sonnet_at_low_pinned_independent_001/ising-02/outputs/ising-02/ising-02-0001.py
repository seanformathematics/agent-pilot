
import numpy as np, time, json
N=16;R=128
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
masks=[mask,~mask]
def run(T,seed,neq,nm):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<.5,1,-1).astype(np.int8)
    p={k:np.exp(-2*k*2/T) for k in (1,2)}
    acc=[];
    for it in range(neq+nm):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb  # dE=2x
            pr=np.exp(-2.0*x/T)
            fl=(rng.random(s.shape)<pr)&mk
            s=np.where(fl,-s,s)
        if it>=neq:
            m=s.reshape(R,-1).sum(1)/N/N
            acc.append((m**2,m**4))
    a=np.array(acc)  # nm,2,R
    return a.mean(0)  # 2,R
out={}
t0=time.time()
for i,T in enumerate(Ts):
    m=run(T,1000+i,1000,8000)
    m2,m4=m
    def U(a,b):return 1-b.mean()/(3*a.mean()**2)
    u=U(m2,m4)
    jk=np.array([U(np.delete(m2,k),np.delete(m4,k)) for k in range(R)])
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    out[T]=(u,err);print(T,u,err,time.time()-t0,flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/o.json','w'))
