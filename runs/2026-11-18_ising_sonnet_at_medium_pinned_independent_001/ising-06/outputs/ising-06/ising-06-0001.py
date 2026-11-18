
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=((np.indices((N,N)).sum(0))%2==0)
masks=[mask,~mask]
def run(T,seed,neq,nms):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),dtype=np.int8)
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=rng.random(s.shape)<np.exp(-dE/T)
            s[...]=np.where(acc&mk,-s,s)
    for _ in range(neq):sweep()
    m2=[];m4=[]
    for i in range(nms):
        sweep()
        m=s.sum((1,2))/N**2
        m2.append(m**2);m4.append(m**4)
    m2=np.array(m2).mean(0);m4=np.array(m4).mean(0)  # per replica
    U=1-m4.mean()/(3*m2.mean()**2)
    jk=[]
    for r in range(R):
        k=np.arange(R)!=r
        jk.append(1-m4[k].mean()/(3*m2[k].mean()**2))
    jk=np.array(jk)
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    return U,err
t=time.time()
res={}
for T in Ts:
    res[T]=run(T,1000+int(T*10),2000,4000)
    print(T,res[T],time.time()-t,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/r.json','w'))
