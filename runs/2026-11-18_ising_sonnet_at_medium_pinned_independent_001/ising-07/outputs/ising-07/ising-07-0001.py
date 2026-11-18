
import numpy as np, time, json
N=16;R=128
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
masks=[mask,~mask]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    rng=np.random.default_rng(1000+k)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=(rng.random(s.shape)<np.exp(-dE/T))&mk
            s[acc]*=-1
    for _ in range(2000): sweep()
    nm=8000
    m2=np.zeros(R);m4=np.zeros(R)
    # block sums per replica
    for _ in range(nm):
        sweep()
        m=s.sum((1,2),dtype=np.int64)/N**2
        m2+=m**2;m4+=m**4
    m2/=nm;m4/=nm
    U=1-m4.mean()/(3*m2.mean()**2)
    jk=[]
    for r in range(R):
        idx=np.arange(R)!=r
        jk.append(1-m4[idx].mean()/(3*m2[idx].mean()**2))
    jk=np.array(jk);err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    res[T]=(U,err,nm*R)
    print(T,U,err,time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res.json','w'))
