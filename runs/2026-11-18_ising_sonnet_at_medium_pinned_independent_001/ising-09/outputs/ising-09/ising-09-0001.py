
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.indices((N,N));mask=[((ii+jj)%2==p) for p in(0,1)]
res={}
for k,T in enumerate(Ts):
    rng=np.random.default_rng(1000+k)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep():
        for p in(0,1):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=(rng.random(s.shape)<np.exp(-dE/T))&mask[p]
            s[acc]*=-1
    for _ in range(2000): sweep()
    m2=[];m4=[]
    t=time.time()
    for it in range(16000):
        sweep()
        if it%4==0:
            m=s.sum((1,2),dtype=np.int64)/N**2
            m2.append(m**2);m4.append(m**4)
    a=np.mean(m2,0);b=np.mean(m4,0)  # per replica
    n=len(m2)
    U=1-b.mean()/(3*a.mean()**2)
    jk=[]
    for g in range(R):
        idx=np.arange(R)!=g
        jk.append(1-b[idx].mean()/(3*a[idx].mean()**2))
    jk=np.array(jk);err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    res[T]=(U,err,n*R);print(T,U,err,time.time()-t,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res.json','w'))
