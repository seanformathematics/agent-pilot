
import numpy as np,time,json
N=16;R=256
rng=np.random.default_rng(12345)
ii,jj=np.indices((N,N));masks=[((ii+jj)%2==c) for c in (0,1)]
res={}
t0=time.time()
for T in [1.5+0.1*k for k in range(14)]:
    T=round(T,1)
    s=np.ones((R,N,N),np.int8)
    p=np.exp(-8/T),np.exp(-4/T)
    def sweep():
        global s
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb  # dE=2x
            acc=(x<=0)|(rng.random(s.shape)<np.exp(-2*x/T))
            s=np.where(acc&m,-s,s).astype(np.int8)
    for _ in range(1500):sweep()
    m2=[];m4=[]
    for k in range(12000):
        sweep()
        if k%4==0:
            m=s.mean((1,2),dtype=float);m2.append(m**2);m4.append(m**4)
    a=np.array(m2).mean(0);b=np.array(m4).mean(0)  # per replica
    U=lambda a,b:1-b.mean()/(3*a.mean()**2)
    u=U(a,b)
    jk=np.array([U(np.delete(a,i),np.delete(b,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    res[T]=(u,err,len(m2)*R);print(T,u,err,time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/r.json','w'))
