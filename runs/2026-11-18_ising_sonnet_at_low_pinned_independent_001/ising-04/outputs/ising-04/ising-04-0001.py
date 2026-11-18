
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
mask=((np.indices((N,N)).sum(0))%2==0)
out={}
t0=time.time()
for T in Ts:
    s=np.ones((R,N,N),np.int8)
    s[rng.random((R,N,N))<0.5]=-1
    def sweep():
        for p in (mask,~mask):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=rng.random(s.shape)<np.exp(-dE/T)
            s[:]=np.where(acc&p,-s,s)
    for _ in range(1000): sweep()
    m2=[];m4=[]
    for i in range(6000):
        sweep()
        if i%2==0:
            m=s.reshape(R,-1).mean(1,dtype=float)
            m2.append(m**2);m4.append(m**4)
    a=np.mean(m2,0);b=np.mean(m4,0)
    U=lambda a,b:1-b.mean()/(3*a.mean()**2)
    u=U(a,b)
    jk=np.array([U(np.delete(a,i),np.delete(b,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    out[T]=(u,err,len(m2)*R);print(T,u,err,time.time()-t0,flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/o.json','w'))
