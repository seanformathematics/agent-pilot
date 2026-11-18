
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=((np.indices((N,N)).sum(0))%2==0)
masks=[mask[None],~mask[None]]
def run(T,seed,neq=2000,nm=8000):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),dtype=np.int8)
    m2=np.zeros(R);m4=np.zeros(R)
    for it in range(neq+nm):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=np.exp(-2.0*(s*nb)/T)
            acc=(rng.random(s.shape)<p)&mk
            s=np.where(acc,-s,s)
        if it>=neq:
            m=s.sum((1,2))/(N*N)
            m2+=m**2;m4+=m**4
    m2/=nm;m4/=nm
    U=1-m4.mean()/(3*m2.mean()**2)
    Uj=np.array([1-np.delete(m4,i).mean()/(3*np.delete(m2,i).mean()**2) for i in range(R)])
    err=np.sqrt((R-1)/R*((Uj-Uj.mean())**2).sum())
    return float(U),float(err)
res={}
for k,T in enumerate(Ts):
    res[T]=run(T,100+k); print(T,res[T],flush=True)
json.dump(res,open('/work/res.json','w'))
