
import numpy as np, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=((np.indices((N,N)).sum(0))%2==0)
masks=[mask,~mask]
def run(T,seed,neq,nmeas,every):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            p=np.exp(-2*s*nb/T)
            s[(rng.random(s.shape)<p)&mk]*=-1
    for _ in range(neq): sweep()
    m2=np.zeros(R);m4=np.zeros(R)
    for i in range(nmeas):
        for _ in range(every): sweep()
        m=s.sum((1,2))/N**2
        m2+=m**2;m4+=m**4
    return m2/nmeas,m4/nmeas,nmeas
res={}
for k,T in enumerate(Ts):
    a,b,n=run(T,100+k,1000,3000,5)
    U=1-b.mean()/(3*a.mean()**2)
    jk=[1-np.delete(b,i).mean()/(3*np.delete(a,i).mean()**2) for i in range(R)]
    err=np.sqrt((R-1)/R*np.sum((np.array(jk)-np.mean(jk))**2))
    res[T]=(U,err,n*R);print(T,U,err,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res.json','w'))
