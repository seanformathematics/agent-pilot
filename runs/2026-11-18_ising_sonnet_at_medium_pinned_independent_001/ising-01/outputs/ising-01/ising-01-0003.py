
import numpy as np, json
N=16;R=128
def run(T,seed,neq=3000,nm=20000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.indices((N,N));masks=[((ii+jj)%2==k) for k in (0,1)]
    def sweep():
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            acc=rng.random(s.shape)<np.exp(-2*s*nb/T)
            s[:]=np.where(m&acc,-s,s)
    for _ in range(neq): sweep()
    m2=np.zeros(R);m4=np.zeros(R)
    for _ in range(nm):
        sweep()
        m=s.sum((1,2))/N**2
        m2+=m**2;m4+=m**4
    m2/=nm;m4/=nm
    Uj=np.array([1-m4[np.arange(R)!=r].mean()/(3*m2[np.arange(R)!=r].mean()**2) for r in range(R)])
    err=np.sqrt((R-1)/R*((Uj-Uj.mean())**2).sum())
    return 1-m4.mean()/(3*m2.mean()**2),err
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
for k,T in enumerate(Ts):
    u,e=run(T,100+k);res[T]=(float(u),float(e));print(T,u,e,flush=True)
json.dump(res,open('/work/res.json','w'))
