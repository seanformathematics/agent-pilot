
import numpy as np, json, time
N=16;R=200
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
ii,jj=np.indices((N,N));masks=[((ii+jj)%2==p) for p in (0,1)]
res={}
for T in Ts:
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ex={k:np.exp(-2*k/T) for k in (-4,-2,0,2,4)}
    def sweep():
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb  # dE=2x
            p=np.where(x<=0,1.0,np.exp(-2*x/T))
            acc=(rng.random(s.shape)<p)&m
            s[acc]*=-1
    for _ in range(1000): sweep()
    M=5000;m2=np.zeros(R);m4=np.zeros(R)
    for _ in range(M):
        sweep()
        m=s.sum((1,2),dtype=np.int64)/N**2
        m2+=m**2;m4+=m**4
    m2/=M;m4/=M
    U=1-m4.mean()/(3*m2.mean()**2)
    Uj=[]
    for r in range(R):
        k=np.arange(R)!=r
        Uj.append(1-m4[k].mean()/(3*m2[k].mean()**2))
    Uj=np.array(Uj);err=np.sqrt((R-1)/R*((Uj-Uj.mean())**2).sum())
    res[T]=(U,err);print(T,U,err,flush=True)
json.dump(res,open('/work/res.json','w'))
