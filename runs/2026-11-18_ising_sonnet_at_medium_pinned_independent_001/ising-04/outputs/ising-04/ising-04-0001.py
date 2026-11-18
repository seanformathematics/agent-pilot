
import numpy as np, json, time
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
masks=[mask,~mask]
res={}
for T in Ts:
    t0=time.time()
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # Boltzmann lookup for dE=2*s*nn : nn*s in {-4,-2,0,2,4}
    p={k:np.exp(-2*k/T) for k in (-4,-2,0,2,4)}
    def sweep():
        for m in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=(s*nn).astype(np.int8)
            acc=rng.random(s.shape)<np.exp(-2.0*x/T)
            s[m[None]&acc|False] *= 0  if False else 1
            flip=acc&m[None]
            s[flip]*=-1
    for _ in range(1000): sweep()
    m2=[];m4=[]
    for i in range(6000):
        sweep()
        if i%2==0:
            m=s.sum((1,2))/N**2
            m2.append(m**2);m4.append(m**4)
    m2=np.array(m2);m4=np.array(m4)  # (t,R)
    a2=m2.mean(0);a4=m4.mean(0)
    U=1-a4.mean()/(3*a2.mean()**2)
    G=16;g=R//G
    jk=[]
    for j in range(G):
        idx=np.ones(R,bool);idx[j*g:(j+1)*g]=False
        jk.append(1-a4[idx].mean()/(3*a2[idx].mean()**2))
    jk=np.array(jk);err=np.sqrt((G-1)/G*((jk-jk.mean())**2).sum())
    res[T]=(U,err,m2.size)
    print(T,U,err,time.time()-t0,flush=True)
json.dump({str(k):v for k,v in res.items()},open('/work/res.json','w'))
