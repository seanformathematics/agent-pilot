
import numpy as np, json, os
N=16
def run(T,R=1000,neq=3000,nmeas=5000,seed=0,start='ordered'):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),np.int8) if start=='ordered' else rng.choice(np.array([-1,1],np.int8),(R,N,N))
    ii,jj=np.indices((N,N)); masks=[((ii+jj)%2==p) for p in (0,1)]
    pt=np.array([min(1,np.exp(-2*x/T)) for x in range(-4,5)],np.float32)
    m2=np.zeros(R);m4=np.zeros(R)
    for t in range(neq+nmeas):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            flip=(rng.random((R,N,N),dtype=np.float32)<pt[s*h+4])&mk
            s[flip]*=-1
        if t>=neq:
            m=s.sum((1,2),dtype=np.int32)/N**2
            m2+=m*m; m4+=m**4
    m2/=nmeas;m4/=nmeas
    U=1-m4.mean()/(3*m2.mean()**2)
    G=50;jk=[]
    for g in np.array_split(np.arange(R),G):
        keep=np.ones(R,bool);keep[g]=False
        jk.append(1-m4[keep].mean()/(3*m2[keep].mean()**2))
    jk=np.array(jk);err=np.sqrt((G-1)/G*((jk-jk.mean())**2).sum())
    return float(U),float(err),R*nmeas
f='/work/res.json'
res=json.load(open(f)) if os.path.exists(f) else {}
import time;t0=time.time()
for k,T in enumerate([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]):
    if str(T) in res: continue
    if time.time()-t0>480: break
    res[str(T)]=run(T,seed=1000+k)
    json.dump(res,open(f,'w')); print(T,res[str(T)],time.time()-t0,flush=True)
