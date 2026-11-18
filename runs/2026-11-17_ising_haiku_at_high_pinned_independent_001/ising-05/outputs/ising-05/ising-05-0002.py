
import numpy as np, time, json, os
N=16
def run(T, R, neq, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    pr={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            u=rng.random(s.shape)
            p=np.where(dE<=0,1.0,np.where(dE==4,pr[4],np.where(dE==8,pr[8],0.0)))
            flip=(u<p)&mk
            s[flip]*=-1
    for _ in range(neq): sweep()
    out=[]
    for half in range(2):
        m2=np.zeros(R); m4=np.zeros(R)
        for _ in range(nmeas//2):
            sweep()
            m=s.reshape(R,-1).mean(axis=1)
            m2+=m**2; m4+=m**4
        out.append((m2/(nmeas//2), m4/(nmeas//2)))
    return out
Ts=[1.5,1.6,1.7,1.8,1.9]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    out=run(T,1000,3000,10000,1000+k)
    (a2,a4),(b2,b4)=out
    # per-replica U
    Ua=1-a4/(3*a2**2); Ub=1-b4/(3*b2**2)
    Uall=1-(a4+b4)/2/(3*((a2+b2)/2)**2)
    res[T]=dict(U=float(Uall.mean()), err=float(Uall.std(ddof=1)/np.sqrt(len(Uall))),
                Ua=float(Ua.mean()), Ub=float(Ub.mean()))
    print(T, res[T], round(time.time()-t0,1), flush=True)
json.dump(res,open('/work/batch1.json','w'))
