
import numpy as np, json
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
N=16;R=256
def run(T,sweeps,seed,burn=2000):
    rng=np.random.default_rng(seed)
    if T<2.3: s=np.ones((R,N,N),np.int8)*rng.choice(np.array([-1,1],np.int8),size=(R,1,1))
    else: s=rng.choice(np.array([-1,1],np.int8),size=(R,N,N))
    ii,jj=np.indices((N,N)); masks=[((ii+jj)%2==p) for p in (0,1)]
    m2=np.zeros(R);m4=np.zeros(R);n=0
    for t in range(burn+sweeps):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb
            acc=(x<=0)|(rng.random(s.shape)<np.exp(-2*x/T))
            s=np.where(acc&mk,-s,s)
        if t>=burn:
            m=s.sum((1,2))/N**2; m2+=m**2; m4+=m**4; n+=1
    return m2/n,m4/n,n
def U_jk(a,b):
    U=1-b.mean()/(3*a.mean()**2)
    ma=(a.sum()-a)/(R-1);mb=(b.sum()-b)/(R-1);jk=1-mb/(3*ma**2)
    return U,np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
res={}
for k,T in enumerate(Ts):
    a,b,n=run(T,8000,100+k)
    U,e=U_jk(a,b);res[T]=(U,e,n*R);print(T,U,e,flush=True)
json.dump(res,open('/work/res.json','w'))
