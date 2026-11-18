
import numpy as np, time, json
N=16;R=128
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
ii,jj=np.meshgrid(range(N),range(N),indexing='ij')
masks=[((ii+jj)%2==k)[None] for k in (0,1)]
def run(T,neq,nmeas,every=2):
    s=np.ones((R,N,N),np.int8) if T<2.2 else rng.choice(np.array([-1,1],np.int8),(R,N,N))
    p={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep():
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random(s.shape)<np.exp(-dE/T))
            s[...]=np.where(acc&m,-s,s)
    for _ in range(neq): sweep()
    m2=[];m4=[]
    for k in range(nmeas):
        for _ in range(every): sweep()
        m=s.mean((1,2),dtype=float); m2.append(m**2); m4.append(m**4)
    return np.array(m2),np.array(m4)
t=time.time()
res={}
for T in Ts:
    m2,m4=run(T,2000,3000)
    a2=m2.mean(0);a4=m4.mean(0)  # per replica
    U=1-a4.mean()/(3*a2.mean()**2)
    b=np.random.default_rng(1).integers(0,R,(400,R))
    Ub=1-a4[b].mean(1)/(3*a2[b].mean(1)**2)
    res[T]=(U,Ub.std(),m2.size)
    print(T,U,Ub.std(),time.time()-t,flush=True)
