
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
mask=((np.indices((N,N)).sum(0))%2==0)
masks=[mask,~mask]
out={}
t0=time.time()
for T in Ts:
    s=np.ones((R,N,N),np.int8)
    s[R//2:]=rng.choice(np.array([-1,1],np.int8),size=(R-R//2,N,N))
    pr={k:np.exp(-2*k/T) for k in (-4,0,4)}
    # lookup by dE=2*s*nn: values -8..8
    tab=np.minimum(1,np.exp(-2*np.arange(-4,5)*1.0/T))  # index nn*s+4 -> dE=2*(nn*s)
    def sweep():
        for mk in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=(nn*s+4)
            acc=rng.random(s.shape)<tab[x]
            f=acc&mk
            s[f]*=-1
    for _ in range(1000): sweep()
    M2=[];M4=[]
    for i in range(8000):
        sweep()
        m=s.sum((1,2),dtype=np.int32)/N**2
        M2.append(m**2);M4.append(m**4)
    a=np.mean(M2,0);b=np.mean(M4,0)  # per replica
    U=1-b.mean()/(3*a.mean()**2)
    jk=[1-(b.sum()-b[i])/(R-1)/(3*((a.sum()-a[i])/(R-1))**2) for i in range(R)]
    err=np.sqrt((R-1)/R*np.sum((np.array(jk)-np.mean(jk))**2))
    out[T]=(U,err);print(T,U,err,time.time()-t0,flush=True)
json.dump(out,open('/work/o.json','w'))
