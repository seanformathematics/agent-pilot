
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=(np.add.outer(np.arange(N),np.arange(N))%2==0)
def run(T,seed,eq=1000,ns=6000):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),np.int8)
    if T>2.3: s=rng.choice(np.array([-1,1],np.int8),size=(R,N,N))
    p={k:np.exp(-2*k/T) for k in (-4,-2,0,2,4)}
    lut=np.array([min(1,np.exp(-2*k/T)) for k in range(-4,5)])  # index k+4 for k=s*nbr
    m2=np.zeros(R);m4=np.zeros(R)
    for it in range(eq+ns):
        for par in (0,1):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            k=s*nb
            acc=rng.random(s.shape)<lut[k+4]
            mk=mask if par==0 else ~mask
            s=np.where(acc&mk,-s,s)
        if it>=eq:
            m=s.sum((1,2))/N**2
            m2+=m**2;m4+=m**4
    m2/=ns;m4/=ns
    return m2,m4
out={}
t0=time.time()
for i,T in enumerate(Ts):
    m2,m4=run(T,1000+i)
    U=1-m4.mean()/(3*m2.mean()**2)
    j=[1-np.delete(m4,r).mean()/(3*np.delete(m2,r).mean()**2) for r in range(R)]
    err=np.sqrt((R-1)/R*np.sum((np.array(j)-np.mean(j))**2))
    out[T]=(U,err);print(T,U,err,time.time()-t0,flush=True)
json.dump(out,open('/work/r.json','w'))
