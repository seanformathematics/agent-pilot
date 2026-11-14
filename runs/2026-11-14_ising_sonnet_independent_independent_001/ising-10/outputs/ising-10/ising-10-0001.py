
import numpy as np, time
N=16;R=100
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rng=np.random.default_rng(12345)
ii,jj=np.indices((N,N));masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T,eq,ns):
    s=np.where(rng.random((R,N,N))<.5,1,-1).astype(np.int8)
    m2=np.zeros(R);m4=np.zeros(R)
    pr=np.exp(-2*np.arange(-4,5)*1.0/T)  # index by s*nb+4
    for t in range(eq+ns):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb
            acc=(x<=0)|(rng.random(s.shape)<np.exp(-2*x/T))
            s=np.where(mk&acc,-s,s)
        if t>=eq:
            m=s.sum((1,2))/N**2
            m2+=m**2;m4+=m**4
    return m2/ns,m4/ns
out={}
t0=time.time()
for T in Ts:
    a,b=run(T,1000,20000)
    U=lambda a,b:1-b.mean()/(3*a.mean()**2)
    bs=[];
    for _ in range(300):
        k=rng.integers(0,R,R);bs.append(U(a[k],b[k]))
    out[T]=(U(a,b),np.std(bs));print(T,out[T],time.time()-t0,flush=True)
