
import numpy as np, time
N=16
def run(T,R,n_eq,n_meas,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    # acceptance table indexed by sn+4, sn = s*nb in {-4..4}
    pacc=np.array([1.0 if x<=0 else np.exp(-2*x/T) for x in range(-4,5)])
    def sweep(s):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=(s*nb+4).astype(np.int8)
            acc=rng.random((R,N,N))<pacc[sn]
            s=np.where(acc&m,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(n_meas):
        s=sweep(s)
        m=s.sum((1,2))/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2
    S2/=n_meas; S4/=n_meas
    return S2,S4
def binder(S2,S4):
    return 1-S4.mean()/(3*S2.mean()**2)
t=time.time()
S2,S4=run(2.3,500,200,200,1)
print('time',time.time()-t)
t=time.time()
S2,S4=run(2.269,500,500,500,2)
print('time',time.time()-t, 'U',binder(S2,S4))
