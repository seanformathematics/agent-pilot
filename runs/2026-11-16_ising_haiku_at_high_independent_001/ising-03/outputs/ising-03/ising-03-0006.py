
import numpy as np
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
def run_series(R,T,nburn,nmeas,seed):
    rng=np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p=np.array([1.0 if 2*x<=0 else np.exp(-2*x/T) for x in range(-4,5)])
    def sweep(s):
        for mask in (mA,mB):
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x = (s*nn).astype(np.int8)+4
            acc = rng.random((R,N,N)) < p[x]
            s = np.where(acc & mask,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    m2s=np.zeros(R); m4s=np.zeros(R)
    for k in range(nmeas):
        s=sweep(s)
        m=s.mean(axis=(1,2)); m2=m*m
        m2s+=m2; m4s+=m2*m2
    return m2s/nmeas, m4s/nmeas
for T,seed in [(2.0,999),(2.3,998),(2.6,997)]:
    a2,a4=run_series(1000,T,1000,4000,seed)
    print(T, 1-a4.mean()/(3*a2.mean()**2))
