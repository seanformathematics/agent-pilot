
import numpy as np, time
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        acc&=masks[c]
        s[acc]*=-1
    return s
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}_N16.npz'); m2,m4=d['m2'],d['m4']
    U_r=1-m4/(3*m2**2)
    Upool=1-m4.mean()/(3*m2.mean()**2)
    print(T,'mean_r',U_r.mean(),'pooled',Upool,'std_r',U_r.std(ddof=1))
# independent check: cold start, different seed, R=400, burn 1000 + meas 4000 (weakly different protocol)
R=400
for T in [1.9,2.3]:
    rng=np.random.default_rng(999)
    s=np.ones((R,N,N),dtype=np.int8)
    for _ in range(1000): sweep(s,T,rng)
    S2=np.zeros(R);S4=np.zeros(R);cnt=0
    for k in range(4000):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1); m2=m*m
        S2+=m2;S4+=m2*m2;cnt+=1
    m2=S2/cnt;m4=S4/cnt; U_r=1-m4/(3*m2**2)
    print('check',T,U_r.mean(),U_r.std(ddof=1)/np.sqrt(R))
