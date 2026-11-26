
import numpy as np, time, os
N=16; R=1000
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
def run(T,seed,burn=2000,meas=10000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(burn): sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); SA=np.zeros(R); cnt=0
    for k in range(meas):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2; SA+=np.abs(m); cnt+=1
    return S2/cnt,S4/cnt,SA/cnt
for idx,T in enumerate([2.2,2.3,2.4]):
    t=time.time()
    m2,m4,ma=run(T,seed=200+idx)
    np.savez(f'/work/T{T}_N16.npz',m2=m2,m4=m4,ma=ma)
    U_r=1-m4/(3*m2**2)
    print(T, 'U=',U_r.mean(), 'err=',U_r.std(ddof=1)/np.sqrt(R), '<|m|>=',ma.mean(), 'time',time.time()-t)
