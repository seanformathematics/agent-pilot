import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=2000; NTH=3000; NMEAS=6000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def sweep(s,T,rng):
    for mk in (mask0,mask1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float64)
        dE=2*s*h
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(rng.random(s.shape)<p)&mk
        s=np.where(acc,-s,s).astype(np.int8)
    return s
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    for _ in range(NTH): s=sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R)
    for _ in range(NMEAS):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2/NMEAS, S4/NMEAS
out={}
t0=time.time()
for k,T in enumerate([2.3]):
    m2,m4=run(T,100+k)
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4)
    U=1-m4.mean()/(3*m2.mean()**2)
    print(T,U,time.time()-t0)
