import numpy as np, time
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
    H=NMEAS//2
    S2=np.zeros((2,R)); S4=np.zeros((2,R))
    for t in range(NMEAS):
        s=sweep(s,T,rng)
        m=s.reshape(R,-1).mean(axis=1); m2=m*m
        S2[t//H]+=m2; S4[t//H]+=m2*m2
    return S2/H, S4/H
t0=time.time()
for k,T in enumerate([2.2,2.4]):
    a2,a4=run(T,300+k)
    np.savez(f'/work/T{T}.npz',m2=a2.mean(0),m4=a4.mean(0),h2=a2,h4=a4)
    U=lambda x2,x4: 1-x4.mean()/(3*x2.mean()**2)
    print(T,U(a2.mean(0),a4.mean(0)),'halves',U(a2[0],a4[0]),U(a2[1],a4[1]),time.time()-t0,flush=True)
