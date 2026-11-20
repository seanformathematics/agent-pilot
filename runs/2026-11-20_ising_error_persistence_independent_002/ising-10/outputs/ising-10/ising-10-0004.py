
import numpy as np, time, os
N=16; R=1000
NEQ=5000; NSW=60000; BL=500
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    pt=np.exp(-2*np.arange(-4,5)/T).astype(np.float32)
    def sweep():
        for mk in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            delta=(s*nn).astype(np.int8)
            r=rng.random(s.shape,dtype=np.float32)
            acc=(delta<=0)|(r<pt[delta+4])
            s[acc&mk]*=-1
    for _ in range(NEQ): sweep()
    nb=NSW//BL
    S2=np.zeros((nb,R)); S4=np.zeros((nb,R)); S1=np.zeros((nb,R))
    for b in range(nb):
        a2=np.zeros(R); a4=np.zeros(R); a1=np.zeros(R)
        for k in range(BL):
            sweep()
            m=s.sum(axis=(1,2))/(N*N)
            a2+=m*m; a4+=m**4; a1+=np.abs(m)
        S2[b]=a2/BL; S4[b]=a4/BL; S1[b]=a1/BL
    np.savez(f'/work/T{T}.npz',S2=S2,S4=S4,S1=S1,T=T,R=R,NEQ=NEQ,BL=BL,seed=seed)
    print("T",T,"U pooled",1-S4.mean()/(3*S2.mean()**2), flush=True)
t0=time.time()
run(2.0,102)
print("elapsed",time.time()-t0,flush=True)
run(2.1,103)
print("elapsed",time.time()-t0,flush=True)
