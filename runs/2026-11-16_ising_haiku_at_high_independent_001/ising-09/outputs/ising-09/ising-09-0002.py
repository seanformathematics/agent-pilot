
import numpy as np, time, os
N=16
def run(T, R, nequil, nblocks, blen, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    def sweep():
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            acc=(dE<=0)|(rng.random((R,N,N))<np.exp(-np.maximum(dE,0)/T))
            s[:]=np.where(masks[p]&acc,-s,s)
    for _ in range(nequil): sweep()
    M2=np.zeros((nblocks,R)); M4=np.zeros((nblocks,R))
    for b in range(nblocks):
        a2=np.zeros(R); a4=np.zeros(R)
        for _ in range(blen):
            sweep()
            m=s.sum(axis=(1,2))/(N*N)
            a2+=m*m; a4+=m**4
        M2[b]=a2/blen; M4[b]=a4/blen
    return M2,M4
temps=[1.5,1.6,1.7,1.8]
t0=time.time()
for k,T in enumerate(temps):
    M2,M4=run(T,1000,2000,10,1000,seed=1000+k)
    np.savez(f'/work/run_T{T}.npz',M2=M2,M4=M4,seed=1000+k)
    m2=M2.mean(0); m4=M4.mean(0)
    U=1-m4/(3*m2**2)
    print(T, "U=",U.mean(), "sem=",U.std(ddof=1)/np.sqrt(len(U)), "blockU:", [round(1-M4[b].mean()/(3*M2[b].mean()**2),4) for b in range(10)], round(time.time()-t0,1), flush=True)
