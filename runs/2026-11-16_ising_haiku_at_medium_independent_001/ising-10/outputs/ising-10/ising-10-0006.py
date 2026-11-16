
import numpy as np, time
N=16
def run(T, R, nequil, nmeas, every, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    ptab=np.zeros(9,dtype=np.float32)
    for x in [-4,-2,0,2,4]: ptab[x+4]= 1.0 if x<=0 else np.exp(-2*x/T)
    def sweep(s):
        for m in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            p=ptab[(s*nn).astype(np.int16)+4]
            flip=m[None]&(rng.random((R,N,N),dtype=np.float32)<p)
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    for k in range(nmeas):
        for _ in range(every): s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; M2+=m2; M4+=m2*m2; cnt+=1
    return M2/cnt, M4/cnt
def ubin(M2,M4):
    return 1-M4.mean()/(3*M2.mean()**2)
def jack(M2,M4,nb=50):
    R=len(M2); idx=np.arange(R)%nb
    U=ubin(M2,M4)
    Uj=np.array([ubin(M2[idx!=b],M4[idx!=b]) for b in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    return U,err
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
t0=time.time()
for i,T in enumerate(Ts):
    M2,M4=run(T,1000,1000,2000,5,200+i)
    U,e=jack(M2,M4)
    print(T,U,e,time.time()-t0,flush=True)
    np.savez(f"/work/rep_{T}.npz",M2=M2,M4=M4)
