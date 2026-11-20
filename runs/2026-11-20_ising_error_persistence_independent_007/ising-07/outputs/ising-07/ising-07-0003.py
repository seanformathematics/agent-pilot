
import numpy as np, time, os
N=16
def run(T, R, nsweep, nequil, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep(s):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            prob=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            u=rng.random((R,N,N))
            flip=(u<prob)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    h2=np.zeros(R); h4=np.zeros(R)
    for k in range(nsweep):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        if k==nsweep//2: pass
        m2+=m*m; m4+=m**4; cnt+=1
        if k<nsweep//2: h2+=m*m; h4+=m**4
    return m2/cnt, m4/cnt, h2/(nsweep//2), h4/(nsweep//2)
def jack(m2,m4):
    R=len(m2); S2=m2.sum(); S4=m4.sum()
    a=(S2-m2)/(R-1); b=(S4-m4)/(R-1)
    Ujk=1-b/(3*a*a)
    U=1-m4.mean()/(3*m2.mean()**2)
    err=np.sqrt((R-1)/R*np.sum((Ujk-Ujk.mean())**2))
    return U,err
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    t=time.time()
    m2,m4,h2,h4=run(T,1000,8000,3000,seed=7919+int(T*1000))
    np.savez(f'/work/B_T{T}.npz',m2=m2,m4=m4)
    U,e=jack(m2,m4)
    U1,_=jack(h2,h4); U2,_=jack(2*m2-h2,2*m4-h4)
    print(T,round(U,6),e,'halves',round(U1,5),round(U2,5),round(time.time()-t,1),flush=True)
