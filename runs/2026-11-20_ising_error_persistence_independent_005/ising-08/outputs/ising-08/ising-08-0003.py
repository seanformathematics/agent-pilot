import numpy as np, time, os
N=16; R=400
def run(T, seed, nequil, nmeas):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    k=np.arange(-4,5)
    accT=np.minimum(1.0,np.exp(-2*k/T))
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            idx=(s*nb).astype(np.int16)+4
            p=accT[idx]
            flip=(rng.random(s.shape)<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    M2=np.zeros((nmeas,R)); M4=np.zeros((nmeas,R)); AM=np.zeros((nmeas,R))
    for t in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        M2[t]=m*m; M4[t]=m**4; AM[t]=np.abs(m)
    return M2,M4,AM
def U_est(M2,M4):
    a2=M2.mean(); a4=M4.mean()
    return 1-a4/(3*a2*a2)
def jack(M2,M4):
    # jackknife over chains (columns)
    Rr=M2.shape[1]
    S2=M2.sum(axis=0); S4=M4.sum(axis=0); n=M2.shape[0]
    tot2=S2.sum(); tot4=S4.sum(); Ntot=n*Rr
    Uf=1-(tot4/Ntot)/(3*(tot2/Ntot)**2)
    Us=np.empty(Rr)
    for r in range(Rr):
        a2=(tot2-S2[r])/((Rr-1)*n); a4=(tot4-S4[r])/((Rr-1)*n)
        Us[r]=1-a4/(3*a2*a2)
    err=np.sqrt((Rr-1)/Rr*np.sum((Us-Us.mean())**2))
    return Uf,err
for T in [1.9,2.0]:
    t0=time.time()
    M2,M4,AM=run(T,seed=int(T*1000)+7,nequil=2000,nmeas=20000)
    np.savez(f'/work/T{T}.npz',M2=M2,M4=M4,AM=AM)
    Uf,err=jack(M2,M4)
    # equilibration check: <|m|> first vs last quarter
    q=M2.shape[0]//4
    print(T,'U=%.5f err=%.5f'%(Uf,err),'<|m|> early %.4f late %.4f'%(AM[:q].mean(),AM[-q:].mean()),'time %.1f'%(time.time()-t0))
