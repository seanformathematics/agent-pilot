import numpy as np, time
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def go(T,EQ=2000,MEAS=18000,seed=None):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    p4=np.exp(-4/T); p8=np.exp(-8/T)
    def sweep(s):
        for m in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sh=s*h
            p=np.where(sh<=0,1.0,np.where(sh==2,p4,np.where(sh==4,p8,0.0)))
            flip=(rng.random((R,N,N))<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(EQ): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    for k in range(MEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2
    n=MEAS; tot2=S2.sum(); tot4=S4.sum()
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    jk=np.array([1-((tot4-S4[r])/((R-1)*n))/(3*((tot2-S2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # also a split-half check: two halves of replicas
    h1=slice(0,R//2); h2=slice(R//2,R)
    def Uof(sl): return 1-(S4[sl].sum()/(len(S4[sl])*n))/(3*(S2[sl].sum()/(len(S2[sl])*n))**2)
    print(f"T={T} seed={seed} U={U:.5f} jk_err={err:.5f} halves: {Uof(h1):.5f} {Uof(h2):.5f}")
go(2.1,seed=777)
go(2.3,seed=778)