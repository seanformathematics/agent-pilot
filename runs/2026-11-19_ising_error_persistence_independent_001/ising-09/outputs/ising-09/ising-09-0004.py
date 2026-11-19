import numpy as np, time
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def go(T,EQ=2000,MEAS=18000,seed=None):
    seed = seed if seed is not None else 1000+int(round(T*100))
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
    t0=time.time()
    for _ in range(EQ): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); Q=[]
    for k in range(MEAS):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2
        if (k+1)%1000==0: Q.append(np.mean(1-(S4/(k+1))/(3*(S2/(k+1))**2)))
    n=MEAS; tot2=S2.sum(); tot4=S4.sum()
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    jk=np.array([1-((tot4-S4[r])/((R-1)*n))/(3*((tot2-S2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(f"T={T} U={U:.5f} jk_err={err:.5f} time={time.time()-t0:.1f}s  running U per 1000 sweeps (replica-mean): {np.round(Q,4)}")
    np.savez(f"/work/res_T{T}.npz",S2=S2,S4=S4)
    return U,err
for T in (2.0,2.1): go(T)