import numpy as np, time, json
N=16; R=1024
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
EQ=2000; MEAS=10000; h=MEAS//2
out={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    SEED=1000+int(round(T*100))
    rng=np.random.default_rng(SEED)
    beta=1/T
    acc={4:np.exp(-beta*4),8:np.exp(-beta*8)}
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            u=rng.random(s.shape)
            flip=(dE<=0)|((dE==4)&(u<acc[4]))|((dE==8)&(u<acc[8]))
            flip&=m
            s[flip]*=-1
        return s
    t0=time.time()
    for _ in range(EQ): sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); H2=np.zeros((2,R)); H4=np.zeros((2,R))
    for k in range(MEAS):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; m4=m2*m2
        S2+=m2; S4+=m4
        H2[k//h]+=m2; H4[k//h]+=m4
    # per-replica U and pooled estimate; jackknife over replicas
    def Ufun(a2,a4): return 1-a4/(3*a2**2)
    U_pool=Ufun(S2.sum()/(R*MEAS), S4.sum()/(R*MEAS))
    # jackknife over replicas
    tot2=S2.sum(); tot4=S4.sum()
    jk=np.empty(R)
    for r in range(R):
        a2=(tot2-S2[r])/((R-1)*MEAS); a4=(tot4-S4[r])/((R-1)*MEAS)
        jk[r]=Ufun(a2,a4)
    U_err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    Ua=Ufun(H2[0].sum()/(R*h),H4[0].sum()/(R*h)); Ub=Ufun(H2[1].sum()/(R*h),H4[1].sum()/(R*h))
    out[T]=dict(U=float(U_pool),U_err=float(U_err),U_half1=float(Ua),U_half2=float(Ub),
                m2=float(S2.sum()/(R*MEAS)),m4=float(S4.sum()/(R*MEAS)),seed=SEED,time=time.time()-t0)
    np.savez(f'/work/sums_T{T:.2f}.npz',S2=S2,S4=S4,H2=H2,H4=H4)
    print(f"T={T} U={U_pool:.5f} err={U_err:.5f} halves={Ua:.5f},{Ub:.5f} <m2>={out[T]['m2']:.5f} t={out[T]['time']:.0f}s",flush=True)
json.dump({str(k):v for k,v in out.items()},open('/work/results.json','w'),indent=1)