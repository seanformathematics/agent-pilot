import numpy as np, time, os
N=16; R=1024
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
EQ=2000; MEAS=10000
for T in [1.9,2.0,2.1,2.2,2.4]:
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
    M2=np.zeros((MEAS,R)); M4=np.zeros((MEAS,R))
    for k in range(MEAS):
        sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        M2[k]=m*m; M4[k]=m**4
    np.savez(f'/work/raw_T{T:.2f}.npz',M2=M2.astype(np.float32),M4=M4.astype(np.float32),T=T,seed=SEED)
    U_r=1-M4.mean(0)/(3*M2.mean(0)**2)
    h=MEAS//2
    Ua=1-M4[:h].mean()/(3*M2[:h].mean()**2); Ub=1-M4[h:].mean()/(3*M2[h:].mean()**2)
    print(f"T={T} time={time.time()-t0:.1f} U={1-M4.mean()/(3*M2.mean()**2):.5f} replicaSE={U_r.std()/np.sqrt(R):.6f} halves {Ua:.5f} {Ub:.5f}",flush=True)