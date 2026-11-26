import numpy as np, time, os
N=16
def run(T, R, nsweep, ntherm, seed, meas_every=2):
    rng=np.random.default_rng(seed)
    beta=1.0/T
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    def sweep(s):
        for c in (0,1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            p=np.exp(-2*beta*(s*h).astype(np.float64))
            flip=(rng.random(s.shape)<p)&masks[c]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(ntherm): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); nm=0
    for k in range(nsweep):
        s=sweep(s)
        if (k+1)%meas_every==0:
            m=s.reshape(R,-1).mean(1)
            m2=m*m; S2+=m2; S4+=m2*m2; nm+=1
    return S2/nm, S4/nm
def U_jack(a2,a4):
    R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    tot2=a2.sum(); tot4=a4.sum()
    b2=(tot2-a2)/(R-1); b4=(tot4-a4)/(R-1)
    Uj=1-b4/(3*b2*b2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
for T in [2.2,2.3,2.4]:
    t=time.time()
    a2,a4=run(T,1000,4000,2000,seed=int(T*1000)+7)
    U,err=U_jack(a2,a4)
    np.savez(f'/work/T{T}.npz',a2=a2,a4=a4)
    print(T, "U=%.5f err=%.5f"%(U,err), "<m2>=%.4f"%a2.mean(), "time %.1f"%(time.time()-t), flush=True)
