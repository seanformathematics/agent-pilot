import numpy as np, time, json
N=16
def run(T, R, neq, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p)[None] for p in (0,1)]
    pA=np.exp(-4/T); pB=np.exp(-8/T)
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sn=s*nb
            p=np.where(sn<=0,1.0,np.where(sn==2,pA,pB))
            r=rng.random(s.shape)
            flip=m & (r<p)
            s[flip]*=-1
    for _ in range(neq): sweep()
    S2=np.zeros(R); S4=np.zeros(R)
    blk=[]; b2=np.zeros(R); b4=np.zeros(R); bn=0
    absm=[]
    for k in range(nmeas):
        sweep()
        m=s.reshape(R,-1).mean(1)
        S2+=m**2; S4+=m**4; b2+=m**2; b4+=m**4; bn+=1
        absm.append(np.abs(m).mean())
        if bn==500:
            blk.append(1-b4.mean()/(3*b2.mean()**2)); b2[:]=0; b4[:]=0; bn=0
    return S2/nmeas,S4/nmeas,blk,absm
def U_and_err(a2,a4):
    R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    Uj=np.empty(R)
    tot2=a2.sum(); tot4=a4.sum()
    for i in range(R):
        m2=(tot2-a2[i])/(R-1); m4=(tot4-a4[i])/(R-1)
        Uj[i]=1-m4/(3*m2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
res={}
for T,seed in zip([1.9,2.0,2.1,2.2,2.3,2.4],[201,202,203,204,205,206]):
    t0=time.time()
    a2,a4,blk,absm=run(T,1000,3000,3000,seed)
    U,err=U_and_err(a2,a4)
    np.savez(f"/work/T{T}_prod.npz",a2=a2,a4=a4)
    res[T]=(U,err)
    print(f"T={T} U={U:.6f} err={err:.6f} time={time.time()-t0:.1f}")
    print("  block U:",np.round(blk,4))
    print("  <|m|> first/last 500 meas:",np.round(np.mean(absm[:500]),4),np.round(np.mean(absm[-500:]),4))
