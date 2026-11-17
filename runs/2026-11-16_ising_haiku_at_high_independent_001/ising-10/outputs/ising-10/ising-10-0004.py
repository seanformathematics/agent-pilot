import numpy as np, time, json
N=16
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
def sweep(s,expt,rng):
    for m in (mA,mB):
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nn
        acc = rng.random(s.shape,dtype=np.float32) < expt[dE+8]
        s[acc & m] *= -1
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=1000; burn=1000; nmeas=2000
out={}
t0=time.time()
for idx,T in enumerate(Ts):
    seed=1000+idx
    rng=np.random.default_rng(seed)
    expt=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)],dtype=np.float32)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    for k in range(burn): sweep(s,expt,rng)
    S2=np.zeros(R); S4=np.zeros(R)
    # also check halves for drift
    h2=[];
    for k in range(nmeas):
        sweep(s,expt,rng)
        m=s.reshape(R,-1).mean(1)
        m2=m*m
        S2+=m2; S4+=m2*m2
        if k==nmeas//2-1: Sh2=S2.copy(); Sh4=S4.copy()
    n=nmeas
    tot2=S2.sum(); tot4=S4.sum()
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    Uj=np.array([1-((tot4-S4[r])/((R-1)*n))/(3*((tot2-S2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # half-split consistency
    a2=Sh2.sum()/(R*nmeas/2); a4=Sh4.sum()/(R*nmeas/2)
    b2=(tot2-Sh2.sum())/(R*nmeas/2); b4=(tot4-Sh4.sum())/(R*nmeas/2)
    Uh1=1-a4/(3*a2**2); Uh2=1-b4/(3*b2**2)
    out[T]=dict(seed=seed,U=float(U),err=float(err),U_first_half=float(Uh1),U_second_half=float(Uh2))
    print(T, out[T], round(time.time()-t0,1), flush=True)
    json.dump(out,open('/work/results.json','w'),indent=1)
