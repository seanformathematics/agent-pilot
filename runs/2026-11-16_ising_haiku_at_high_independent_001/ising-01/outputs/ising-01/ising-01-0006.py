import numpy as np, time, json
def simulate(T,R,n_eq,n_meas,seed,start_ordered):
    N=16
    rng=np.random.default_rng(seed)
    if start_ordered:
        s=np.where(rng.random((R,1,1))<0.5,1,-1).astype(np.int8)*np.ones((R,N,N),np.int8)
    else:
        s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    d=np.array([-4,-2,0,2,4]); ptab=np.minimum(1.0,np.exp(-2.0*d/T))
    me=((np.arange(N)[:,None]+np.arange(N)[None,:])%2==0); masks=[me,~me]
    def sweep(s):
        for mk in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            delta=(s*nn).astype(np.int8)
            p=ptab[((delta+4)>>1)]
            acc=(rng.random((R,N,N))<p)&mk
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq): s=sweep(s)
    M2=np.zeros(R); M4=np.zeros(R)
    for _ in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1); m2=m*m
        M2+=m2; M4+=m2*m2
    return M2/n_meas, M4/n_meas
def jack(S2,S4):
    R=len(S2); tot2=S2.sum(); tot4=S4.sum()
    U=1-S4.mean()/(3*S2.mean()**2)
    Uj=1-((tot4-S4)/(R-1))/(3*((tot2-S2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
t0=time.time()
for i,T in enumerate(Ts):
    S2,S4=simulate(T,1000,1000,3000,1000+i,T<2.0)
    U,e=jack(S2,S4)
    res[T]=dict(U=float(U),err=float(e),n=1000*3000)
    print("T=%.2f U=%.5f err=%.5f  t=%.0f"%(T,U,e,time.time()-t0),flush=True)
json.dump(res,open("/work/results.json","w"),indent=1)
