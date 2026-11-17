import numpy as np, time
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
    M2=np.zeros((R,)); M4=np.zeros((R,))
    for _ in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1); m2=m*m
        M2+=m2; M4+=m2*m2
    return M2/n_meas, M4/n_meas
def binder(S2,S4):
    # S2,S4: per-replica averages
    return 1-S4.mean()/(3*S2.mean()**2)
def jack(S2,S4):
    R=len(S2); tot2=S2.sum(); tot4=S4.sum()
    Uj=1-((tot4-S4)/(R-1))/(3*((tot2-S2)/(R-1))**2)
    U=binder(S2,S4)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t=time.time()
S2,S4=simulate(2.27,1000,1000,3000,11,False)
U,e=jack(S2,S4)
print("T=2.27 U=%.5f err=%.5f time=%.1f"%(U,e,time.time()-t))
# split into halves in replicas for consistency check
U1,_=jack(S2[:500],S4[:500]); U2,_=jack(S2[500:],S4[500:])
print("half replicas:",U1,U2)
