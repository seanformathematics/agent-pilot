import numpy as np, time
N=16
ii,jj=np.indices((N,N)); par=(ii+jj)%2
masks=[par==0, par==1]
def run(T,R,nsweep_eq,nmeas,seed,nblk=10):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    xs=np.arange(-4,5)
    tab=np.minimum(1.0,np.exp(-2.0*xs/T))
    def sweep(s):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x=(s*h).astype(np.int8)
            p=tab[x+4]
            flip=(rng.random(s.shape)<p)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nsweep_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    blk=[]
    per=nmeas//nblk
    b2=np.zeros(R); b4=np.zeros(R)
    for k in range(nmeas):
        s=sweep(s)
        m=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2; b2+=m2; b4+=m2*m2
        if (k+1)%per==0:
            blk.append(1-(b4.mean()/per)/(3*(b2.mean()/per)**2))
            b2[:]=0; b4[:]=0
    return S2/nmeas, S4/nmeas, blk
def binder(S2,S4):
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    # jackknife over replicas
    idx=np.arange(R)
    tot2=S2.sum(); tot4=S4.sum()
    loo=[]
    Uj=1-((tot4-S4)/(R-1))/(3*((tot2-S2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t=time.time()
S2,S4,blk=run(2.269,2000,2000,4000,11)
print("time",time.time()-t)
print("U,err",binder(S2,S4))
print("blocks",np.round(blk,4))
