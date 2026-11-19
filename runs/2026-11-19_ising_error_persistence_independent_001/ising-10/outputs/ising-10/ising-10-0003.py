import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=2000; NEQ=2000; NMEAS=4000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    # acceptance table for dE in {4,8} (dE<=0 always accepted)
    p4=np.float32(np.exp(-4/T)); p8=np.float32(np.exp(-8/T))
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            u=rng.random(s.shape,dtype=np.float32)
            acc=(dE<=0)|((dE==4)&(u<p4))|((dE==8)&(u<p8))
            s[acc&m]*=-1
    for k in range(NEQ): sweep()
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R); S2b=np.zeros(R)
    for k in range(NMEAS):
        sweep()
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    return S2/NMEAS,S4/NMEAS,S1/NMEAS
def jack(S2,S4):
    R=len(S2)
    tot2=S2.sum(); tot4=S4.sum()
    U_full=1-(tot4/R)/(3*(tot2/R)**2)
    Ui=np.empty(R)
    for r in range(R):
        a2=(tot2-S2[r])/(R-1); a4=(tot4-S4[r])/(R-1)
        Ui[r]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    return U_full,err
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    S2,S4,S1=run(T,seed)
    np.savez(f'/work/N16_T{T}.npz',S2=S2,S4=S4,S1=S1,seed=seed)
    U,e=jack(S2,S4)
    print(T,U,e,time.time()-t,flush=True)