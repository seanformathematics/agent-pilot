import numpy as np, time, os
N=16
i,j=np.indices((N,N)); M0=((i+j)%2==0); M1=~M0
def run(T,R,seed,nburn,nmeas):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    tab=np.zeros(17)
    for dE in range(-8,9,4): tab[dE+8]=min(1.0,np.exp(-dE/T))
    def sweep(s):
        for mask in (M0,M1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s.astype(np.int16)*h
            p=tab[dE+8]
            flip=mask & (rng.random(s.shape)<p)
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); chunk_u=[]
    c2=np.zeros(R); c4=np.zeros(R); cnt=0
    for k in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        S2+=m*m; S4+=m**4; c2+=m*m; c4+=m**4; cnt+=1
        if cnt==200:
            chunk_u.append(float(1-(c4.mean()/cnt)/(3*(c2.mean()/cnt)**2))); c2[:]=0;c4[:]=0;cnt=0
    return S2/nmeas, S4/nmeas, chunk_u
def jack(m2,m4):
    R=len(m2); tot2=m2.sum(); tot4=m4.sum()
    U=1-m4.mean()/(3*m2.mean()**2)
    loo=1-((tot4-m4)/(R-1))/(3*((tot2-m2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return U,err
for k,T in enumerate([2.2,2.3,2.4]):
    t=time.time()
    m2,m4,cu=run(T,2000,2000+k,2000,2000)
    U,err=jack(m2,m4)
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4)
    print(T,U,err,'chunk sd',np.std(cu),'time',round(time.time()-t,1),flush=True)