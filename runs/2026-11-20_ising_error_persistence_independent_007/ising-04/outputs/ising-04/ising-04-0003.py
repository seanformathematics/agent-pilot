
import numpy as np, time, os, json
N=16
i,j=np.indices((N,N))
MASKS=[((i+j)%2==p) for p in (0,1)]
def run(T,R,nsweep,neq,seed,block=500):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    tab=np.zeros(9)
    for x in (-4,-2,0,2,4): tab[x+4]=min(1.0,np.exp(-2*x/T))
    nb=(nsweep)//block
    out2=np.zeros((nb,R)); out4=np.zeros((nb,R)); bi=0
    for sweep in range(neq+nsweep):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            prod=(s*nn).astype(np.int16)
            pr=tab[prod+4]
            flip=(rng.random((R,N,N))<pr)&MASKS[p]
            s=np.where(flip,-s,s).astype(np.int8)
        if sweep>=neq:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            out2[(sweep-neq)//block]+=m2/block
            out4[(sweep-neq)//block]+=m2*m2/block
    return out2,out4
def U_jack(s2,s4):
    # s2,s4: (blocks,R) block means per replica; pooled over blocks per replica
    a2=s2.mean(0); a4=s4.mean(0)
    R=len(a2)
    M2=a2.sum(); M4=a4.sum()
    U=1-M4/(3*M2*M2/1.0)  # placeholder
    full=1-(a4.mean())/(3*a2.mean()**2)
    idx=np.arange(R)
    loo=np.empty(R)
    for k in range(R):
        mk=np.delete(np.arange(R),k)
        loo[k]=1-a4[mk].mean()/(3*a2[mk].mean()**2)
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return full,err
res={}
for T in [1.9,2.0]:
    t=time.time()
    s2,s4=run(T,2000,8000,2000,seed=int(T*1000))
    full,err=U_jack(s2,s4)
    # drift check: U by block halves
    h=len(s2)//2
    u1=1-s4[:h].mean()/(3*s2[:h].mean()**2); u2=1-s4[h:].mean()/(3*s2[h:].mean()**2)
    np.savez(f'/work/run_T{T}_a.npz',s2=s2,s4=s4)
    print(T,full,err,'halves',u1,u2,'time',time.time()-t,flush=True)
