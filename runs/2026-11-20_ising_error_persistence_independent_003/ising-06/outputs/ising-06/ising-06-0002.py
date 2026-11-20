
import numpy as np, time, sys
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T,rng):
    for par in (0,1):
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE=2*S*nn
        acc=(dE<=0)|(rng.random(S.shape)<np.exp(-dE/T))
        S[:,masks[par]] = np.where(acc,-S,S)[:,masks[par]]
    return S
def run(T,seed,nburn=2000,nmeas=20000,blk=500):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    S=S.astype(np.int8)
    for _ in range(nburn): S=sweep(S,T,rng)
    nb=nmeas//blk
    s2=np.zeros((nb,R)); s4=np.zeros((nb,R))
    for b in range(nb):
        acc2=np.zeros(R); acc4=np.zeros(R)
        for _ in range(blk):
            S=sweep(S,T,rng)
            m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m; acc2+=m2; acc4+=m2*m2
        s2[b]=acc2/blk; s4[b]=acc4/blk
    np.savez(f'/work/T{T:.2f}_N{N}.npz',s2=s2,s4=s4,seed=seed)
    M2=s2.mean(); M4=s4.mean()
    U=1-M4/(3*M2**2)
    # jackknife over replicas
    Ur=[]
    S2r=s2.sum(0); S4r=s4.sum(0); tot2=S2r.sum(); tot4=S4r.sum()
    for r in range(R):
        a2=(tot2-S2r[r])/((R-1)*nb); a4=(tot4-S4r[r])/((R-1)*nb)
        Ur.append(1-a4/(3*a2*a2))
    Ur=np.array(Ur)
    err=np.sqrt((R-1)/R*np.sum((Ur-Ur.mean())**2))
    blockU=[1-s4[b].mean()/(3*s2[b].mean()**2) for b in range(nb)]
    return U,err,np.array(blockU)
for T,seed in [(1.9,101),(2.0,102)]:
    t=time.time()
    U,err,bU=run(T,seed)
    print(f'T={T} U={U:.5f} err={err:.5f} blockU(first..last)={bU[:3].round(4)}..{bU[-3:].round(4)} time={time.time()-t:.1f}',flush=True)
