
import numpy as np, time
N=16; R=1000
for T in [1.9,2.0]:
    d=np.load(f'/work/T{T:.2f}_N{N}.npz'); s2=d['s2']; s4=d['s4']
    Ur=1-s4.mean(0)/(3*s2.mean(0)**2)
    print(T,'per-replica U mean',Ur.mean(),'std',Ur.std(ddof=1),'SEM',Ur.std(ddof=1)/np.sqrt(R))
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
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N)).astype(np.int8)
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
    return s2,s4
def summarize(T,s2,s4):
    nb=s2.shape[0]
    M2=s2.mean(); M4=s4.mean(); U=1-M4/(3*M2**2)
    Ur=1-s4.mean(0)/(3*s2.mean(0)**2)
    # jackknife over replicas (ratio estimator)
    S2r=s2.sum(0); S4r=s4.sum(0); tot2=S2r.sum(); tot4=S4r.sum()
    J=np.array([1-((tot4-S4r[r])/((R-1)*nb))/(3*((tot2-S2r[r])/((R-1)*nb))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((J-J.mean())**2))
    return U,err,Ur.std(ddof=1)/np.sqrt(R)
for T,seed in [(2.1,103),(2.2,104)]:
    t=time.time()
    s2,s4=run(T,seed)
    U,err,sem=summarize(T,s2,s4)
    print(f'T={T} U={U:.5f} jack_err={err:.5f} sem_check={sem:.5f} time={time.time()-t:.1f}',flush=True)
