import numpy as np, time, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def sweep(S,acc,rng):
    for p in (0,1):
        nb=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
        dE=2*S*nb
        flip=masks[p]&(rng.random(S.shape)<acc[dE+8])
        S[flip]*=-1
def run(T,seed,nburn,nmeas,blk=100):
    rng=np.random.default_rng(seed)
    acc=np.array([min(1.0,np.exp(-d/T)) for d in range(-8,9)])
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    t=time.time()
    for k in range(nburn):
        sweep(S,acc,rng)
        if (k+1)%1000==0:
            m=S.reshape(R,-1).mean(1)
            print(f"  burn {k+1}: <m2>={np.mean(m*m):.4f} t={time.time()-t:.0f}s",flush=True)
    nb=nmeas//blk
    M2=np.zeros((nb,R)); M4=np.zeros((nb,R))
    for b in range(nb):
        s2=np.zeros(R); s4=np.zeros(R)
        for k in range(blk):
            sweep(S,acc,rng)
            m=S.reshape(R,-1).mean(1)
            m2=m*m
            s2+=m2; s4+=m2*m2
        M2[b]=s2/blk; M4[b]=s4/blk
    np.savez(f'/work/T{T}_N{N}.npz',M2=M2,M4=M4,T=T,seed=seed,nburn=nburn,nmeas=nmeas)
    U=1-M4.mean()/(3*M2.mean()**2)
    print(f"T={T} U(all)={U:.5f} <m2>={M2.mean():.5f} time={time.time()-t:.0f}s",flush=True)
run(2.2,1001,4000,20000)
