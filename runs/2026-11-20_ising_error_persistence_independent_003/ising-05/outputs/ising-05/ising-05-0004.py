import numpy as np, time, json, os
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0), ((ii+jj)%2==1)]
def sweep(s,rng,acc):
    for m in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb
        r = rng.random(s.shape,dtype=np.float32)
        flip = ((dE<=0) | (r < acc[np.clip(dE,0,8)])) & m
        s[flip] *= -1
def jk_U(S2,S4,nm):
    # S2,S4: per-replica sums over measurements (R,)
    R=len(S2)
    def U_of(a2,a4):
        m2=a2.sum()/(R*nm); m4=a4.sum()/(R*nm)
        return 1-m4/(3*m2*m2)
    U=U_of(S2,S4)
    jk=np.empty(R)
    tot2=S2.sum(); tot4=S4.sum()
    for r in range(R):
        m2=(tot2-S2[r])/((R-1)*nm); m4=(tot4-S4[r])/((R-1)*nm)
        jk[r]=1-m4/(3*m2*m2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err
def run(T,R=1000,neq=2000,nmeas=5000,seed=1):
    acc=np.array([np.exp(-dE/T) for dE in range(9)],dtype=np.float32)
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for k in range(neq): sweep(s,rng,acc)
    S2=np.zeros(R); S4=np.zeros(R)
    blk=[]
    for k in range(nmeas):
        sweep(s,rng,acc)
        m=s.reshape(R,-1).mean(1)
        m2=m*m
        S2+=m2; S4+=m2*m2
    return S2,S4,nmeas
t0=time.time()
for T,seed in [(2.3,101),(2.2,102)]:
    S2,S4,nm=run(T,seed=seed)
    U,err=jk_U(S2,S4,nm)
    print(T,"U=%.5f err=%.5f"%(U,err),"time",time.time()-t0,flush=True)
    np.savez(f'/work/prod_T{T}.npz',S2=S2,S4=S4,nm=nm)
