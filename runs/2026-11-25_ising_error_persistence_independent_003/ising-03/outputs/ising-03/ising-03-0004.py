import numpy as np, time, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, n_eq=2000, n_meas=20000, block=1000):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    P=np.array([min(1.0,np.exp(-2*p/T)) for p in range(-4,5)])
    def sweep(s):
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            prod=(s*nb+4)
            acc=(rng.random(s.shape,dtype=np.float32)<P[prod]) & mk
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for k in range(n_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    blocks=[]; b2=np.zeros(R); b4=np.zeros(R); bc=0
    for k in range(n_meas):
        s=sweep(s)
        m=s.sum(axis=(1,2))/(N*N); m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
        b2+=m2; b4+=m2*m2; bc+=1
        if bc==block:
            blocks.append(float(1-(b4.sum()/bc/R)/(3*(b2.sum()/bc/R)**2)))
            b2[:]=0;b4[:]=0;bc=0
    return S2,S4,cnt,blocks
for T,seed in [(2.2,104),(2.3,105),(2.4,106)]:
    t=time.time()
    S2,S4,cnt,blocks=run(T,seed)
    np.savez(f'/work/raw_N16_T{T}.npz',S2=S2,S4=S4,cnt=cnt)
    m2=S2.sum()/cnt/R; m4=S4.sum()/cnt/R
    print(T,"U=",round(1-m4/(3*m2*m2),5),"blocks:",[round(b,4) for b in blocks],"time",round(time.time()-t,1),flush=True)
# jackknife over replicas
print("\nT  U  U_err  (jackknife over 1000 independent replicas)")
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/raw_N16_T{T}.npz'); S2=d['S2']; S4=d['S4']; cnt=int(d['cnt'])
    def Uf(a2,a4):
        m2=a2.sum(); m4=a4.sum()
        return 1-(m4/(R-0) if False else m4)/(3*m2*m2)*(1)  # placeholder
    def U_of(idx_mask):
        a2=S2[idx_mask].sum(); a4=S4[idx_mask].sum(); n=idx_mask.sum()*cnt
        return 1-(a4/n)/(3*(a2/n)**2)
    full=np.ones(R,bool)
    U=U_of(full)
    jk=np.empty(R)
    for r in range(R):
        mk=full.copy(); mk[r]=False
        jk[r]=U_of(mk)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(T, round(U,5), round(err,5), "n_samples", R*cnt)
