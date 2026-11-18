import numpy as np, time
N=16
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
sns = np.array([-4,-2,0,2,4])
def run(T, R, nequil, nmeas, seed, ngroups=20):
    rng=np.random.default_rng(seed)
    probtab = np.minimum(1.0, np.exp(-2*sns/T)).astype(np.float32)
    if T<2.269: s=np.ones((R,N,N),dtype=np.int8)
    else: s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sn = (s*nn).astype(np.int8)
            acc = rng.random(s.shape, dtype=np.float32) < probtab[(sn+4)//2]
            s[acc & m] *= -1
    for _ in range(nequil): sweep(s)
    # per-replica sums of m2, m4 over measurements, stored in blocks of 50 sweeps
    B=50; nb=nmeas//B
    m2b=np.zeros((nb,R)); m4b=np.zeros((nb,R))
    for b in range(nb):
        acc2=np.zeros(R); acc4=np.zeros(R)
        for _ in range(B):
            sweep(s)
            m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
            m2=m*m; acc2+=m2; acc4+=m2*m2
        m2b[b]=acc2/B; m4b[b]=acc4/B
    return m2b, m4b
def binder(m2r, m4r):
    # m2r, m4r: arrays over replicas (time-averaged)
    return 1 - m4r.mean()/(3*m2r.mean()**2)
def jk(m2b,m4b,ngroups=20):
    R=m2b.shape[1]
    M2=m2b.mean(0); M4=m4b.mean(0)
    U=binder(M2,M4)
    idx=np.arange(R)%ngroups
    Us=[]
    for g in range(ngroups):
        k=idx!=g
        Us.append(binder(M2[k],M4[k]))
    Us=np.array(Us)
    n=ngroups
    err=np.sqrt((n-1)/n*np.sum((Us-Us.mean())**2))
    return U, err
for T in [1.5,2.3,2.8]:
    t0=time.time()
    m2b,m4b=run(T,400,1000,3000,seed=int(T*100))
    U,err=jk(m2b,m4b)
    # check equilibration: compare first and second half of blocks
    h=m2b.shape[0]//2
    U1,_=jk(m2b[:h],m4b[:h]); U2,_=jk(m2b[h:],m4b[h:])
    print(f"T={T} U={U:.5f} err={err:.5f} halves {U1:.4f} {U2:.4f} time={time.time()-t0:.1f}s")
