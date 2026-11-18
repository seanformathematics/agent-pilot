import numpy as np, time, os
N=16; R=800; NEQ=2000; NMEAS=6000
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
sns = np.array([-4,-2,0,2,4])
def run(T, R, nequil, nmeas, seed):
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
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    t0=time.time()
    m2b,m4b=run(T,R,NEQ,NMEAS,seed=1000+int(round(T*100)))
    np.savez(f'/work/T{T:.2f}.npz', m2b=m2b, m4b=m4b)
    print(T, time.time()-t0, flush=True)
