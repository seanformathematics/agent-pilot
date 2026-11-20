import numpy as np, time
N=16; R=1000; NBURN=5000; NMEAS=20000; BLK=500
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ptab=np.array([1.0 if dE<=0 else np.exp(-dE/T) for dE in range(-8,9)])
    nb=NMEAS//BLK
    S2=np.zeros((nb,R)); S4=np.zeros((nb,R))
    def sweep(s):
        for mk in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=(2*s*nn)
            p=ptab[dE+8]
            acc=(rng.random(size=s.shape,dtype=np.float32)<p)&mk
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for _ in range(NBURN): s=sweep(s)
    for b in range(nb):
        acc2=np.zeros(R); acc4=np.zeros(R)
        for _ in range(BLK):
            s=sweep(s)
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            acc2+=m*m; acc4+=m**4
        S2[b]=acc2/BLK; S4[b]=acc4/BLK
    return S2,S4
for T,seed in [(2.3,2321),(2.4,2421)]:
    t=time.time()
    S2,S4=run(T, seed)
    np.savez(f'/work/raw_T{T}.npz',S2=S2,S4=S4)
    m2=S2.mean(0); m4=S4.mean(0)
    U=1-m4.mean()/(3*m2.mean()**2)
    h1=1-S4[:20].mean()/(3*S2[:20].mean()**2); h2=1-S4[20:].mean()/(3*S2[20:].mean()**2)
    print(T,"time",round(time.time()-t,1),"U",round(U,5),"halves",round(h1,5),round(h2,5),"m2",m2.mean())