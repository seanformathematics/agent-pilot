import numpy as np, time
N=16
def bench(R, T, sweeps=50, seed=1):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance for dE = 2*s*nn, s*nn in {-4,-2,0,2,4}; need dE>0 cases: s*nn=2 ->dE=4, 4->dE=8
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    t=time.time()
    for sw in range(sweeps):
        for p in (0,1):
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=(s*nn).astype(np.int8)  # s*nn in -4..4
            u=rng.random((R,N,N))
            flip=(u< np.where(sn<=0,1.0,np.where(sn==2,acc[4],acc[8])))&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
    dt=time.time()-t
    return dt/sweeps
for R in [200,1000,4000]:
    print(R, bench(R,2.3))
