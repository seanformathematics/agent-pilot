
import numpy as np, time
N=16
def bench(R, T, sweeps=200, seed=1):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.indices((N,N))
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance table indexed by s*h: flip prob exp(-2*s*h/T) for s*h in {-4..4}
    sh_vals=np.array([-4,-2,0,2,4])
    acc=np.minimum(1.0,np.exp(-2*sh_vals/T))
    accT=np.zeros(9); 
    # index = s*h + 4 (s*h in -4..4 -> 0..8)
    for v in range(-4,5):
        accT[v+4]=min(1.0,np.exp(-2*v/T))
    accT=accT.astype(np.float32)
    t0=time.time()
    for sw in range(sweeps):
        for p in (0,1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            sh=(s*h).astype(np.int16)+4
            pr=accT[sh]
            flip=(rng.random((R,N,N),dtype=np.float32)<pr)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
    dt=time.time()-t0
    return dt/sweeps
for R in (500,1000,2000):
    print(R, bench(R,2.27,sweeps=50))
