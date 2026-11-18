import numpy as np, time
N=16
def make(R, rng):
    S = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    return S
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==p) for p in (0,1)]
def sweep(S, thr, rng, masks):
    for p in (0,1):
        nn = np.roll(S,1,1)+np.roll(S,-1,1)
        nn += np.roll(S,1,2)
        nn += np.roll(S,-1,2)
        x = S*nn
        t = thr[x+4]
        acc = rng.random(S.shape, dtype=np.float32) < t
        acc &= masks[p]
        S *= np.where(acc, np.int8(-1), np.int8(1))
    return S
for R in [512, 2048]:
    rng = np.random.default_rng(1)
    S = make(R, rng)
    T=2.0
    thr = np.array([min(1.0, np.exp(-2.0*x/T)) for x in range(-4,5)], dtype=np.float32)
    t0=time.time()
    for _ in range(50):
        sweep(S,thr,rng,masks)
    dt=(time.time()-t0)/50
    print(R, "sec/sweep", dt, "replica-sweeps/s", R/dt)
