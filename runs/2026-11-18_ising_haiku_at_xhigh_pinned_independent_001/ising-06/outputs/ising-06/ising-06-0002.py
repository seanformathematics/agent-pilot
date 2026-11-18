import numpy as np, time
N=16
def make_table(T):
    # prod = s*nb in {-4,-2,0,2,4}; accept prob for flip
    tab = np.zeros(9, dtype=np.float32)
    for p in range(-4,5):
        tab[p+4] = 1.0 if p<=0 else np.exp(-2.0*p/T)
    return tab
def run(R, T, sweeps, seed):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((R,N,N))<0.5, -1, 1).astype(np.int8)
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks = [((ii+jj)%2==c) for c in (0,1)]
    tab = make_table(T)
    t0=time.time()
    for sw in range(sweeps):
        for c in (0,1):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            prod = (s*nb + 4)
            pt = tab[prod]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pt) & masks[c]
            s = np.where(flip, -s, s).astype(np.int8)
    dt=time.time()-t0
    return dt/sweeps
for R in [200, 1000]:
    print(R, run(R, 2.27, 20, 1)*1000, "ms per sweep")
