import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def build_edges(N, R):
    idx = np.arange(N*N).reshape(N,N)
    right = np.roll(idx,-1,axis=1); down = np.roll(idx,-1,axis=0)
    src = np.concatenate([idx.ravel(), idx.ravel()])
    dst = np.concatenate([right.ravel(), down.ravel()])
    off = (np.arange(R)*N*N)[:,None]
    S = (src[None,:]+off).ravel(); D = (dst[None,:]+off).ravel()
    return S, D, np.tile(np.arange(2*N*N)//(N*N)*0,1)

def sw_run(N, T, R, nsweep, nequil, rng, spins=None):
    n = N*N
    idx = np.arange(n).reshape(N,N)
    right = np.roll(idx,-1,axis=1).ravel(); down = np.roll(idx,-1,axis=0).ravel()
    src0 = np.concatenate([np.arange(n), np.arange(n)])
    dst0 = np.concatenate([right, down])
    off = (np.arange(R)*n)[:,None]
    S = (src0[None,:]+off).ravel(); D = (dst0[None,:]+off).ravel()
    p = 1-np.exp(-2.0/T)
    if spins is None:
        s = rng.choice([-1,1], size=(R,n)).astype(np.int8)
    else:
        s = spins.copy()
    sflat = s.ravel()
    M=[]
    total = nequil+nsweep
    for it in range(total):
        same = sflat[S]==sflat[D]
        keep = same & (rng.random(S.size) < p)
        g = coo_matrix((np.ones(keep.sum(),dtype=np.int8),(S[keep],D[keep])), shape=(R*n,R*n))
        nc, lab = connected_components(g, directed=False)
        flip = rng.random(nc) < 0.5
        sflat = np.where(flip[lab], -sflat, sflat)
        if it>=nequil:
            m = sflat.reshape(R,n).mean(axis=1)
            M.append(m)
    return np.array(M), sflat.reshape(R,n)

rng = np.random.default_rng(1)
t=time.time()
M, st = sw_run(16, 2.269, 500, 50, 50, rng)
print("time for 100 sweeps R=500:", time.time()-t)
print(M.shape, np.abs(M).mean())
