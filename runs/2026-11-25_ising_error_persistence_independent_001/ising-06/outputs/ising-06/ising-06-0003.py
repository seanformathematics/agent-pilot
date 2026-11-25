import numpy as np, time, sys
sys.path.insert(0,'/work')
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
exec(open('/tmp/sw.py').read()) if False else None
def sw_run(N, T, R, nsweep, nequil, rng):
    n = N*N
    idx = np.arange(n).reshape(N,N)
    right = np.roll(idx,-1,axis=1).ravel(); down = np.roll(idx,-1,axis=0).ravel()
    src0 = np.concatenate([np.arange(n), np.arange(n)])
    dst0 = np.concatenate([right, down])
    off = (np.arange(R)*n)[:,None]
    S = (src0[None,:]+off).ravel(); D = (dst0[None,:]+off).ravel()
    p = 1-np.exp(-2.0/T)
    sflat = rng.choice(np.array([-1,1],dtype=np.int8), size=R*n)
    M=[]
    for it in range(nequil+nsweep):
        same = sflat[S]==sflat[D]
        keep = same & (rng.random(S.size) < p)
        g = coo_matrix((np.ones(int(keep.sum()),dtype=np.int8),(S[keep],D[keep])), shape=(R*n,R*n))
        nc, lab = connected_components(g, directed=False)
        flip = rng.random(nc) < 0.5
        sflat = np.where(flip[lab], -sflat, sflat)
        if it>=nequil:
            M.append(sflat.reshape(R,n).mean(axis=1, dtype=np.float64))
    return np.array(M)  # shape (nsweep, R)

N=16
temps=[1.9,2.0,2.1]
for k,T in enumerate(temps):
    seed = 1000+k
    rng = np.random.default_rng(seed)
    t=time.time()
    M = sw_run(N,T,1000,3000,500,rng)
    np.save(f'/work/M_T{T}_s{seed}.npy', M)
    m2=(M**2).mean(); m4=(M**4).mean()
    print(T, seed, "time",round(time.time()-t,1), "U=",1-m4/(3*m2**2), flush=True)
