import numpy as np, time
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

N=16; NN=N*N

def make_bonds(R):
    # site index within replica: i*N+j ; right neighbor and down neighbor
    idx = np.arange(NN).reshape(N,N)
    right = idx[:, (np.arange(N)+1)%N].ravel()
    down = idx[(np.arange(N)+1)%N, :].ravel()
    a0 = np.concatenate([idx.ravel(), idx.ravel()])
    b0 = np.concatenate([right, down])
    off = (np.arange(R)*NN)[:,None]
    a = (a0[None,:]+off).ravel()
    b = (b0[None,:]+off).ravel()
    return a, b

def sw_sweep(s, a, b, T, rng, R):
    # s: flat int8 array of length R*NN
    p = 1.0 - np.exp(-2.0/T)
    same = s[a]==s[b]
    r = rng.random(a.size)
    act = same & (r < p)
    aa = a[act]; bb = b[act]
    M = R*NN
    g = csr_matrix((np.ones(aa.size, dtype=np.int8), (aa, bb)), shape=(M,M))
    nc, lab = connected_components(g, directed=False)
    flip = rng.random(nc) < 0.5
    return s * (1 - 2*flip[lab].astype(np.int8))

R=500
rng = np.random.default_rng(12345)
a,b = make_bonds(R)
s = (rng.integers(0,2,size=R*NN)*2-1).astype(np.int8)
T=2.27
t0=time.time()
ms=[]
for sweep in range(400):
    s = sw_sweep(s,a,b,T,rng,R)
    if sweep>=100:
        m = s.reshape(R,NN).sum(axis=1)/NN
        ms.append(m)
print("time per sweep", (time.time()-t0)/400)
ms=np.array(ms)  # (S,R)
m2=(ms**2).mean(); m4=(ms**4).mean()
print("U", 1-m4/(3*m2**2), "<|m|>", np.abs(ms).mean())
