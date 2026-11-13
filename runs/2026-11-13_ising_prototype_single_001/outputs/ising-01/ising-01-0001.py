
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; L=N*N
idx=np.arange(L).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel(); src=idx.ravel()
def run(T,nsweep,neq,seed):
    rng=np.random.default_rng(seed)
    s=np.ones(L,dtype=np.int8) if T<2.27 else rng.choice([-1,1],L).astype(np.int8)
    p=1-np.exp(-2/T); ms=np.empty(nsweep)
    for t in range(neq+nsweep):
        r=rng.random(2*L)
        b1=(s[src]==s[right])&(r[:L]<p); b2=(s[src]==s[down])&(r[L:]<p)
        i=np.concatenate([src[b1],src[b2]]); j=np.concatenate([right[b1],down[b2]])
        g=coo_matrix((np.ones(len(i)),(i,j)),shape=(L,L))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        s=np.where(flip[lab],-s,s).astype(np.int8)
        if t>=neq: ms[t-neq]=s.mean()
    return ms
t0=time.time(); m=run(2.3,2000,100,1); print(time.time()-t0)
