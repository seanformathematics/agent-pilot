
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; L=N*N
idx=np.arange(L).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel(); a=idx.ravel()
def run(T,nmeas,seed,neq=2000):
    rng=np.random.default_rng(seed)
    s=rng.choice([-1,1],L).astype(np.int8)
    p=1-np.exp(-2/T)
    ms=np.empty(nmeas)
    for it in range(neq+nmeas):
        b1=(s[a]==s[right])&(rng.random(L)<p)
        b2=(s[a]==s[down])&(rng.random(L)<p)
        r=np.concatenate([a[b1],a[b2]]); c=np.concatenate([right[b1],down[b2]])
        g=coo_matrix((np.ones(len(r),np.int8),(r,c)),shape=(L,L))
        nc,lab=connected_components(g,directed=False)
        fl=rng.choice(np.array([-1,1],np.int8),nc)
        s=fl[lab]
        if it>=neq: ms[it-neq]=s.mean()
    return ms
def U_jk(ms,nb=100):
    m2=ms**2; m4=ms**4
    n=len(ms)//nb*nb
    B2=m2[:n].reshape(nb,-1).mean(1); B4=m4[:n].reshape(nb,-1).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    S2=B2.sum(); S4=B4.sum()
    Uj=1-((S4-B4)/(nb-1))/(3*((S2-B2)/(nb-1))**2)
    err=np.sqrt((nb-1)/nb*((Uj-Uj.mean())**2).sum())
    return U,err
t=time.time(); ms=run(2.3,2000,1,neq=100); print(time.time()-t, U_jk(ms))
