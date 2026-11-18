
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; V=N*N
idx=np.arange(V).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel()
def run(T,R,nsweep,neq,seed):
    rng=np.random.default_rng(seed)
    p=1-np.exp(-2/T)
    s=rng.choice([-1,1],size=(R,V)).astype(np.int8)
    off=(np.arange(R)*V)[:,None]
    ms=[]
    for t in range(neq+nsweep):
        bonds=[]
        rows=[];cols=[]
        for nb in (right,down):
            b=(s==s[:,nb])&(rng.random((R,V))<p)
            r,c=np.nonzero(b)
            rows.append(r*V+c); cols.append(r*V+nb[c])
        rows=np.concatenate(rows);cols=np.concatenate(cols)
        g=coo_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(R*V,R*V))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        sf=s.ravel(); sf[flip[lab]]*=-1; s=sf.reshape(R,V)
        if t>=neq: ms.append(s.mean(1,dtype=np.float64))
    return np.array(ms)  # (nsweep,R)
t=time.time(); m=run(2.3,64,200,50,1); print(time.time()-t)
