
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; L=N*N
idx=np.arange(L).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel()
I=np.concatenate([idx.ravel(),idx.ravel()]); J=np.concatenate([right,down])
def run(T,nmeas,neq,seed):
    rng=np.random.default_rng(seed)
    p=1-np.exp(-2/T)
    s=rng.choice([-1,1],L)
    ms=np.empty(nmeas)
    for t in range(neq+nmeas):
        bond=(s[I]==s[J])&(rng.random(2*L)<p)
        g=coo_matrix((np.ones(bond.sum()),(I[bond],J[bond])),shape=(L,L))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        s=np.where(flip[lab],-s,s)
        if t>=neq: ms[t-neq]=s.mean()
    return ms
def analyze(ms,nb=50):
    m2=ms**2;m4=ms**4
    B=len(ms)//nb
    m2b=m2[:B*nb].reshape(nb,B).mean(1);m4b=m4[:B*nb].reshape(nb,B).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    jk=[]
    for k in range(nb):
        a=(m2b.sum()-m2b[k])/(nb-1);b=(m4b.sum()-m4b[k])/(nb-1)
        jk.append(1-b/(3*a*a))
    jk=np.array(jk)
    return U,np.sqrt((nb-1)*np.mean((jk-jk.mean())**2))
t=time.time();ms=run(2.8,20000,1000,1);print(time.time()-t,analyze(ms))
t=time.time();ms=run(1.5,20000,1000,2);print(time.time()-t,analyze(ms))
