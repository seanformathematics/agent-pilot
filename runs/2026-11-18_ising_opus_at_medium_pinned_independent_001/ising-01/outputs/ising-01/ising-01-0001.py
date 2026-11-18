
import numpy as np, time, json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; R=64; L=N*N
idx=np.arange(L).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel()
off=(np.arange(R)*L)[:,None]
def run(T,seed,neq=500,nmeas=4000):
    rng=np.random.default_rng(seed)
    p=1-np.exp(-2/T)
    s=np.ones((R,L),np.int8) if T<2.27 else rng.choice(np.array([-1,1],np.int8),(R,L))
    A=(off+np.arange(L)).ravel()
    m2=np.zeros(R); m4=np.zeros(R); ms=[]
    for t in range(neq+nmeas):
        b1=(s==s[:,right])&(rng.random((R,L))<p)
        b2=(s==s[:,down])&(rng.random((R,L))<p)
        rows=np.concatenate([A[b1.ravel()],A[b2.ravel()]])
        cols=np.concatenate([(off+right).ravel()[b1.ravel()],(off+down).ravel()[b2.ravel()]])
        g=coo_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(R*L,R*L))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        s=np.where(flip[lab].reshape(R,L),-s,s).astype(np.int8)
        if t>=neq:
            m=s.mean(1,dtype=np.float64); ms.append(m)
    ms=np.array(ms)  # nmeas x R
    a2=(ms**2).mean(0); a4=(ms**4).mean(0)
    U=1-a4.mean()/(3*a2.mean()**2)
    jk=np.array([1-np.delete(a4,i).mean()/(3*np.delete(a2,i).mean()**2) for i in range(R)])
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    return U,err,ms.size
t0=time.time()
print(run(2.3,1,100,200),time.time()-t0)
