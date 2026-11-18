
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; R=256; V=N*N
idx=np.arange(V).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel()
off=(np.arange(R)*V)[:,None]
def run(T,seed,neq=200,nmeas=4000):
    rng=np.random.default_rng(seed)
    s=rng.choice([-1,1],size=(R,V)).astype(np.int8)
    p=1-np.exp(-2/T)
    ms=[]
    for it in range(neq+nmeas):
        br=(s==s[:,right])&(rng.random((R,V))<p)
        bd=(s==s[:,down])&(rng.random((R,V))<p)
        a=np.concatenate([(np.arange(V)+off)[br],(np.arange(V)+off)[bd]])
        b=np.concatenate([(right+off)[br],(down+off)[bd]])
        g=coo_matrix((np.ones(len(a),np.int8),(a,b)),shape=(R*V,R*V))
        nc,lab=connected_components(g,directed=False)
        fl=rng.random(nc)<0.5
        s=np.where(fl[lab].reshape(R,V),-s,s).astype(np.int8)
        if it>=neq: ms.append(s.mean(1))
    return np.array(ms)  # (nmeas,R)
t=time.time(); m=run(2.3,1,50,100); print(time.time()-t)
