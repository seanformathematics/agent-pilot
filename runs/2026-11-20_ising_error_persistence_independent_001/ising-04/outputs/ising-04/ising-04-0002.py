import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16
idx=np.arange(N*N).reshape(N,N)
right=idx.ravel(); right_n=np.roll(idx,-1,axis=1).ravel()
down=idx.ravel(); down_n=np.roll(idx,-1,axis=0).ravel()
EA=np.concatenate([right,down]); EB=np.concatenate([right_n,down_n])
rng=np.random.default_rng(1)
def sw_sweep(s,T,rng):
    p=1-np.exp(-2.0/T)
    s=s.ravel()
    same=(s[EA]==s[EB])
    u=rng.random(EA.size)<p
    keep=same&u
    a=EA[keep]; b=EB[keep]
    g=coo_matrix((np.ones(a.size,dtype=np.int8),(a,b)),shape=(N*N,N*N))
    nc,lab=connected_components(g,directed=False)
    flip=rng.random(nc)<0.5
    s=np.where(flip[lab],-s,s)
    return s.reshape(N,N)
s=np.ones((N,N),dtype=np.int64)
t=time.time()
for _ in range(2000):
    s=sw_sweep(s,2.269,rng)
print("per sweep us",(time.time()-t)/2000*1e6, np.mean(s))