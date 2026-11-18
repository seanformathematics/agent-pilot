
import numpy as np, time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

N=16
def build_bonds(R):
    idx=np.arange(N*N).reshape(N,N)
    right=idx; right_n=np.roll(idx,-1,axis=1)
    down=idx; down_n=np.roll(idx,-1,axis=0)
    a=np.concatenate([right.ravel(),down.ravel()])
    b=np.concatenate([right_n.ravel(),down_n.ravel()])
    off=(np.arange(R)*N*N)[:,None]
    A=(a[None,:]+off).ravel(); B=(b[None,:]+off).ravel()
    return A,B

class SW:
    def __init__(self,R,T,rng,init=None):
        self.R=R; self.T=T; self.rng=rng
        self.A,self.B=build_bonds(R)
        self.n=R*N*N
        self.p=1-np.exp(-2.0/T)
        if init is None:
            self.S=rng.choice(np.array([-1,1],dtype=np.int8),size=self.n)
        else:
            self.S=init.copy()
    def sweep(self):
        S=self.S
        same=S[self.A]==S[self.B]
        act=same & (self.rng.random(self.A.size)<self.p)
        A=self.A[act]; B=self.B[act]
        g=csr_matrix((np.ones(A.size,dtype=np.int8),(A,B)),shape=(self.n,self.n))
        nc,lab=connected_components(g,directed=False)
        flip=self.rng.random(nc)<0.5
        self.S=(S*(1-2*flip[lab].astype(np.int8))).astype(np.int8)
    def mags(self):
        return self.S.reshape(self.R,-1).mean(axis=1)

rng=np.random.default_rng(12345)
for R in [64,256,1024]:
    s=SW(R,2.269,rng)
    t=time.time()
    for _ in range(20): s.sweep()
    dt=(time.time()-t)/20
    print(R, "time per sweep", dt, "per replica-sweep us", dt/R*1e6)
