
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

N=16
def build_edges(R):
    idx = np.arange(N*N).reshape(N,N)
    right = idx; right_n = np.roll(idx,-1,axis=1)
    down = idx; down_n = np.roll(idx,-1,axis=0)
    a = np.concatenate([right.ravel(), down.ravel()])
    b = np.concatenate([right_n.ravel(), down_n.ravel()])
    off = (np.arange(R)*N*N)[:,None]
    A = (a[None,:]+off).ravel(); B = (b[None,:]+off).ravel()
    return A,B

class SW:
    def __init__(self,R,T,seed):
        self.R=R; self.T=T
        self.rng=np.random.default_rng(seed)
        self.A,self.B=build_edges(R)
        self.s=self.rng.choice(np.array([-1,1],dtype=np.int8),size=R*N*N)
        self.p=1-np.exp(-2.0/T)
    def step(self):
        s=self.s
        same = s[self.A]==s[self.B]
        acc = same & (self.rng.random(self.A.size)<self.p)
        A=self.A[acc]; B=self.B[acc]
        n=self.R*N*N
        g=coo_matrix((np.ones(A.size,dtype=np.int8),(A,B)),shape=(n,n))
        nc,lab=connected_components(g,directed=False)
        flip=self.rng.random(nc)<0.5
        self.s = np.where(flip[lab], -s, s).astype(np.int8)
    def mags(self):
        return self.s.reshape(self.R,-1).mean(axis=1)

sim=SW(1000,2.269,12345)
t=time.time()
for _ in range(20): sim.step()
print('time per step R=1000:',(time.time()-t)/20)
m=sim.mags(); print(np.mean(m**2))
