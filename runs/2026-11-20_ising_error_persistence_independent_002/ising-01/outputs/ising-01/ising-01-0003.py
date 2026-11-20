
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
exec(open('/dev/null').read())
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16
def build_edges(R):
    idx = np.arange(N*N).reshape(N,N)
    a = np.concatenate([idx.ravel(), idx.ravel()])
    b = np.concatenate([np.roll(idx,-1,axis=1).ravel(), np.roll(idx,-1,axis=0).ravel()])
    off = (np.arange(R)*N*N)[:,None]
    return (a[None,:]+off).ravel(), (b[None,:]+off).ravel()
class SW:
    def __init__(self,R,T,seed):
        self.R=R; self.T=T
        self.rng=np.random.default_rng(seed)
        self.A,self.B=build_edges(R)
        self.s=self.rng.choice(np.array([-1,1],dtype=np.int8),size=R*N*N)
        self.p=1-np.exp(-2.0/T)
        self.n=R*N*N
    def step(self):
        s=self.s
        same = s[self.A]==s[self.B]
        acc = same & (self.rng.random(self.A.size)<self.p)
        g=coo_matrix((np.ones(int(acc.sum()),dtype=np.int8),(self.A[acc],self.B[acc])),shape=(self.n,self.n))
        nc,lab=connected_components(g,directed=False)
        flip=self.rng.random(nc)<0.5
        self.s = np.where(flip[lab], -s, s).astype(np.int8)
    def mags(self):
        return self.s.reshape(self.R,-1).mean(axis=1,dtype=np.float64)

Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
R=1000; warm=300; meas=3000
t0=time.time()
for k,T in enumerate(Ts):
    sim=SW(R,T,1000+k)
    for _ in range(warm): sim.step()
    M=np.empty((meas,R))
    for i in range(meas):
        sim.step(); M[i]=sim.mags()
    np.save(f'/work/pilot_T{T}.npy',M.astype(np.float32))
    m2=(M**2).mean(); m4=(M**4).mean()
    print(T, 'U=',1-m4/(3*m2**2), 'time',round(time.time()-t0,1), flush=True)
