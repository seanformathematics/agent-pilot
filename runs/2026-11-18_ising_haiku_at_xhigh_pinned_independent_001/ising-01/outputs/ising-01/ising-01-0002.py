
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

N=16
S=N*N
x=np.arange(S)//N; y=np.arange(S)%N
right=x*N+(y+1)%N
down=((x+1)%N)*N+y
src1=np.arange(S); dst1=right
src2=np.arange(S); dst2=down
base_src=np.concatenate([src1,src2]); base_dst=np.concatenate([dst1,dst2])

def make_bonds(R):
    off=(np.arange(R)*S)[:,None]
    src=(off+base_src[None,:]).ravel()
    dst=(off+base_dst[None,:]).ravel()
    return src,dst

class SW:
    def __init__(self,R,T,rng,init=None):
        self.R=R; self.T=T; self.rng=rng
        self.n=R*S
        self.src,self.dst=make_bonds(R)
        self.p=1-np.exp(-2.0/T)
        if init is None:
            self.s=np.where(rng.random(self.n)<0.5,1,-1).astype(np.int8)
        else:
            self.s=init.astype(np.int8).ravel().copy()
    def sweep(self):
        s=self.s; rng=self.rng
        same=s[self.src]==s[self.dst]
        r=rng.random(self.src.size)<self.p
        m=same&r
        g=coo_matrix((np.ones(m.sum(),dtype=np.int8),(self.src[m],self.dst[m])),shape=(self.n,self.n))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        self.s=(s*(1-2*flip[lab].astype(np.int8))).astype(np.int8)
    def mags(self):
        return self.s.reshape(self.R,S).mean(axis=1)
    def energy_per_site(self):
        s=self.s.reshape(self.R,N,N).astype(np.int32)
        e=-(s*np.roll(s,1,axis=1)).sum(axis=(1,2))-(s*np.roll(s,1,axis=2)).sum(axis=(1,2))
        return e/(2.0*S)

def onsager_e(T):
    b=1/T
    K=2*np.sinh(2*b)/np.cosh(2*b)**2
    from scipy.special import ellipk
    # K(k) with parameter m=k^2
    Kk=ellipk(K**2)
    return -(1/np.tanh(2*b))*(1+(2/np.pi)*(2*np.tanh(2*b)**2-1)*Kk)

rng=np.random.default_rng(12345)
for T in [2.0,2.3,2.6]:
    sim=SW(256,T,rng)
    t=time.time()
    for _ in range(200):
        sim.sweep()
    dt=time.time()-t
    es=[]
    for _ in range(300):
        sim.sweep()
        es.append(sim.energy_per_site().mean())
    print("T",T,"E/site sim",np.mean(es),"+-",np.std(es)/np.sqrt(len(es)),"onsager(inf)",onsager_e(T),"time per sweep (R=256)",dt/200)
