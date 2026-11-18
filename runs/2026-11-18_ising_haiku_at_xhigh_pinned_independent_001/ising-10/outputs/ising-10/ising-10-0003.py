
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def build_bonds(N,R):
    idx=np.arange(N*N).reshape(N,N)
    a=np.concatenate([idx.ravel(),idx.ravel()])
    b=np.concatenate([np.roll(idx,-1,axis=1).ravel(),np.roll(idx,-1,axis=0).ravel()])
    off=(np.arange(R)*N*N)[:,None]
    return (a[None,:]+off).ravel(), (b[None,:]+off).ravel()

class SW:
    def __init__(self,N,R,T,rng,init=None):
        self.N=N; self.R=R; self.T=T; self.rng=rng
        self.A,self.B=build_bonds(N,R)
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
    def energies(self):
        N=self.N
        S=self.S.reshape(self.R,N,N).astype(np.int64)
        e=-(S*np.roll(S,-1,axis=1)).sum(axis=(1,2))-(S*np.roll(S,-1,axis=2)).sum(axis=(1,2))
        return e
'''
open('/work/sw.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import numpy as np
from sw import SW
# exact enumeration N=4
N=4
confs=np.array([[1 if (k>>i)&1 else -1 for i in range(N*N)] for k in range(2**(N*N))],dtype=np.int64)
S=confs.reshape(-1,N,N)
E=-(S*np.roll(S,-1,axis=1)).sum(axis=(1,2))-(S*np.roll(S,-1,axis=2)).sum(axis=(1,2))
M=S.reshape(-1,N*N).mean(axis=1)
for T in [2.0, 3.0]:
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    m2=(w*M**2).sum(); m4=(w*M**4).sum()
    print("exact T",T,"<m2>",m2,"<m4>",m4,"U",1-m4/(3*m2**2),"<|E|>",(w*E).sum())
    rng=np.random.default_rng(7)
    s=SW(N,20000,T,rng)
    for _ in range(50): s.sweep()
    m2s=[];m4s=[];Es=[]
    for _ in range(400):
        s.sweep()
        m=s.mags(); m2s.append((m**2).mean()); m4s.append((m**4).mean()); Es.append(s.energies().mean())
    m2e=np.mean(m2s); m4e=np.mean(m4s)
    print("  SW  <m2>",m2e,"<m4>",m4e,"U",1-m4e/(3*m2e**2),"<E>",np.mean(Es))
