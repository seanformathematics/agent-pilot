
import numpy as np, time, json
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
def analyze(ms,nb=100):
    m2=ms**2;m4=ms**4
    B=len(ms)//nb
    m2b=m2[:B*nb].reshape(nb,B).mean(1);m4b=m4[:B*nb].reshape(nb,B).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    jk=np.array([1-((m4b.sum()-m4b[k])/(nb-1))/(3*((m2b.sum()-m2b[k])/(nb-1))**2) for k in range(nb)])
    return U,np.sqrt((nb-1)*np.mean((jk-jk.mean())**2))
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res=[]
for i,T in enumerate(Ts):
    ms=run(T,200000,5000,1000+i)
    U,e=analyze(ms)
    # also check half-halves consistency
    U1,_=analyze(ms[:100000]);U2,_=analyze(ms[100000:])
    res.append(dict(T=T,U=float(U),U_err=float(e),n=len(ms),seed=1000+i))
    print(T,U,e,U1,U2,flush=True)
json.dump(res,open('/work/res.json','w'))
