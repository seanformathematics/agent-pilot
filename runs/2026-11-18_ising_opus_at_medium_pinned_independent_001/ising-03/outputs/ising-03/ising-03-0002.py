
import numpy as np, time, json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; L=N*N
idx=np.arange(L).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel()
I=np.concatenate([idx.ravel(),idx.ravel()]); J=np.concatenate([right,down])
def run(T,nmeas,ntherm,seed):
    rng=np.random.default_rng(seed); p=1-np.exp(-2/T)
    s=rng.choice([-1,1],L)
    ms=np.empty(nmeas)
    for k in range(ntherm+nmeas):
        bond=(s[I]==s[J])&(rng.random(2*L)<p)
        g=coo_matrix((np.ones(bond.sum()),(I[bond],J[bond])),shape=(L,L))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        s=np.where(flip[lab],-s,s)
        if k>=ntherm: ms[k-ntherm]=s.mean()
    return ms
def jack(ms,nb):
    m2=(ms**2).reshape(nb,-1).mean(1); m4=(ms**4).reshape(nb,-1).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    S2=m2.sum();S4=m4.sum()
    Uj=1-((S4-m4)/(nb-1))/(3*((S2-m2)/(nb-1))**2)
    return U,np.sqrt((nb-1)/nb*((Uj-Uj.mean())**2).sum())
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res=[]
for i,T in enumerate(Ts):
    seed=1000+i; n=200000
    ms=run(T,n,2000,seed)
    U,e=jack(ms,100); U2,e2=jack(ms,50); U4,e4=jack(ms,400)
    # equilibration check: halves
    Ua,_=jack(ms[:n//2],50);Ub,_=jack(ms[n//2:],50)
    res.append(dict(T=T,U=U,err=e,n=n,seed=seed))
    print(T,U,e,e2,e4,Ua,Ub,flush=True)
json.dump(res,open('/work/res.json','w'))
