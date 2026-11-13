
import numpy as np, time, json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; L=N*N
idx=np.arange(L).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel(); src=idx.ravel()
def run(T,nsweep,neq,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice([-1,1],L).astype(np.int8)
    p=1-np.exp(-2/T); ms=np.empty(nsweep)
    for t in range(neq+nsweep):
        r=rng.random(2*L)
        b1=(s[src]==s[right])&(r[:L]<p); b2=(s[src]==s[down])&(r[L:]<p)
        i=np.concatenate([src[b1],src[b2]]); j=np.concatenate([right[b1],down[b2]])
        g=coo_matrix((np.ones(len(i)),(i,j)),shape=(L,L))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        s=np.where(flip[lab],-s,s).astype(np.int8)
        if t>=neq: ms[t-neq]=s.mean()
    return ms
def jack(ms,nb):
    m2=ms**2; m4=m2**2; n=len(ms)//nb*nb
    b2=m2[:n].reshape(nb,-1).mean(1); b4=m4[:n].reshape(nb,-1).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    j2=(b2.sum()-b2)/(nb-1); j4=(b4.sum()-b4)/(nb-1); Uj=1-j4/(3*j2**2)
    return U, np.sqrt((nb-1)/nb*((Uj-Uj.mean())**2).sum())
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
for k,T in enumerate(Ts):
    ms=run(T,200000,2000,1000+k)
    U,e=jack(ms,100); _,e2=jack(ms,25)
    h=len(ms)//2; Ua=1-np.mean(ms[:h]**4)/(3*np.mean(ms[:h]**2)**2); Ub=1-np.mean(ms[h:]**4)/(3*np.mean(ms[h:]**2)**2)
    res[T]=(U,max(e,e2),len(ms)); print(T,U,e,e2,Ua,Ub,flush=True)
json.dump(res,open('/work/res.json','w'))
