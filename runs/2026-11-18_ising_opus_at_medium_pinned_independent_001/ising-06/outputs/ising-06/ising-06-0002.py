
import numpy as np, json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16; V=N*N
idx=np.arange(V).reshape(N,N)
right=np.roll(idx,-1,1).ravel(); down=np.roll(idx,-1,0).ravel()
def run(T,R,nsweep,neq,seed):
    rng=np.random.default_rng(seed)
    p=1-np.exp(-2/T)
    s=rng.choice([-1,1],size=(R,V)).astype(np.int8)
    m2=np.zeros(R);m4=np.zeros(R)
    for t in range(neq+nsweep):
        rows=[];cols=[]
        for nb in (right,down):
            b=(s==s[:,nb])&(rng.random((R,V))<p)
            r,c=np.nonzero(b)
            rows.append(r*V+c); cols.append(r*V+nb[c])
        rows=np.concatenate(rows);cols=np.concatenate(cols)
        g=coo_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(R*V,R*V))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        sf=s.ravel(); sf[flip[lab]]*=-1; s=sf.reshape(R,V)
        if t>=neq:
            m=s.mean(1,dtype=np.float64); m2+=m*m; m4+=m**4
    return m2/nsweep,m4/nsweep
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=64;ns=8000;neq=500;res=[]
for i,T in enumerate(Ts):
    a2,a4=run(T,R,ns,neq,1000+i)
    U=1-a4.mean()/(3*a2.mean()**2)
    jk=np.array([1-np.delete(a4,k).mean()/(3*np.delete(a2,k).mean()**2) for k in range(R)])
    err=np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
    # also half-split check
    h=[1-a4[j::2].mean()/(3*a2[j::2].mean()**2) for j in (0,1)]
    res.append((T,U,err,R*ns)); print(T,U,err,h,flush=True)
json.dump(res,open('/work/res.json','w'))
