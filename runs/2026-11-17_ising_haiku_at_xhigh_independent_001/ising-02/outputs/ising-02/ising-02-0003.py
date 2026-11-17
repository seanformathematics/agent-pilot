
code = r'''
import numpy as np, scipy.sparse as sp, scipy.sparse.csgraph as csg, time
N=16; NN=N*N
Ts=np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])

def setup(T_rep):
    R=len(T_rep)
    p=1-np.exp(-2.0/np.asarray(T_rep,float))
    idx=np.arange(NN).reshape(N,N)
    right=np.roll(idx,-1,axis=1); down=np.roll(idx,-1,axis=0)
    bu=np.concatenate([idx.ravel(),idx.ravel()])
    bv=np.concatenate([right.ravel(),down.ravel()])
    off=(np.arange(R)*NN)[:,None]
    U=(bu[None,:]+off).ravel(); V=(bv[None,:]+off).ravel()
    rep=np.repeat(np.arange(R),2*NN)
    return p,U,V,rep

def sw_sweep(s,rng,p,U,V,rep,R):
    aligned=s[U]==s[V]
    keep=aligned&(rng.random(U.size)<p[rep])
    uu=U[keep]; vv=V[keep]
    g=sp.csr_matrix((np.ones(uu.size,dtype=np.int8),(uu,vv)),shape=(R*NN,R*NN))
    ncomp,labels=csg.connected_components(g,directed=False)
    flip=(rng.integers(0,2,size=ncomp)*2-1).astype(np.int8)
    return s*flip[labels]

def jk_U(S2,S4,groups_idx):
    """S2,S4: per-replica sums of m^2, m^4 (same count n each). Returns U, err per group."""
    out=[]
    for gi in groups_idx:
        a2=S2[gi]; a4=S4[gi]; g=len(gi)
        def U_of(x2,x4):
            m2=x2.mean(); m4=x4.mean()
            return 1-m4/(3*m2*m2)
        Ufull=U_of(a2,a4)
        tot2=a2.sum(); tot4=a4.sum()
        loo=np.empty(g)
        for k in range(g):
            m2=(tot2-a2[k])/(g-1); m4=(tot4-a4[k])/(g-1)
            loo[k]=1-m4/(3*m2*m2)
        err=np.sqrt((g-1)/g*np.sum((loo-loo.mean())**2))
        out.append((Ufull,err))
    return out
'''
open('/work/swlib.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import numpy as np, time
from swlib import *
seed=2024
rng=np.random.default_rng(seed)
reps=100
T_rep=np.repeat(Ts,reps); R=len(T_rep)
p,U,V,rep=setup(T_rep)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=R*NN)
t0=time.time()
burn=600
# burn-in, track group mean of |m| to watch drift
for it in range(burn):
    s=sw_sweep(s,rng,p,U,V,rep,R)
    if it in (50,150,300,599):
        m=s.reshape(R,NN).sum(1)/NN
        print("burn",it,[round(float(np.mean(m[k*reps:(k+1)*reps]**2)),3) for k in (0,5,7,9,13)])
S2=np.zeros(R); S4=np.zeros(R); n=0
meas=2000
ts_m2=np.zeros((meas,14)); ts_m4=np.zeros((meas,14))
for it in range(meas):
    s=sw_sweep(s,rng,p,U,V,rep,R)
    m=s.reshape(R,NN).sum(1)/NN
    m2=m*m; m4=m2*m2
    S2+=m2; S4+=m4; n+=1
    ts_m2[it]=m2.reshape(14,reps).mean(1); ts_m4[it]=m4.reshape(14,reps).mean(1)
print("time",time.time()-t0)
groups=[np.arange(k*reps,(k+1)*reps) for k in range(14)]
res=jk_U(S2/n,S4/n,groups)
for T,(u,e) in zip(Ts,res):
    print(T, round(u,5), round(e,5))
np.savez('/work/pilot.npz',S2=S2,S4=S4,n=n,ts_m2=ts_m2,ts_m4=ts_m4)
