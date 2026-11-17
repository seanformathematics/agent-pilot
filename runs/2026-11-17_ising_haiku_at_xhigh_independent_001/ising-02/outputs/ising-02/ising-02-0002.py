
import numpy as np, scipy.sparse as sp, scipy.sparse.csgraph as csg, time
N=16; NN=N*N

def setup(T_rep):
    R=len(T_rep)
    p = 1-np.exp(-2.0/np.asarray(T_rep,float))
    idx=np.arange(NN).reshape(N,N)
    right=np.roll(idx,-1,axis=1); down=np.roll(idx,-1,axis=0)
    bu=np.concatenate([idx.ravel(),idx.ravel()])
    bv=np.concatenate([right.ravel(),down.ravel()])
    off=(np.arange(R)*NN)[:,None]
    U=(bu[None,:]+off).ravel(); V=(bv[None,:]+off).ravel()
    rep=np.repeat(np.arange(R),2*NN)
    return p,U,V,rep

def sw_sweep(s, rng, p, U, V, rep, R):
    # s: flat int8 array length R*NN
    aligned = s[U]==s[V]
    keep = aligned & (rng.random(U.size) < p[rep])
    uu=U[keep]; vv=V[keep]
    g=sp.csr_matrix((np.ones(uu.size,dtype=np.int8),(uu,vv)),shape=(R*NN,R*NN))
    ncomp,labels=csg.connected_components(g,directed=False)
    flip=(rng.integers(0,2,size=ncomp)*2-1).astype(np.int8)
    return s*flip[labels]

rng=np.random.default_rng(1)
Ts=np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
T_rep=np.repeat(Ts,100)
p,U,V,rep=setup(T_rep)
R=len(T_rep)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=R*NN)
t0=time.time()
for it in range(5):
    s=sw_sweep(s,rng,p,U,V,rep,R)
t1=time.time()
print("time per sweep (R=%d):"%R,(t1-t0)/5)
# sanity: magnetization per T
for it in range(200):
    s=sw_sweep(s,rng,p,U,V,rep,R)
m=s.reshape(R,NN).sum(1)/NN
for k,T in enumerate(Ts):
    mm=m[k*100:(k+1)*100]
    print(T, np.mean(np.abs(mm)).round(4), np.mean(mm**2).round(4))
print("time total",time.time()-t0)
