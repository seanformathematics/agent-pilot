import os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

N=16; NN=N*N

def make_bonds(R, N=N):
    NN=N*N
    idx = np.arange(NN).reshape(N,N)
    right = idx[:, (np.arange(N)+1)%N].ravel()
    down = idx[(np.arange(N)+1)%N, :].ravel()
    a0 = np.concatenate([idx.ravel(), idx.ravel()])
    b0 = np.concatenate([right, down])
    off = (np.arange(R)*NN)[:,None]
    a = (a0[None,:]+off).ravel()
    b = (b0[None,:]+off).ravel()
    return a, b

def sw_sweep(s, a, b, T, rng, R, NN=NN):
    p = 1.0 - np.exp(-2.0/T)
    same = s[a]==s[b]
    r = rng.random(a.size)
    act = same & (r < p)
    aa = a[act]; bb = b[act]
    M = R*NN
    g = csr_matrix((np.ones(aa.size, dtype=np.int8), (aa, bb)), shape=(M,M))
    nc, lab = connected_components(g, directed=False)
    flip = rng.random(nc) < 0.5
    return s * (1 - 2*flip[lab].astype(np.int8))

def run(T, R, S, burn, seed, N=N, s0=None):
    """Return ms: array (S, R) of magnetization per sweep per replica (after burn-in)."""
    NN=N*N
    rng = np.random.default_rng(seed)
    a,b = make_bonds(R, N)
    if s0 is None:
        s = (rng.integers(0,2,size=R*NN)*2-1).astype(np.int8)
    else:
        s = s0.copy()
    for _ in range(burn):
        s = sw_sweep(s,a,b,T,rng,R,NN)
    out = np.empty((S,R))
    for k in range(S):
        s = sw_sweep(s,a,b,T,rng,R,NN)
        out[k] = s.reshape(R,NN).sum(axis=1)/NN
    return out

def binder_from(ms):
    """ms: (S,R). Returns U computed from pooled samples."""
    m2 = np.mean(ms**2)
    m4 = np.mean(ms**4)
    return 1 - m4/(3*m2*m2)

def jack_U(ms, nblock=None):
    """Delete-one-replica jackknife over replicas (independent chains)."""
    S,R = ms.shape
    # per replica sums
    s2 = (ms**2).sum(axis=0); s4=(ms**4).sum(axis=0)
    S2=s2.sum(); S4=s4.sum(); n=S*R
    U_full = 1 - (S4/n)/(3*(S2/n)**2)
    Uj = np.empty(R)
    for k in range(R):
        m2 = (S2-s2[k])/((R-1)*S)
        m4 = (S4-s4[k])/((R-1)*S)
        Uj[k] = 1 - m4/(3*m2*m2)
    var = (R-1)/R * np.sum((Uj-Uj.mean())**2)
    return U_full, np.sqrt(var), Uj

def tau_int(x, maxlag=200):
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x,x)/n
    tau = 0.5
    for t in range(1,maxlag):
        c = np.dot(x[:-t], x[t:])/(n-t)/c0
        if c < 0.05: break
        tau += c
    return tau
'''
with open('/work/ising.py','w') as f:
    f.write(code)

import sys
sys.path.insert(0,'/work')
import importlib, ising
importlib.reload(ising)
import numpy as np, time
t0=time.time()
ms = ising.run(2.27, R=200, S=3000, burn=200, seed=2024)
print("time", time.time()-t0)
U, Ue, Uj = ising.jack_U(ms)
print("U", U, "err", Ue)
# autocorrelation of m^2 and m^4 averaged over replicas
x2 = (ms**2)
x4 = (ms**4)
# average autocorrelation: compute tau on a few replicas
taus=[ising.tau_int(ms[:,k]**2, 300) for k in range(0,200,20)]
print("tau int m^2 per replica sample", np.round(taus,2))
taus=[ising.tau_int(np.abs(ms[:,k]), 300) for k in range(0,200,20)]
print("tau int |m|", np.round(taus,2))
