
ana = r'''
import numpy as np

def U_from_moments(m2, m4):
    return 1.0 - m4 / (3.0 * m2 ** 2)

def jackknife_U(S2, S4, n):
    """S2, S4: per-replica sums over n measurements of m^2 and m^4 (independent replicas).
    Returns U (ratio of pooled means), delete-one-replica jackknife error, bias-corrected U."""
    S2 = np.asarray(S2, float); S4 = np.asarray(S4, float)
    R = len(S2)
    T2, T4 = S2.sum(), S4.sum()
    U_full = U_from_moments(T2 / (R * n), T4 / (R * n))
    m2_i = (T2 - S2) / ((R - 1) * n)
    m4_i = (T4 - S4) / ((R - 1) * n)
    U_i = U_from_moments(m2_i, m4_i)
    Ubar = U_i.mean()
    err = np.sqrt((R - 1) / R * np.sum((U_i - Ubar) ** 2))
    U_bc = R * U_full - (R - 1) * Ubar
    return U_full, err, U_bc
'''
open('/work/ana.py','w').write(ana)

import sys, time, itertools
import numpy as np
sys.path.insert(0,'/work')
from sw import SWReplicas
from ana import jackknife_U, U_from_moments

# ---- exact enumeration for N=4
N=4
nsite=N*N
states = ((np.arange(2**nsite)[:,None] >> np.arange(nsite)[None,:]) & 1)*2-1   # (65536,16)
grid = states.reshape(-1,N,N).astype(np.int32)
E = -(grid*(np.roll(grid,-1,axis=2)+np.roll(grid,-1,axis=1))).sum(axis=(1,2))
Mt = grid.sum(axis=(1,2))
m = Mt/nsite
def exact(T):
    w = np.exp(-(E-E.min())/T); Z=w.sum()
    m2=(w*m**2).sum()/Z; m4=(w*m**4).sum()/Z
    e=(w*E).sum()/Z/nsite; e2=(w*E.astype(float)**2).sum()/Z
    C=(e2-(e*nsite)**2)/T**2/nsite
    return m2,m4,1-m4/(3*m2**2),e,C

R=2000; nburn=100; nmeas=3000
for T in [1.5,2.0,2.5,3.0,4.0]:
    rng=np.random.default_rng(777+int(T*100))
    sim=SWReplicas(N,R,T,rng,'hot')
    for _ in range(nburn): sim.sweep()
    S2=np.zeros(R); S4=np.zeros(R); SE=np.zeros(R); SE2=np.zeros(R)
    t=time.time()
    for _ in range(nmeas):
        sim.sweep()
        mm=sim.magnetization_sum()/nsite
        S2+=mm**2; S4+=mm**4
        ee=sim.energy().astype(float); SE+=ee; SE2+=ee**2
    U,err,Ubc=jackknife_U(S2,S4,nmeas)
    m2_ex,m4_ex,U_ex,e_ex,C_ex=exact(T)
    # energy per site and its error (jackknife over replicas, simple)
    e_r=SE/nmeas/nsite
    e_mc=e_r.mean(); e_err=e_r.std(ddof=1)/np.sqrt(R)
    print(f"T={T}: U_mc={U:.5f}+-{err:.5f} (bc {Ubc:.5f})  U_exact={U_ex:.5f}  z={(U-U_ex)/err:+.2f} | e_mc={e_mc:.5f}+-{e_err:.5f} e_ex={e_ex:.5f} z={(e_mc-e_ex)/e_err:+.2f}  [{time.time()-t:.1f}s]")
