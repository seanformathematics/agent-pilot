
import numpy as np, sys, itertools
sys.path.insert(0,'/work')
src = open('/work/ising.py').read().replace('N = 16','N = 16')
# generalize N
src = src.replace("def run_chain(T, R, nsweep, seed, nburn=0, init=None, rec_E=False):\n    rng","def run_chain(T, R, nsweep, seed, nburn=0, init=None, rec_E=False, N=16):\n    rng")
src = src.replace("black, white = make_masks(N)\n    bm","black, white = make_masks(N)\n    bm")
src = src.replace("m_rec[k] = S.reshape(N * N, R)","m_rec[k] = S.reshape(N * N, R)")
open('/work/ising.py','w').write(src)
import importlib, ising; importlib.reload(ising)

# exact enumeration on 4x4
L=4
def exact(T):
    Z=0; E1=0; M2=0; M4=0
    for bits in range(2**(L*L)):
        s = np.array([1 if (bits>>k)&1 else -1 for k in range(L*L)]).reshape(L,L)
        E = -np.sum(s*np.roll(s,1,0)) - np.sum(s*np.roll(s,1,1))
        w = np.exp(-E/T)
        m = s.mean()
        Z+=w; E1+=w*E; M2+=w*m*m; M4+=w*m**4
    return E1/Z/(L*L), M2/Z, M4/Z
for T in [1.5, 2.5]:
    e,m2,m4 = exact(T)
    S,m,E = ising.run_chain(T, 4000, 400, seed=7, nburn=200, rec_E=True, N=4)
    print("T",T,"exact E/N2 %.4f m2 %.4f m4 %.4f U %.4f"%(e,m2,m4,1-m4/(3*m2*m2)))
    print("   MC   E/N2 %.4f m2 %.4f m4 %.4f U %.4f"%(E.mean()/16, np.mean(m**2), np.mean(m**4), 1-np.mean(m**4)/(3*np.mean(m**2)**2)))
