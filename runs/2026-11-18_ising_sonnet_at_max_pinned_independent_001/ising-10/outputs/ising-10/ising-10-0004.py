
exact_code = r'''
import numpy as np

def lnZ_torus(N, K):
    """Exact ln Z for NxN periodic Ising (J=1, K=1/T), Kaufman / Ferdinand-Fisher."""
    Ks = -0.5 * np.log(np.tanh(K))
    r = np.arange(2 * N)
    arg = np.cosh(2 * K) * np.cosh(2 * Ks) - np.cos(np.pi * r / N)
    g = np.arccosh(np.maximum(arg, 1.0))
    g[0] = 2 * K - 2 * Ks                      # signed
    def terms(gs):
        x = N * gs / 2.0
        ax = np.abs(x)
        lncosh2 = np.sum(ax + np.log1p(np.exp(-2 * ax)))
        with np.errstate(divide='ignore'):
            lnsinh2 = np.sum(ax + np.log1p(-np.exp(-2 * ax)))
        sgn = np.prod(np.sign(x))
        return lncosh2, (lnsinh2, sgn)
    c1, (s2, sg2) = terms(g[1::2])
    c3, (s4, sg4) = terms(g[0::2])
    # Z1 = prod 2cosh(N g_odd/2), Z2 = prod 2sinh(N g_odd/2), Z3 = prod 2cosh(N g_even/2), Z4 = prod 2sinh(N g_even/2)
    lnZ1 = c1
    lnZ2 = s2; sign2 = sg2
    lnZ3 = c3
    lnZ4 = s4; sign4 = sg4
    L = np.array([lnZ1, lnZ2, lnZ3, lnZ4]); sg = np.array([1.0, sign2, 1.0, sign4])
    Lm = L.max()
    ssum = np.sum(sg * np.exp(L - Lm))
    return np.log(0.5) + 0.5 * N * N * np.log(2 * np.sinh(2 * K)) + Lm + np.log(ssum)

def exact_e_C(N, T, h=1e-4):
    K = 1.0 / T
    f = lambda k: lnZ_torus(N, k)
    d1 = (f(K + h) - f(K - h)) / (2 * h)
    d2 = (f(K + h) - 2 * f(K) + f(K - h)) / h ** 2
    e = -d1 / N ** 2                 # energy per site
    C = K ** 2 * d2 / N ** 2         # specific heat per site
    return e, C
'''
open('/work/exact.py','w').write(exact_code)

import sys, time
import numpy as np
sys.path.insert(0,'/work')
from exact import lnZ_torus, exact_e_C
from sw import SWReplicas
from ana import jackknife_U

# --- check formula against N=4 enumeration
N=4; nsite=16
states = ((np.arange(2**nsite)[:,None] >> np.arange(nsite)[None,:]) & 1)*2-1
grid = states.reshape(-1,N,N).astype(np.int32)
E = -(grid*(np.roll(grid,-1,axis=2)+np.roll(grid,-1,axis=1))).sum(axis=(1,2))
for T in [1.5,2.269,3.0]:
    K=1/T
    lnZ_enum = np.log(np.exp(-K*E.astype(float)*(-1)*(-1)).sum()) if False else np.log(np.exp(-E/T).sum())
    print(f"N=4 T={T}: lnZ enum={lnZ_enum:.10f}  formula={lnZ_torus(4,K):.10f}")
    w=np.exp(-(E-E.min())/T); Z=w.sum(); e=(w*E).sum()/Z/nsite; e2=(w*E.astype(float)**2).sum()/Z
    print("      e enum=%.8f formula=%.8f ; C enum=%.8f formula=%.8f"%(e,exact_e_C(4,T)[0],(e2-(e*nsite)**2)/T**2/nsite,exact_e_C(4,T)[1]))
