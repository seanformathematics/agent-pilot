
code = r'''
import numpy as np

def _log2cosh(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2*ax))
def _log2sinh_abs(x):
    ax = np.abs(x)
    return ax + np.log1p(-np.exp(-2*ax))

def lnZ_torus(K, m, n, gamma0_sign=+1):
    """ln Z for an m x n torus, isotropic coupling K = 1/T (J=1), Z = sum exp(K sum s s')."""
    Ks = -0.5*np.log(np.tanh(K))
    l = np.arange(2*n)
    cg = np.cosh(2*K)*np.cosh(2*Ks) - np.sinh(2*K)*np.sinh(2*Ks)*np.cos(np.pi*l/n)
    gam = np.arccosh(np.maximum(cg, 1.0))
    gam[0] = gamma0_sign*(2*K - 2*Ks)
    odd = gam[1::2]; even = gam[0::2]
    L = []; S = []
    L.append(_log2cosh(m*odd/2).sum()); S.append(1.0)
    L.append(_log2sinh_abs(m*odd/2).sum()); S.append(1.0)
    L.append(_log2cosh(m*even/2).sum()); S.append(1.0)
    L.append(_log2sinh_abs(m*even/2).sum()); S.append(np.sign(np.prod(np.sign(even))))
    L = np.array(L); S = np.array(S)
    Lmax = L.max()
    tot = np.sum(S*np.exp(L-Lmax))
    return np.log(0.5) + 0.5*m*n*np.log(2*np.sinh(2*K)) + Lmax + np.log(tot)

def energy_per_spin_exact(T, N, h=1e-5):
    K = 1.0/T
    d = (lnZ_torus(K+h, N, N) - lnZ_torus(K-h, N, N))/(2*h)
    return -d/(N*N)
'''
open('/work/exact_torus.py','w').write(code)

import sys
sys.path.insert(0,'/work')
import numpy as np, exact_torus as ex

# brute-force 4x4 check
N=4
nst = 1<<(N*N)
states = np.arange(nst, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(N*N, dtype=np.uint32)[None,:]) & 1).astype(np.int8)
spins = (2*bits-1).reshape(nst, N, N)
E = -(spins*(np.roll(spins,-1,axis=2)+np.roll(spins,-1,axis=1))).sum(axis=(1,2)).astype(np.float64)
for T in [1.5, 2.0, 2.3, 2.8, 4.0]:
    K=1.0/T
    w = np.exp(-(E-E.min())*K)
    lnZ_bf = np.log(w.sum()) - E.min()*K
    for sg in (+1,-1):
        lz = ex.lnZ_torus(K, N, N, gamma0_sign=sg)
        print(f"T={T} sign={sg:+d} lnZ formula={lz:.10f} brute={lnZ_bf:.10f} diff={lz-lnZ_bf:.2e}")
    e_bf = (w*E).sum()/w.sum()/N**2
    print("   energy per spin: formula", ex.energy_per_spin_exact(T,N), " brute", e_bf)
# also a non-square torus consistency check is skipped; print N=16 exact energies
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
print("N=16 exact energy per spin:")
for T in Ts:
    print(T, "%.8f" % ex.energy_per_spin_exact(T,16))
