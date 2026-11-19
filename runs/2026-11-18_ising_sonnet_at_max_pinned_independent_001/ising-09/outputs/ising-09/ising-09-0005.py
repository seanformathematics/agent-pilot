
import sys, time
sys.path.insert(0, "/work")
import numpy as np

code = r'''
import numpy as np

def lnZ_exact(N, T):
    """ln Z for the NxN periodic Ising model, J=1, E=-sum s s (2N^2 bonds). Ferdinand-Fisher / Kaufman."""
    K = 1.0/T
    Kst = -0.5*np.log(np.tanh(K))          # dual coupling: tanh K = exp(-2K*)
    l = np.arange(2*N)
    cg = np.cosh(2*Kst)*np.cosh(2*K) - np.sinh(2*Kst)*np.sinh(2*K)*np.cos(np.pi*l/N)
    g = np.arccosh(np.maximum(cg, 1.0))
    g[0] = 2*K + np.log(np.tanh(K))        # signed gamma_0 = 2K - 2K*
    g_odd = g[1::2]; g_even = g[0::2]
    M = N
    # log of each term, handle signs
    def logcosh_prod(gs):
        return np.sum(np.log(2*np.cosh(M*gs/2)))
    def sinh_prod(gs):
        vals = 2*np.sinh(M*gs/2)
        sign = np.prod(np.sign(vals))
        return sign, np.sum(np.log(np.abs(vals)))
    L1 = logcosh_prod(g_odd)
    s2, L2 = sinh_prod(g_odd)
    L3 = logcosh_prod(g_even)
    s4, L4 = sinh_prod(g_even)
    Ls = np.array([L1, L2, L3, L4]); sg = np.array([1.0, s2, 1.0, s4])
    Lmax = Ls.max()
    tot = np.sum(sg*np.exp(Ls-Lmax))
    return np.log(0.5) + (N*N/2.0)*np.log(2*np.sinh(2*K)) + Lmax + np.log(tot)

def energy_exact(N, T, h=1e-4):
    """<E>/N^2 = -(1/N^2) d lnZ / d beta."""
    b = 1.0/T
    f = lambda beta: lnZ_exact(N, 1.0/beta)
    dlnZ = (f(b+h) - f(b-h))/(2*h)
    return -dlnZ/(N*N)
'''
open("/work/exact.py","w").write(code)
from exact import lnZ_exact, energy_exact

# brute-force N=4
conf = np.arange(2**16)
bits = ((conf[:,None] >> np.arange(16)) & 1)*2 - 1
s = bits.reshape(-1,4,4)
E = -(s*np.roll(s,-1,axis=1) + s*np.roll(s,-1,axis=2)).sum(axis=(1,2))
for T in [1.5, 2.0, 2.269, 3.0]:
    w = np.exp(-E/T); Z = w.sum()
    e_bf = (w*E).sum()/Z/16
    print(f"N=4 T={T}: lnZ brute={np.log(Z):.10f} formula={lnZ_exact(4,T):.10f}  e brute={e_bf:.8f} formula={energy_exact(4,T):.8f}")

# N=3 brute force too (2^9)
conf = np.arange(2**9)
bits = ((conf[:,None] >> np.arange(9)) & 1)*2 - 1
s = bits.reshape(-1,3,3)
E3 = -(s*np.roll(s,-1,axis=1) + s*np.roll(s,-1,axis=2)).sum(axis=(1,2))
for T in [1.5, 2.5]:
    w = np.exp(-E3/T); Z = w.sum()
    print(f"N=3 T={T}: lnZ brute={np.log(Z):.10f} formula={lnZ_exact(3,T):.10f}")

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
print("N=16 exact energies per site:")
for T in Ts:
    print(f"  T={T}: e={energy_exact(16,T):.8f}")
