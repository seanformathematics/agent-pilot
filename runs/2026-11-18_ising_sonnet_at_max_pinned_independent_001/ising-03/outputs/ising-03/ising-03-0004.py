
code = r'''
import numpy as np

def lnZ_torus(K, N, M=None):
    """ln Z for the N x M periodic Ising lattice, Z = sum exp(K sum s_i s_j) (Kaufman 1949; Ferdinand-Fisher 1969).
    Uses log-sum-exp with signed products."""
    if M is None: M = N
    n = N           # number of gamma indices per product is n; exponent uses M
    Kd = -0.5*np.log(np.tanh(K))            # dual coupling K*, tanh(K*) = exp(-2K)
    c2K, s2K = np.cosh(2*K), np.sinh(2*K)
    c2Kd, s2Kd = np.cosh(2*Kd), np.sinh(2*Kd)
    l = np.arange(2*n)
    arg = c2K*c2Kd - s2K*s2Kd*np.cos(np.pi*l/n)
    gam = np.arccosh(np.maximum(arg, 1.0))
    gam[0] = 2*K + np.log(np.tanh(K))        # signed: negative for K<Kc, positive for K>Kc
    g_odd = gam[1::2]   # gamma_{2r+1}, r=0..n-1
    g_even = gam[0::2]  # gamma_{2r}
    # logs of the four products (as (sign, log|.|))
    def prod_log(f, g):
        vals = f(M*g/2.0)
        sgn = np.prod(np.sign(vals))
        return sgn, np.sum(np.log(np.abs(vals)))
    def lcosh2(x):   # 2 cosh x, in a form stable for log
        return 2*np.cosh(x)
    terms = [prod_log(lambda x: 2*np.cosh(x), g_odd),
             prod_log(lambda x: 2*np.sinh(x), g_odd),
             prod_log(lambda x: 2*np.cosh(x), g_even),
             prod_log(lambda x: 2*np.sinh(x), g_even)]
    mx = max(t[1] for t in terms)
    tot = sum(sg*np.exp(lg-mx) for sg,lg in terms)
    return np.log(0.5) + 0.5*N*M*np.log(2*s2K) + mx + np.log(tot)

def energy_per_site_exact(T, N, h=1e-5):
    K = 1.0/T
    d = (lnZ_torus(K+h, N) - lnZ_torus(K-h, N))/(2*h)
    return -d/(N*N)
'''
open('/work/exact.py','w').write(code)

import sys
sys.path.insert(0,'/work')
import numpy as np
import exact
# brute force N=4
N=4; n=16
states = np.arange(2**n, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(n,dtype=np.uint32)[None,:]) & 1).astype(np.int8)
s = (2*bits-1).reshape(-1,N,N).astype(np.int16)
E = -(s*np.roll(s,-1,axis=2) + s*np.roll(s,-1,axis=1)).sum(axis=(1,2)).astype(np.float64)
for T in [1.0,1.5,2.0,2.269,2.5,3.0,5.0]:
    K=1/T
    lz_bf = np.log(np.exp(-E*K).sum())      # exp(K sum ss) = exp(-K E)
    lz_k = exact.lnZ_torus(K,4)
    w=np.exp(-(E-E.min())*K); e_bf=(w*E).sum()/w.sum()/n
    e_k = exact.energy_per_site_exact(T,4)
    print(f"T={T}: lnZ brute={lz_bf:.12f} kaufman={lz_k:.12f} diff={lz_bf-lz_k:.2e} | e brute={e_bf:.10f} kaufman={e_k:.10f}")
print()
for T in [1.5,2.0,2.269,2.8]:
    print("N=16 exact e(T=%g) = %.8f"%(T, exact.energy_per_site_exact(T,16)))
