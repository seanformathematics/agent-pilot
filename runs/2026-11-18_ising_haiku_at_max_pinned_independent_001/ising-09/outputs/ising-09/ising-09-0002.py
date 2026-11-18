
mod = r'''
import random, math, array
import numpy as np

def make_nbrs(L):
    nb = []
    for x in range(L):
        for y in range(L):
            nb.append(((((x+1)%L)*L + y), (((x-1)%L)*L + y), (x*L + (y+1)%L), (x*L + (y-1)%L)))
    return nb

def wolff_run(L, T, nsteps, seed, nburn=0, init_spin=None, spin=None):
    """Run nsteps Wolff cluster updates at temperature T on an LxL periodic Ising lattice.
    Returns (M_array_int32 of length nsteps, final spin list). M = sum of spins (not normalized)."""
    N = L*L
    NB = make_nbrs(L)
    rng = random.Random(seed)
    rnd = rng.random
    padd = 1.0 - math.exp(-2.0/T)
    if spin is None:
        if init_spin is None:
            spin = [1 if rnd() < 0.5 else -1 for _ in range(N)]
        else:
            spin = list(init_spin)
    M = sum(spin)
    # burn-in
    for _ in range(nburn):
        k = int(rnd()*N)
        s0 = spin[k]
        spin[k] = -s0
        stack = [k]
        pop = stack.pop; push = stack.append
        csize = 1
        while stack:
            i = pop()
            for j in NB[i]:
                if spin[j] == s0 and rnd() < padd:
                    spin[j] = -s0
                    push(j)
                    csize += 1
        M -= 2*s0*csize
    Ms = array.array('i', [0])*nsteps
    for t in range(nsteps):
        k = int(rnd()*N)
        s0 = spin[k]
        spin[k] = -s0
        stack = [k]
        pop = stack.pop; push = stack.append
        csize = 1
        while stack:
            i = pop()
            for j in NB[i]:
                if spin[j] == s0 and rnd() < padd:
                    spin[j] = -s0
                    push(j)
                    csize += 1
        M -= 2*s0*csize
        Ms[t] = M
    return np.frombuffer(Ms, dtype=np.int32).copy(), spin

def binder_from_M(M, N2):
    m = M.astype(np.float64)/N2
    m2 = m*m
    m4 = m2*m2
    return 1.0 - m4.mean()/(3.0*m2.mean()**2)
'''
with open('/work/wolff.py', 'w') as f:
    f.write(mod)

import time, sys
sys.path.insert(0, '/work')
import importlib
import numpy as np
import wolff
importlib.reload(wolff)

L = 16; N2 = L*L
t0 = time.time()
M, sp = wolff.wolff_run(L, 2.269, 200000, seed=1001, nburn=5000)
t1 = time.time()
print("time for 200k steps at Tc:", round(t1-t0,2), "s")
m = M/N2
print("mean m", m.mean(), "mean|m|", np.abs(m).mean(), "<m^2>", (m*m).mean(), "<m^4>", (m**4).mean())
print("U_naive", 1 - (m**4).mean()/(3*(m*m).mean()**2))

def tau_int(x, maxlag=2000):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    # FFT autocorrelation
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    # Sokal windowing
    tau = 0.5
    for w in range(1, min(maxlag, n//2)):
        tau += acf[w]
        if w >= 6*tau:
            return tau, w
    return tau, -1

m2 = m*m; m4 = m2*m2
for name, x in [("|m|", np.abs(m)), ("m^2", m2), ("m^4", m4)]:
    tau, w = tau_int(x)
    print(name, "tau_int", round(tau,2), "window", w)

# simple binned estimate of U error
def jk_error(M, nb=50):
    m = M.astype(np.float64)/N2
    m2 = m*m; m4 = m2*m2
    n = len(m)
    bs = n//nb
    idx = np.arange(bs*nb).reshape(nb, bs)
    s2 = m2[:bs*nb].reshape(nb,bs).sum(axis=1); s4 = m4[:bs*nb].reshape(nb,bs).sum(axis=1)
    S2 = s2.sum(); S4 = s4.sum(); nn = bs*nb
    Uj = np.empty(nb)
    for b in range(nb):
        a2 = (S2 - s2[b])/(nn - bs); a4 = (S4 - s4[b])/(nn - bs)
        Uj[b] = 1 - a4/(3*a2*a2)
    U = 1 - (S4/nn)/(3*(S2/nn)**2)
    err = np.sqrt((nb-1)/nb * np.sum((Uj - Uj.mean())**2))
    return U, err

for nb in [20, 50, 100]:
    print("jackknife nb", nb, jk_error(M, nb))
