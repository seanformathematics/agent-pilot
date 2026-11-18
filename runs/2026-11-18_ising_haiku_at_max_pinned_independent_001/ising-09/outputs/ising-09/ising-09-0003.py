
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import wolff

L = 4; N = L*L; N2 = N
# exact enumeration
confs = np.arange(2**N, dtype=np.int64)
bits = ((confs[:, None] >> np.arange(N)) & 1).astype(np.int8)
spins = (2*bits - 1).astype(np.int64)          # shape (65536, 16)
# bonds: right and down neighbours with periodic BC, each bond once (2N bonds total)
bonds = []
for x in range(L):
    for y in range(L):
        i = x*L + y
        bonds.append((i, ((x+1) % L)*L + y))
        bonds.append((i, x*L + (y+1) % L))
assert len(bonds) == 2*N
E = np.zeros(2**N)
for (i, j) in bonds:
    E -= spins[:, i]*spins[:, j]
M = spins.sum(axis=1)
m = M/N2
for T in [2.0, 3.0]:
    w = np.exp(-(E - E.min())/T)
    w /= w.sum()
    m2 = (w*m*m).sum(); m4 = (w*m**4).sum(); am = (w*np.abs(m)).sum()
    U_exact = 1 - m4/(3*m2*m2)
    print(f"T={T} exact: <m^2>={m2:.6f} <m^4>={m4:.6f} <|m|>={am:.6f} U={U_exact:.6f} <E>/N={(w*E).sum()/N:.6f}")
    t0 = time.time()
    Ms, sp = wolff.wolff_run(L, T, 1000000, seed=424242, nburn=2000)
    mm = Ms/N2
    # chunk-based error for U
    chunks = np.array_split(np.arange(len(mm)), 50)
    Uc = []
    for c in chunks:
        x = mm[c]; a2 = (x*x).mean(); a4 = (x**4).mean()
        Uc.append(1 - a4/(3*a2*a2))
    Uc = np.array(Uc)
    Uw = 1 - (mm**4).mean()/(3*((mm*mm).mean())**2)
    err = Uc.std(ddof=1)/np.sqrt(len(Uc))
    print(f"   Wolff: <m^2>={(mm*mm).mean():.6f} <m^4>={(mm**4).mean():.6f} <|m|>={np.abs(mm).mean():.6f} U={Uw:.6f} +- {err:.6f} (chunk-based), time {time.time()-t0:.1f}s")
    print("   z-score U:", (Uw-U_exact)/err)
