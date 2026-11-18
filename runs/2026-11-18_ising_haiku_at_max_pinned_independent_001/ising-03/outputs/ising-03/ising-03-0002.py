import math, random, time
import numpy as np

def build_nbrs(N):
    nb = []
    for x in range(N):
        for y in range(N):
            r = x*N + (y+1) % N
            l = x*N + (y-1) % N
            d = ((x+1) % N)*N + y
            u = ((x-1) % N)*N + y
            nb.append((r, l, d, u))
    return nb

def wolff_series(N, T, seed, n_updates, n_equil=2000):
    nb = build_nbrs(N)
    L = N*N
    rnd = random.Random(seed).random
    p = 1.0 - math.exp(-2.0/T)
    spins = [1 if rnd() < 0.5 else -1 for _ in range(L)]
    M = sum(spins)
    Ms = np.empty(n_updates, dtype=np.int32)
    stack = []
    pop = stack.pop; push = stack.append
    for k in range(n_equil + n_updates):
        i0 = int(rnd()*L)
        s = spins[i0]
        spins[i0] = -s
        push(i0)
        c = 1
        while stack:
            i = pop()
            for j in nb[i]:
                if spins[j] == s and rnd() < p:
                    spins[j] = -s
                    push(j)
                    c += 1
        M -= 2*s*c
        if k >= n_equil:
            Ms[k-n_equil] = M
    return Ms

def binder_from_M(M):
    x = M.astype(np.float64)**2
    y = x*x
    return 1.0 - y.mean()/(3.0*x.mean()**2)

def exact_U(N, T):
    L = N*N
    configs = np.arange(2**L, dtype=np.int64)
    bits = (configs[:, None] >> np.arange(L)) & 1
    s = (2*bits - 1).astype(np.int32)
    M = s.sum(axis=1).astype(np.float64)
    E = np.zeros(len(configs))
    nb = build_nbrs(N)
    for i in range(L):
        r, l, d, u = nb[i]
        E -= s[:, i]*s[:, r] + s[:, i]*s[:, d]
    w = np.exp(-(E - E.min())/T)
    Z = w.sum()
    m2 = (M**2*w).sum()/Z
    m4 = (M**4*w).sum()/Z
    return 1.0 - m4/(3*m2**2)

# Validation on N=4 against exact enumeration
t0 = time.time()
for T in [1.5, 2.269, 3.0]:
    Ex = exact_U(4, T)
    Ms = wolff_series(4, T, seed=12345, n_updates=400000)
    Uw = binder_from_M(Ms)
    print(f"N=4 T={T}: exact U={Ex:.6f}  wolff U={Uw:.6f}  diff={Uw-Ex:+.6f}")
print("time N=4 validation", time.time()-t0)

# Timing for N=16
t0 = time.time()
Ms = wolff_series(16, 2.269, seed=1, n_updates=20000, n_equil=2000)
dt = time.time()-t0
print("N=16 T=2.269: 22000 updates in", round(dt,2), "s ->", round(22000/dt), "updates/s")
print("U (20k):", binder_from_M(Ms), " mean|m|:", np.mean(np.abs(Ms))/256)
