import sys, math, random, time
sys.path.insert(0, "/work")
import numpy as np, ising

def wolff_traj(L, T, nsteps, seed, init='cold'):
    rng = random.Random(seed); rnd = rng.random
    N = L*L; nb = ising.build_neighbors(L); p = 1 - math.exp(-2.0/T)
    sp = [1]*N if init == 'cold' else [1 if rnd() < 0.5 else -1 for _ in range(N)]
    M = sum(sp); traj = np.empty(nsteps)
    for t in range(nsteps):
        s_site = int(rnd()*N); s0 = sp[s_site]; sp[s_site] = -s0
        stack = [s_site]; size = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if sp[j] == s0 and rnd() < p:
                    sp[j] = -s0; stack.append(j); size += 1
        M -= 2*s0*size
        traj[t] = M/N
    return traj

checkpoints = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4999]
for T, nch, nst in [(2.3, 200, 5000), (2.5, 200, 5000), (2.8, 300, 5000)]:
    t0 = time.time()
    trajs = np.array([wolff_traj(16, T, nst, seed=50000 + 1000*int(T*10) + c, init='cold') for c in range(nch)])
    m2 = trajs**2
    line = []
    for cp in checkpoints:
        # average over a window of 5 steps around checkpoint for smoothness
        lo = max(0, cp-2); hi = min(nst, cp+3)
        line.append(f"t={cp}:{m2[:, lo:hi].mean():.4f}")
    # production equilibrium <m^2> from the saved production data
    d = np.load(f"/work/prod_T{T:.2f}.npz")
    eq = (d['S2'].sum()/(len(d['S2'])*1000))
    print(f"T={T} cold-start ensemble <m^2>(t): " + "  ".join(line) + f"  | equilibrium <m^2>={eq:.4f}  ({time.time()-t0:.1f}s)")
