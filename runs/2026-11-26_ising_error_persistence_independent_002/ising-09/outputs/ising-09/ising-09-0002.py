
import numpy as np, time, json, os
os.makedirs('/work', exist_ok=True)

def run(T, N=16, R=1000, n_eq=1000, n_meas=4000, seed=1, ngroups=50):
    rng = np.random.Generator(np.random.PCG64(seed))
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    # acceptance table indexed by s*h (h neighbour sum in {-4..4})
    acc = np.zeros(9)
    for x in range(-4, 5):  # x = s*h ; dE = 2*s*h
        acc[x+4] = np.exp(-2.0*x/T) if x > 0 else 1.0
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj) % 2 == p) for p in (0, 1)]
    def sweep():
        for p in (0, 1):
            h = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)).astype(np.int16)
            x = s.astype(np.int16) * h
            a = acc[x + 4]
            u = rng.random((R, N, N))
            flip = (u < a) & masks[p]
            s[flip] *= -1
    for _ in range(n_eq):
        sweep()
    S2 = np.zeros(R); S4 = np.zeros(R)
    Ms = []
    for _ in range(n_meas):
        sweep()
        m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        S2 += m2; S4 += m2*m2
    S2 /= n_meas; S4 /= n_meas
    # replica jackknife
    def U_of(idx):
        a2 = S2[idx].mean(); a4 = S4[idx].mean()
        return 1 - a4/(3*a2*a2)
    U = U_of(np.arange(R))
    g = np.array_split(np.arange(R), ngroups)
    Uj = np.array([U_of(np.setdiff1d(np.arange(R), gi)) for gi in g])
    k = len(g)
    err = np.sqrt((k-1)/k * np.sum((Uj - Uj.mean())**2))
    return U, err, np.mean(np.abs(s.reshape(R,-1).mean(1)))

t0 = time.time()
U, err, mabs = run(2.3, R=1000, n_eq=100, n_meas=200, seed=11)
print("timing for 300 sweeps:", time.time()-t0, U, err, mabs)
