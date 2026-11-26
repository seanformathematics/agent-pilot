
import numpy as np, time, json
def run(T, N=16, R=1000, n_eq=1000, n_meas=4000, seed=1, ngroups=50):
    rng = np.random.Generator(np.random.PCG64(seed))
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    acc = np.zeros(9)
    for x in range(-4, 5):
        acc[x+4] = np.exp(-2.0*x/T) if x > 0 else 1.0
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj) % 2 == p) for p in (0, 1)]
    def sweep():
        for p in (0, 1):
            h = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)).astype(np.int16)
            x = s.astype(np.int16) * h
            a = acc[x + 4]
            u = rng.random((R, N, N))
            s[(u < a) & masks[p]] *= -1
    for _ in range(n_eq): sweep()
    S2 = np.zeros(R); S4 = np.zeros(R)
    for _ in range(n_meas):
        sweep()
        m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
        S2 += m*m; S4 += (m*m)**2
    S2 /= n_meas; S4 /= n_meas
    U_of = lambda idx: 1 - S4[idx].mean()/(3*S2[idx].mean()**2)
    U = U_of(np.arange(R))
    g = np.array_split(np.arange(R), ngroups)
    Uj = np.array([U_of(np.setdiff1d(np.arange(R), gi)) for gi in g])
    k = len(g)
    return float(U), float(np.sqrt((k-1)/k*np.sum((Uj-Uj.mean())**2)))
out = {}
for i, T in enumerate([1.9, 2.0, 2.1, 2.2, 2.3, 2.4]):
    out[T] = run(T, R=1000, n_eq=500, n_meas=3000, seed=5000+i)
    print(T, out[T], flush=True)
