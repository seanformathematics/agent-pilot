import numpy as np, time
def run(T, R=1000, nEq=3000, nMeas=10000, thin=10, seed=12345, N=16):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    beta = 1.0 / T
    acc = np.array([np.exp(-beta * d) if d > 0 else 1.0 for d in range(-8, 9)], dtype=np.float32)
    def sweep():
        for m in masks:
            nn = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
            dE = 2 * s * nn
            p = acc[dE + 8]
            u = rng.random(s.shape, dtype=np.float32)
            flip = (u < p) & m
            s[flip] *= -1
    for _ in range(nEq):
        sweep()
    S2 = np.zeros(R); S4 = np.zeros(R); nm = 0
    for k in range(nMeas):
        sweep()
        if (k + 1) % thin == 0:
            m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
            m2 = m * m
            S2 += m2; S4 += m2 * m2; nm += 1
    return S2, S4, nm
def U_from(S2, S4, nm, idx):
    a2 = S2[idx].sum() / (len(idx) * nm); a4 = S4[idx].sum() / (len(idx) * nm)
    return 1 - a4 / (3 * a2 * a2)
for T, seed in [(2.4, 7001), (2.2, 7002)]:
    S2, S4, nm = run(T, seed=seed)
    print(T, seed, U_from(S2, S4, nm, np.arange(len(S2))), flush=True)
