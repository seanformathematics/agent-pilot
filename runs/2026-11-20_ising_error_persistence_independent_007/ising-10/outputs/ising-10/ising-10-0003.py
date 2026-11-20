import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
def run(T, R=1000, nEq=3000, nMeas=20000, thin=10, seed=12345, N=16):
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
    blocks = []
    for k in range(nMeas):
        sweep()
        if (k + 1) % thin == 0:
            m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
            m2 = m * m
            S2 += m2; S4 += m2 * m2; nm += 1
            if nm % 200 == 0:
                blocks.append((S2.copy(), S4.copy(), nm))
    return S2, S4, nm, blocks

def U_from(S2, S4, nm, idx=None):
    if idx is None: idx = np.arange(len(S2))
    a2 = S2[idx].sum() / (len(idx) * nm)
    a4 = S4[idx].sum() / (len(idx) * nm)
    return 1 - a4 / (3 * a2 * a2)

def jack(S2, S4, nm, ngroups=100):
    R = len(S2)
    groups = np.array_split(np.arange(R), ngroups)
    full = U_from(S2, S4, nm)
    us = []
    allidx = np.arange(R)
    for g in groups:
        idx = np.setdiff1d(allidx, g)
        us.append(U_from(S2, S4, nm, idx))
    us = np.array(us); G = len(us)
    var = (G - 1) / G * np.sum((us - us.mean()) ** 2)
    return full, np.sqrt(var)

t0 = time.time()
T = 2.3
S2, S4, nm, blocks = run(T, R=1000, nEq=3000, nMeas=10000, thin=10, seed=1001)
full, err = jack(S2, S4, nm)
print("T", T, "U", full, "err", err, "nm", nm, "time", time.time() - t0)
for (b2, b4, n) in blocks:
    print(n, U_from(b2, b4, n), end="; ")
print()
