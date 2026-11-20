import numpy as np, time, os, json
os.makedirs('/work', exist_ok=True)
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
    blocks = []
    for k in range(nMeas):
        sweep()
        if (k + 1) % thin == 0:
            m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
            m2 = m * m
            S2 += m2; S4 += m2 * m2; nm += 1
            if nm % 250 == 0:
                blocks.append((S2.copy(), S4.copy(), nm))
    return S2, S4, nm, blocks

def U_from(S2, S4, nm, idx):
    a2 = S2[idx].sum() / (len(idx) * nm)
    a4 = S4[idx].sum() / (len(idx) * nm)
    return 1 - a4 / (3 * a2 * a2)

def jack(S2, S4, nm, ngroups=100):
    R = len(S2)
    groups = np.array_split(np.arange(R), ngroups)
    full = U_from(S2, S4, nm, np.arange(R))
    allidx = np.arange(R)
    us = np.array([U_from(S2, S4, nm, np.setdiff1d(allidx, g)) for g in groups])
    G = len(us)
    var = (G - 1) / G * np.sum((us - us.mean()) ** 2)
    return full, np.sqrt(var)

results = {}
t0 = time.time()
for i, T in enumerate([1.9, 2.0, 2.1, 2.2, 2.3, 2.4]):
    S2, S4, nm, blocks = run(T, R=1000, nEq=3000, nMeas=10000, thin=10, seed=2000 + i)
    full, err = jack(S2, S4, nm)
    # drift check: U from first vs second half of measurements
    h = len(blocks) // 2
    bU = [U_from(b2, b4, n, np.arange(len(b2))) for (b2, b4, n) in blocks]
    results[T] = dict(U=float(full), err=float(err), nm=int(nm), seed=2000 + i,
                      U_first_half=float(bU[h-1]), U_last=float(bU[-1]))
    print(T, results[T], "elapsed", round(time.time() - t0, 1), flush=True)
json.dump(results, open('/work/results.json', 'w'), indent=1)
