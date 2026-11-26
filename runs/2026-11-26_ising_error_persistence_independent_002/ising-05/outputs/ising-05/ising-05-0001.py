
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)

def run(T, N=16, R=1000, nsweep=2000, ntherm=1000, seed=1, meas_every=1):
    rng = np.random.default_rng(seed)
    # random-sign ordered start per replica
    sign = rng.choice([-1, 1], size=(R, 1, 1)).astype(np.int8)
    s = np.broadcast_to(sign, (R, N, N)).copy()
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == c) for c in (0, 1)]
    # acceptance table for nn sum h in {-4,-2,0,2,4}; dE = 2*s*h
    acc = {}
    for s0 in (-1, 1):
        for h in (-4, -2, 0, 2, 4):
            dE = 2 * s0 * h
            acc[(s0, h)] = min(1.0, np.exp(-dE / T))
    # precompute prob arrays indexed by (s+1)/2 and (h+4)/2
    P = np.zeros((2, 5))
    for si, s0 in enumerate((-1, 1)):
        for hi, h in enumerate((-4, -2, 0, 2, 4)):
            P[si, hi] = acc[(s0, h)]
    def sweep(s):
        for mk in masks:
            h = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
            si = (s > 0).astype(np.int8)
            hi = ((h + 4) // 2).astype(np.int8)
            p = P[si, hi]
            u = rng.random((R, N, N))
            flip = (u < p) & mk
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(ntherm):
        s = sweep(s)
    m2 = []; m4 = []
    for k in range(nsweep):
        s = sweep(s)
        if k % meas_every == 0:
            m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
            m2.append(m*m); m4.append(m**4)
    m2 = np.array(m2); m4 = np.array(m4)  # (nmeas, R)
    return m2, m4

if __name__ == "__main__":
    t0 = time.time()
    m2, m4 = run(2.269, R=1000, nsweep=200, ntherm=200, seed=5)
    dt = time.time() - t0
    print("time", dt, "per sweep", dt/400)
    M2 = m2.mean(); M4 = m4.mean()
    print("U approx", 1 - M4/(3*M2**2))
