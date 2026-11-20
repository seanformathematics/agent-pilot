import time
import numpy as np

N = 16
R = 1000
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
MASK0 = ((ii + jj) % 2 == 0)
MASK1 = ~MASK0


def sweep(s, beta, rng):
    for mask in (MASK0, MASK1):
        nb = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
        dE = 2 * s * nb
        p = np.exp(-beta * np.clip(dE, 0, None))
        u = rng.random(s.shape)
        acc = (dE <= 0) | (u < p)
        s[acc & mask] *= -1
    return s


rng = np.random.default_rng(1)
s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
t = time.time()
for _ in range(20):
    sweep(s, 1 / 2.2, rng)
print("time per sweep", (time.time() - t) / 20)
print("mean m", s.mean())
