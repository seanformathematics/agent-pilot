import time, numpy as np
R = 1000; N = 16
rng = np.random.default_rng(1)
s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
beta = 1 / 2.3
acc = np.array([np.exp(-beta * d) if d > 0 else 1.0 for d in range(-8, 9)], dtype=np.float32)
t = time.time()
for _ in range(50):
    for m in masks:
        nn = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
        dE = 2 * s * nn
        p = acc[dE + 8]
        u = rng.random(s.shape, dtype=np.float32)
        flip = (u < p) & m
        s[flip] *= -1
print("time per sweep (R=1000)", (time.time() - t) / 50)
