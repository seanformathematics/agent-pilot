import time, random, math
import numpy as np

N = 16
NN = N * N
nbr = []
for x in range(N):
    for y in range(N):
        i = x * N + y
        nbr.append([((x + 1) % N) * N + y, ((x - 1) % N) * N + y, x * N + (y + 1) % N, x * N + (y - 1) % N])

def run_wolff(T, nsteps, seed, init=None):
    rng = random.Random(seed)
    padd = 1.0 - math.exp(-2.0 / T)
    if init is None:
        spins = [1 if rng.random() < 0.5 else -1 for _ in range(NN)]
    else:
        spins = list(init)
    M = sum(spins)
    Ms = np.empty(nsteps, dtype=np.int32)
    rnd = rng.random
    rr = rng.randrange
    for t in range(nsteps):
        i = rr(NN)
        s0 = spins[i]
        spins[i] = -s0
        stack = [i]
        c = 1
        while stack:
            j = stack.pop()
            for k in nbr[j]:
                if spins[k] == s0 and rnd() < padd:
                    spins[k] = -s0
                    stack.append(k)
                    c += 1
        M -= 2 * s0 * c
        Ms[t] = M
    return Ms, spins

for T in [2.269, 1.5, 2.8]:
    t0 = time.time()
    Ms, sp = run_wolff(T, 20000, 12345)
    dt = time.time() - t0
    m = Ms / NN
    print("T", T, "time per 20k steps", round(dt, 2), "<m2>", np.mean(m**2), "<m4>", np.mean(m**4), "U", 1 - np.mean(m**4) / (3 * np.mean(m**2)**2))
