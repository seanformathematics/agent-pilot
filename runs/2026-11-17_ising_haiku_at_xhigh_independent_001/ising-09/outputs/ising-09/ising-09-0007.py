import sys, time
sys.path.insert(0, '/work')
import numpy as np
import wolff as wf

N, W, K = 16, 32, 20
T = 2.6
Us, Es = [], []
t0 = time.time()
for j in range(8):
    seed = 5000 + j
    sim = wf.PackedWolff(N, W, T, seed=seed, K=K)
    for _ in range(300):
        sim.step()
    L = W * 64; nmeas = 1500; NN = N * N
    m2 = np.zeros((nmeas, L))
    for t in range(nmeas):
        sim.step()
        m = (2.0 * sim.counts() - NN) / NN
        m2[t] = m * m
    m4 = m2 * m2
    U, Ue = wf.jackknife_U(m2.sum(0), m4.sum(0), nmeas)
    Us.append(U); Es.append(Ue)
    print(f"seed {seed}: U={U:.5f} jk_err={Ue:.6f}", flush=True)
Us = np.array(Us); Es = np.array(Es)
print("across-seed mean U", Us.mean(), "std", Us.std(ddof=1), "mean jackknife err", Es.mean(), "ratio std/jk", Us.std(ddof=1) / Es.mean())
print("time", time.time() - t0)
