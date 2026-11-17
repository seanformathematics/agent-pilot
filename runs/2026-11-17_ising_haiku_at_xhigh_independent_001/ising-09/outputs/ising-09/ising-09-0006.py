import sys, time, json
sys.path.insert(0, '/work')
import numpy as np
import wolff as wf

N, W, K = 16, 32, 20
prod = {}
for T in [1.5, 2.3, 2.6, 2.8]:
    d = np.load(f'/work/res_T{T}.npz')
    prod[T] = wf.jackknife_U(d['s2'], d['s4'], int(d['cnt']))
res = {}
for i, T in enumerate([1.5, 2.3, 2.6, 2.8]):
    seed = 3000 + i
    sim = wf.PackedWolff(N, W, T, seed=seed, K=K)
    for _ in range(300):
        sim.step()
    L = W * 64; nmeas = 3000; NN = N * N
    m2 = np.zeros((nmeas, L))
    for t in range(nmeas):
        sim.step()
        m = (2.0 * sim.counts() - NN) / NN
        m2[t] = m * m
    m4 = m2 * m2
    U, Ue = wf.jackknife_U(m2.sum(0), m4.sum(0), nmeas)
    Up, Upe = prod[T]
    diff = U - Up
    print(f"T={T} repeat U={U:.5f}+-{Ue:.6f}  production U={Up:.5f}+-{Upe:.6f}  diff={diff:+.6f}  diff/sqrt(err^2+err^2)={diff/np.sqrt(Ue**2+Upe**2):+.2f}", flush=True)
