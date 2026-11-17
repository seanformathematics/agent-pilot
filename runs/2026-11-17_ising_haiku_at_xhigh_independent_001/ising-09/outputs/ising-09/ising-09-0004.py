import sys, time, json
sys.path.insert(0, '/work')
import numpy as np
import wolff as wf

N, W, K = 16, 32, 20
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
results = {}
t_all = time.time()
for i, T in enumerate(Ts):
    seed = 1000 + i
    t0 = time.time()
    sim = wf.PackedWolff(N, W, T, seed=seed, K=K)
    for _ in range(500):
        sim.step()
    L = W * 64
    nmeas = 6000
    m2 = np.zeros((nmeas, L), dtype=np.float32)
    NN = N * N
    for t in range(nmeas):
        sim.step()
        n = sim.counts()
        m = (2.0 * n - NN) / NN
        m2[t] = m * m
    m2 = m2.astype(np.float64)
    m4 = m2 * m2
    s2 = m2.sum(0); s4 = m4.sum(0)
    U, Uerr = wf.jackknife_U(s2, s4, nmeas)
    # stationarity check: first vs second half
    h = nmeas // 2
    U1, _ = wf.jackknife_U(m2[:h].sum(0), m4[:h].sum(0), h)
    U2, _ = wf.jackknife_U(m2[h:].sum(0), m4[h:].sum(0), nmeas - h)
    np.savez(f'/work/res_T{T}.npz', s2=s2, s4=s4, cnt=nmeas, seed=seed)
    results[T] = dict(seed=seed, U=U, Uerr=Uerr, U_first_half=U1, U_second_half=U2, n_meas=L * nmeas, time=time.time() - t0)
    print(f"T={T} U={U:.5f} err={Uerr:.6f} halves: {U1:.5f} {U2:.5f} time={time.time()-t0:.1f}s", flush=True)
json.dump(results, open('/work/results_part1.json', 'w'), indent=1)
print("total", time.time() - t_all)
