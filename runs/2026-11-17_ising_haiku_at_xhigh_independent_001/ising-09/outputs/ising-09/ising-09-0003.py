import sys, time
sys.path.insert(0, '/work')
import numpy as np
import wolff as wf

N, W, T = 16, 32, 2.27
sim = wf.PackedWolff(N, W, T, seed=7, K=20)
t0 = time.time()
for _ in range(300):
    sim.step()
print("burn 300 steps time", time.time() - t0)
t0 = time.time()
for _ in range(300):
    sim.step()
dt = (time.time() - t0) / 300
print("per step (W=32, N=16):", dt)

# measure timeseries for a subset of lanes to get autocorrelation
L = W * 64
nmeas = 3000
ms = np.zeros((nmeas, L))
t0 = time.time()
for t in range(nmeas):
    sim.step()
    n = sim.counts()
    ms[t] = (2.0 * n - N * N) / (N * N)
print("meas time", time.time() - t0)
m2 = ms**2
m4 = m2**2
def tau_int(x, maxlag=200):
    # x: (nt, L) ; average normalized autocorrelation over lanes
    x = x - x.mean(axis=0)
    nt = x.shape[0]
    var = (x**2).mean(axis=0)
    rho = []
    for lag in range(1, maxlag):
        c = (x[:-lag] * x[lag:]).mean(axis=0)
        rho.append((c / var).mean())
    rho = np.array(rho)
    # integrated tau with simple windowing
    tau = 0.5
    for i, r in enumerate(rho):
        tau += r
        if i + 1 >= 6 * tau:
            break
    return tau, i + 1
print("tau_int |m|:", tau_int(np.abs(ms)))
print("tau_int m2:", tau_int(m2))
print("tau_int m4:", tau_int(m4))
print("mean U-ish from this run:", wf.binder_from_sums(m2.sum(0), m4.sum(0), nmeas))
print("jackknife:", wf.jackknife_U(m2.sum(0), m4.sum(0), nmeas))
# per-lane mean m2 check
print("mean m2", m2.mean(), "mean |m|", np.abs(ms).mean())
