import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import SW

N = 16
def run_series(T, R, K, seed, start='random'):
    sim = SW(N, R, T, seed=seed)
    if start == 'ordered':
        sim.s = np.ones(sim.n, dtype=bool)
    Ms = np.zeros((K, R), dtype=np.int64)
    Es = np.zeros((K, R))
    for t in range(K):
        sim.sweep()
        Ms[t] = sim.mag_sum()
        Es[t] = sim.energy()
    return Ms, Es

def autocorr_tau(x, maxlag=60):
    # x: (K, R) time series; normalized autocorrelation averaged over replicas, integrated tau with window
    x = x - x.mean(axis=0, keepdims=True)
    K, R = x.shape
    var = (x * x).mean(axis=0)
    rho = []
    for lag in range(maxlag):
        c = (x[:K - lag] * x[lag:]).mean(axis=0)
        rho.append((c / var).mean())
    rho = np.array(rho)
    # integrated tau with automatic window (Sokal-like)
    tau = 0.5
    for W in range(1, maxlag):
        tau = 0.5 + rho[1:W + 1].sum()
        if W >= 5 * tau:
            break
    return tau, rho

t0 = time.time()
for T in [1.5, 2.2, 2.3, 2.6, 2.8]:
    Ms, Es = run_series(T, 512, 600, seed=int(T * 100) + 1, start='random')
    M2 = (Ms.astype(float) ** 2); M4 = (Ms.astype(float) ** 4)
    tau2, _ = autocorr_tau(M2)
    tau4, _ = autocorr_tau(M4)
    # U-related: per-replica blocks of 50 sweeps, tau estimate of U-like quantity is approximated by M4 and M2^2
    m2 = M2.mean(); m4 = M4.mean()
    print("T=%.2f  random-start: U(all)=%.4f  tau_int(M2)=%.2f tau_int(M4)=%.2f  first-100 U=%.4f last-400 U=%.4f" % (
        T, 1 - m4 / (3 * m2**2), tau2, tau4,
        1 - M4[:100].mean() / (3 * M2[:100].mean()**2), 1 - M4[200:].mean() / (3 * M2[200:].mean()**2)))
    # equilibration check: running U by window of 20 sweeps, averaged over replicas
    wins = [1 - M4[i:i+20].mean() / (3 * M2[i:i+20].mean()**2) for i in range(0, 200, 20)]
    print("   U in windows of 20 sweeps (first 200):", np.round(wins, 4))
    print("   <E> per window (first 200):", np.round([Es[i:i+20].mean() for i in range(0, 200, 20)], 1))
print("elapsed %.1f s" % (time.time() - t0))

# ordered start check at low T
for T in [1.5, 2.0]:
    Ms, Es = run_series(T, 512, 200, seed=999, start='ordered')
    M2 = Ms.astype(float) ** 2; M4 = Ms.astype(float) ** 4
    wins = [1 - M4[i:i+20].mean() / (3 * M2[i:i+20].mean()**2) for i in range(0, 200, 20)]
    print("ordered start T=%.2f: U windows:" % T, np.round(wins, 4))
