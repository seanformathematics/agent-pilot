
import numpy as np
def acf(x, maxlag):
    x = x - x.mean()
    n = x.size
    nfft = 1
    while nfft < 2*n: nfft *= 2
    F = np.fft.rfft(x, n=nfft)
    c = np.fft.irfft(F*np.conj(F), n=nfft)[:n]
    c = c/ (n - np.arange(n))  # unbiased
    return c[:maxlag]/c[0]

for fn, label in [("/work/prod_M_T2.50.npy", "production T=2.5"), ("/work/pilot_T2.5.npy", "pilot T=2.5")]:
    Ms = np.load(fn).astype(np.float64)
    R, K = Ms.shape
    m = Ms/256.0; A2 = m*m; A4 = A2*A2
    # replica-averaged series
    g2 = A2.mean(axis=0); g4 = A4.mean(axis=0)
    rho_g = acf(g2, 12)
    # single replica acf (averaged over some replicas) for reference
    rho_1 = np.mean([acf(A2[r], 12) for r in range(0, R, 50)], axis=0)
    # block-mean variance of replica-averaged series vs expectation
    for B in [1, 10, 100, 300]:
        nb = K//B
        bm = g2[:nb*B].reshape(nb, B).mean(axis=1)
        print(f"{label}: block length {B}: sd of block means of replica-avg m2 = {bm.std(ddof=1):.3e}")
    # pairwise correlations between replicas (time-series, centred)
    Xc = A2 - A2.mean(axis=1, keepdims=True)
    Xc = Xc/np.linalg.norm(Xc, axis=1, keepdims=True)
    C = Xc[:200] @ Xc[:200].T
    off = C[np.triu_indices(200, 1)]
    print(f"{label}: mean pairwise replica correlation of m2 series = {off.mean():.5f}  (expected ~0 +- {1/np.sqrt(K):.4f}/sqrt(npairs-ish)), sd {off.std():.4f}; expected sd ~ {1/np.sqrt(K):.4f}")
    print(f"{label}: acf of replica-avg m2, lags 0..6: {np.round(rho_g[:7],4)}")
    print(f"{label}: acf of single replica m2, lags 0..6: {np.round(rho_1[:7],4)}")
    # sd of replica-averaged m2 time series vs per-replica sd / sqrt(R)
    print(f"{label}: sd(replica-avg m2 over time)={g2.std():.5f}, per-replica sd/sqrt(R)={A2.std()/np.sqrt(R):.5f}")
