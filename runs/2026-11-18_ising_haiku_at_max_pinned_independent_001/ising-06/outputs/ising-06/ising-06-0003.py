import sys, time, math
sys.path.insert(0, "/work")
import numpy as np, ising

def iat(x, maxlag=2000):
    """Integrated autocorrelation time (Sokal-style windowing) of series x."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    var = np.dot(x, x) / n
    # FFT-based autocorrelation
    f = np.fft.fft(x, 2 * n)
    ac = np.fft.ifft(f * np.conj(f))[:n].real / n
    ac /= ac[0]
    tau = 0.5
    for t in range(1, min(maxlag, n // 2)):
        tau += ac[t]
        if t >= 6 * tau:
            return tau, t
    return tau, -1

res = {}
for T, nst in [(1.5, 20000), (2.269, 40000), (2.8, 40000)]:
    t0 = time.time()
    out = ising.wolff_run(16, T, nst, seed=2024, nburn=2000, bin_size=1000, keep_series=True)
    dt = time.time() - t0
    m = out['series']
    m2 = m * m; m4 = m2 * m2
    tau2, w2 = iat(m2)
    tau4, w4 = iat(m4)
    taua, wa = iat(np.abs(m))
    U, err = ising.binder_from_bins(out['S2'], out['S4'], 1000)
    print(f"T={T}: steps={nst} time={dt:.1f}s ({dt/nst*1e6:.1f} us/step)  U={U:.4f} +-{err:.4f}  <|m|>={np.mean(np.abs(m)):.4f}")
    print(f"   tau_int(m2)={tau2:.2f} (win {w2}), tau_int(m4)={tau4:.2f} (win {w4}), tau_int(|m|)={taua:.2f}")
    res[T] = m
np.save("/work/pilot_series.npy", np.array([res[1.5][:20000], res[2.269][:20000], res[2.8][:20000]]))
