
import sys, time, math
sys.path.insert(0, "/work")
import numpy as np
import ising

def tau_int(x, maxlag=2000):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    # FFT-based autocorrelation
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf = acf / acf[0]
    tau = 0.5
    W = None
    for t in range(1, min(maxlag, n//4)):
        tau += acf[t]
        if t >= 6.0 * tau:  # Sokal automatic windowing
            return tau, t
    return tau, min(maxlag, n//4)

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16; NN = 256
n_meas = 200000
results = []
for idx, T in enumerate(Ts):
    t0 = time.time()
    seed = 1000 + idx
    Mrec, spins = ising.simulate(N, T, 2000, n_meas, seed=seed, start="hot")
    np.save(f"/work/pilot_T{T:.1f}.npy", Mrec.astype(np.int16))
    dt = time.time() - t0
    U, e100 = ising.binder_jackknife(Mrec, NN, nblocks=100)
    _, e200 = ising.binder_jackknife(Mrec, NN, nblocks=200)
    _, e40 = ising.binder_jackknife(Mrec, NN, nblocks=40)
    m = Mrec / NN
    tm2, _ = tau_int(m*m)
    tm4, _ = tau_int(m**4)
    results.append((T, U, e200, tm2, tm4))
    print(f"T={T:.2f} U={U:.5f} err(B40)={e40:.5f} err(B100)={e100:.5f} err(B200)={e200:.5f} tau_m2={tm2:.2f} tau_m4={tm4:.2f} <|m|>={np.mean(np.abs(m)):.4f} time={dt:.1f}s", flush=True)
