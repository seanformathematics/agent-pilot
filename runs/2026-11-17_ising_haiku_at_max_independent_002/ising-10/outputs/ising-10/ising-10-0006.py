import numpy as np, time, sys
sys.path.insert(0, '/work')
import swnd

def tau_int_avg(X, cmax=None):
    # X: (n, R) series; average normalized autocorrelation over replicas, Sokal automatic window
    X = X - X.mean()
    n, R = X.shape
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(X, n=nfft, axis=0)
    C = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:n].mean(axis=1)
    C = C / C[0]
    tau = 0.5
    for W in range(1, n // 2):
        tau = 0.5 + C[1:W + 1].sum()
        if W >= 5.0 * tau:
            return tau, W
    return tau, n // 2

def Ustats(M, NN=256):
    m = M.astype(np.float64) / NN
    A = (m ** 2).sum(axis=0); B = (m ** 4).sum(axis=0)
    n = m.shape[0]; R = M.shape[1]
    At, Bt = A.sum(), B.sum()
    U_full = 1 - (Bt / (n * R)) / (3 * (At / (n * R)) ** 2)
    Uk = np.empty(R)
    for k in range(R):
        A_k = (At - A[k]) / (n * (R - 1)); B_k = (Bt - B[k]) / (n * (R - 1))
        Uk[k] = 1 - B_k / (3 * A_k ** 2)
    se = np.sqrt((R - 1) / R * ((Uk - Uk.mean()) ** 2).sum())
    return U_full, se

# early relaxation check from random start (no burn-in), and from ordered start at low T
for T in [2.0, 2.27]:
    M0, _ = swnd.sw_run_nd(T, R=256, n_steps=12, n_burn=0, seed=11, N=16)
    m2 = ((M0 / 256.0) ** 2).mean(axis=1)
    print("early <m2> per step T=%.2f:" % T, np.round(m2, 3))

res = {}
for T in [1.5, 2.0, 2.27, 2.6, 2.8]:
    t0 = time.time()
    M, _ = swnd.sw_run_nd(T, R=256, n_steps=3000, n_burn=200, seed=int(T * 1000) + 1, N=16)
    dt = time.time() - t0
    m = M / 256.0
    U, se = Ustats(M)
    t2, W2 = tau_int_avg((m ** 2))
    t4, W4 = tau_int_avg((m ** 4))
    # per-sample std of U-like quantity (time series of x = m^4 - 2*(B/A)*m^2 approx)
    res[T] = (U, se)
    print(f"T={T}: U={U:.5f} se_jack={se:.5f} (R=256,n=3000)  tau_int(m2)={t2:.2f} (W={W2}) tau_int(m4)={t4:.2f} (W={W4})  <|m|>={np.abs(m).mean():.4f}  time={dt:.1f}s")
    np.save(f'/work/pilot_M_T{T}.npy', M)
