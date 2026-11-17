import numpy as np, time, sys
sys.path.insert(0, '/work')
import sw

def exact_U(N, T):
    NN = N * N
    nconf = 1 << NN
    configs = np.arange(nconf, dtype=np.int64)
    bits = (configs[:, None] >> np.arange(NN)) & 1
    S = (2 * bits - 1).astype(np.int32)
    idx = np.arange(NN).reshape(N, N)
    E = np.zeros(nconf)
    for i in range(N):
        for j in range(N):
            a = idx[i, j]
            E -= S[:, a] * S[:, idx[i, (j + 1) % N]]
            E -= S[:, a] * S[:, idx[(i + 1) % N, j]]
    M = S.sum(axis=1)
    m = M / NN
    logw = -E / T
    logw -= logw.max()
    w = np.exp(logw)
    Z = w.sum()
    m2 = (w * m ** 2).sum() / Z
    m4 = (w * m ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2)

def jack_U(M, NN):
    # M: (n_steps, R); replica-level jackknife of pooled Binder cumulant
    m = M.astype(np.float64) / NN
    A = (m ** 2).sum(axis=0)   # per replica sums
    B = (m ** 4).sum(axis=0)
    n = m.shape[0]
    At, Bt = A.sum(), B.sum()
    R = M.shape[1]
    U_full = 1 - (Bt / (n * R)) / (3 * (At / (n * R)) ** 2)
    Uk = np.empty(R)
    for k in range(R):
        A_k = (At - A[k]) / (n * (R - 1))
        B_k = (Bt - B[k]) / (n * (R - 1))
        Uk[k] = 1 - B_k / (3 * A_k ** 2)
    Ubar = Uk.mean()
    se = np.sqrt((R - 1) / R * ((Uk - Ubar) ** 2).sum())
    bias = (R - 1) * (Ubar - U_full)
    return U_full, se, bias

for T in [1.8, 2.27, 3.0]:
    t0 = time.time()
    Ue = exact_U(4, T)
    M, _ = sw.sw_run(T, R=2000, n_steps=2000, n_burn=100, seed=2024, N=4)
    Uf, se, bias = jack_U(M, 16)
    print(f"N=4 T={T}: exact={Ue:.5f}  SW pooled={Uf:.5f} jackknife se={se:.5f} jack-bias={bias:.6f}  diff/se={(Uf-Ue)/se:.2f}  t={time.time()-t0:.1f}s")
