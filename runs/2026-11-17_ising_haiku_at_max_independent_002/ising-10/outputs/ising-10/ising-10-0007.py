import numpy as np, time, sys, json, os
sys.path.insert(0, '/work')
import swnd
os.makedirs('/work/results', exist_ok=True)

def jack_replicas(A, B, n):
    R = A.shape[0]
    At, Bt = A.sum(), B.sum()
    tot = n * R
    U_full = 1 - (Bt / tot) / (3 * (At / tot) ** 2)
    Ak = (At - A) / (n * (R - 1)); Bk = (Bt - B) / (n * (R - 1))
    Uk = 1 - Bk / (3 * Ak ** 2)
    Ubar = Uk.mean()
    se = np.sqrt((R - 1) / R * np.sum((Uk - Ubar) ** 2))
    bias = (R - 1) * (Ubar - U_full)
    return U_full, se, bias

def jack_units(a, b, c):
    At, Bt, Ct = a.sum(), b.sum(), c.sum()
    U_full = 1 - (Bt / Ct) / (3 * (At / Ct) ** 2)
    Ak = (At - a) / (Ct - c); Bk = (Bt - b) / (Ct - c)
    Uk = 1 - Bk / (3 * Ak ** 2)
    n = len(a)
    se = np.sqrt((n - 1) / n * np.sum((Uk - Uk.mean()) ** 2))
    return U_full, se

def tau_int_avg(X):
    X = X - X.mean()
    n, R = X.shape
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(X, n=nfft, axis=0)
    C = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:n].mean(axis=1)
    C = C / C[0]
    for W in range(1, n // 2):
        tau = 0.5 + C[1:W + 1].sum()
        if W >= 5.0 * tau:
            return tau
    return tau

def produce(T, seed, R=512, n_steps=6000, n_burn=300, Lb=200, NN=256):
    t0 = time.time()
    M, _ = swnd.sw_run_nd(T, R=R, n_steps=n_steps, n_burn=n_burn, seed=seed, N=16)
    dt = time.time() - t0
    m = M.astype(np.float64) / NN
    m2 = m ** 2; m4 = m ** 4
    A = m2.sum(axis=0); B = m4.sum(axis=0)
    U, se, bias = jack_replicas(A, B, n_steps)
    nb = n_steps // Lb
    a = m2[:nb * Lb].reshape(nb, Lb, R).sum(axis=1).ravel()
    b = m4[:nb * Lb].reshape(nb, Lb, R).sum(axis=1).ravel()
    c = np.full(a.shape, float(Lb))
    Ub, seb = jack_units(a, b, c)
    tau2 = tau_int_avg(m2[:2000])
    res = dict(T=T, seed=seed, R=R, n_steps=n_steps, n_burn=n_burn, U=float(U), U_jack_bc=float(U - bias),
               se_replica_jack=float(se), bias_replica=float(bias), U_block=float(Ub), se_block_jack=float(seb),
               n_meas=int(R * n_steps), tau_int_m2=float(tau2), mean_abs_m=float(np.abs(m).mean()),
               mean_m2=float(m2.mean()), mean_m4=float(m4.mean()), time_s=float(dt))
    json.dump(res, open(f'/work/results/T_{T:.2f}.json', 'w'), indent=1)
    np.savez(f'/work/results/sums_T_{T:.2f}.npz', A=A, B=B, a=a, b=b)
    return res

for T in [1.5, 1.6, 1.7, 1.8, 1.9]:
    seed = 20250000 + int(round(T * 100))
    r = produce(T, seed)
    print(f"T={r['T']}: U={r['U']:.6f} (jack-bc {r['U_jack_bc']:.6f}) se_rep={r['se_replica_jack']:.6f} se_blk={r['se_block_jack']:.6f} "
          f"tau(m2)={r['tau_int_m2']:.2f} <|m|>={r['mean_abs_m']:.4f} n={r['n_meas']} t={r['time_s']:.0f}s seed={seed}")
