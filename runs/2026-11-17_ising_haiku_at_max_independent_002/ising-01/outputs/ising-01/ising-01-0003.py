
import numpy as np, time, sys
sys.path.insert(0, '/work')
from ising import wolff_run

def exact_hist(N, T):
    n = N * N
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
    S = (2 * bits - 1).reshape(-1, N, N).astype(np.int64)
    E = -((S * np.roll(S, -1, axis=2)).sum(axis=(1, 2)) + (S * np.roll(S, -1, axis=1)).sum(axis=(1, 2)))
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()
    M = S.sum(axis=(1, 2))
    return M, w

N = 4
T = 2.269
M_ex, w_ex = exact_hist(N, T)
vals = np.unique(M_ex)
P_ex = np.array([w_ex[M_ex == v].sum() for v in vals])
t0 = time.time()
R, nsteps, burn = 2048, 1000, 100
Mt, _ = wolff_run(N, T, R, nsteps, seed=99991)
Ms = Mt[burn:].astype(np.int64).ravel()
cnt = np.array([(Ms == v).sum() for v in vals]); tot = cnt.sum()
P_w = cnt / tot
# chi-square-like comparison over bins with expected counts > 20
expc = P_ex * tot
mask = expc > 20
chi2 = (((cnt[mask] - expc[mask]) ** 2) / expc[mask]).sum()
print("N=4 T=2.269: dof", mask.sum(), "chi2", round(chi2, 2))
for v, pe, pw in zip(vals, P_ex, P_w):
    print(f"M={v:3d} exact={pe:.5f} wolff={pw:.5f}")
Mf = Ms / (N * N)
m2 = (Mf ** 2).mean(); m4 = (Mf ** 4).mean()
print("U_wolff", 1 - m4 / (3 * m2 ** 2), "U_exact", 1 - (w_ex * (M_ex / 16.) ** 4).sum() / (3 * (w_ex * (M_ex / 16.) ** 2).sum() ** 2))
print("time N=4 check", round(time.time() - t0, 2))

# ---- 16x16 pilot ----
def analyze(Mt, burn, N=16):
    x = Mt[burn:].astype(np.float64) / (N * N)
    m2 = x ** 2; m4 = x ** 4          # (nt, R)
    R = x.shape[1]
    S2 = m2.sum(axis=0); S4 = m4.sum(axis=0); nt = x.shape[0]
    tot2 = S2.sum(); tot4 = S4.sum(); tot = nt * R
    U = 1 - (tot4 / tot) / (3 * (tot2 / tot) ** 2)
    # replica jackknife
    Ujk = np.empty(R)
    for r in range(R):
        a2 = (tot2 - S2[r]) / ((R - 1) * nt); a4 = (tot4 - S4[r]) / ((R - 1) * nt)
        Ujk[r] = 1 - a4 / (3 * a2 ** 2)
    err = np.sqrt((R - 1) / R * ((Ujk - Ujk.mean()) ** 2).sum())
    return U, err

def tau_int(series_list, maxlag=200):
    # series_list: list of 1D arrays (one per replica); average normalized autocorrelation
    acs = []
    for y in series_list:
        y = y - y.mean()
        n = len(y)
        f = np.fft.rfft(y, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        ac = ac / ac[0]
        acs.append(ac[:maxlag])
    ac = np.mean(acs, axis=0)
    # Sokal window
    t_int = 0.5
    for t in range(1, maxlag):
        t_int += ac[t]
        if t >= 6 * t_int:
            break
    return t_int, ac

N = 16
for T in [1.5, 2.269, 2.8]:
    t0 = time.time()
    R, nsteps, burn = 256, 3000, 200
    Mt, _ = wolff_run(N, T, R, nsteps, seed=int(1000 * T) + 1)
    U, err = analyze(Mt, burn)
    x = Mt[burn:].astype(np.float64) / 256.
    m2s = x ** 2
    # autocorrelation of m2 and |m|
    tau_m2, _ = tau_int([m2s[:, r] for r in range(0, R, 4)])
    tau_am, _ = tau_int([np.abs(x[:, r]) for r in range(0, R, 4)])
    # magnetization drift check: mean m2 in early vs late steps
    early = (Mt[:200].astype(np.float64) / 256.) ** 2
    print(f"T={T}: U={U:.5f} +- {err:.5f} (R={R}, steps={nsteps - burn}); tau_int(m2)={tau_m2:.2f}, tau_int(|m|)={tau_am:.2f}; early m2 means per 25 steps:",
          np.round(early.reshape(8, 25, R).mean(axis=(1, 2)), 4), f"time {time.time()-t0:.1f}s")
