import sys, time, math
sys.path.insert(0, '/work')
import numpy as np
import ising

def U_from_series(M):
    return ising.binder_from_M(M, 256)

# (a) block-size dependence of the jackknife error on saved series
for T in [2.5, 2.6, 2.7, 2.8]:
    M = np.load(f'/work/M_T{T:.2f}.npy').astype(np.int64)
    n = M.size
    row = []
    for nblk in [4000, 1000, 250, 100, 40]:
        U, Ujk, e = ising.binder_jk(M, nblk)
        row.append(f"nblk={nblk}(L={n//nblk}): err={e:.6f}")
    print(f"T={T}: U={U:.6f} | " + " | ".join(row), flush=True)

# (b) integrated autocorrelation time of the per-sample quantities used in U
for T in [2.3, 2.8]:
    M = np.load(f'/work/M_T{T:.2f}.npy').astype(np.float64)
    m2 = M**2; m4 = m2**2
    S2 = m2.sum(); S4 = m4.sum(); n = M.size
    # delta-method linearisation: f_i = 1 - n*S4/(3 S2^2) -> per-sample influence
    # g_i = -(1/3)*(m4_i*n/S2^2 - 2*S4*n*m2_i/S2^3) ... use U = 1 - (S4/n)/(3 (S2/n)^2)
    m2b = m2.mean(); m4b = m4.mean()
    g = -(1.0/3.0)*(m4/(m2b**2) - 2*m4b*m2/(m2b**3))   # influence function of U (per sample), mean ~0
    g = g - g.mean()
    x = g
    f = np.fft.rfft(x, 2*n); acf = np.fft.irfft(f*np.conj(f))[:n]; acf /= acf[0]
    tau = 0.5; 
    for t in range(1, 2000):
        tau += acf[t]
        if t >= 6*tau: break
    var_naive = np.var(g)
    print(f"T={T}: influence-function tau_int(U)={tau:.2f} updates; "
          f"delta-method err = sqrt(tau*2*var(g)/n) = {math.sqrt(2*tau*var_naive/n):.6f}", flush=True)

# (c) direct scatter test at T=2.8: 40 fresh chains, 5e5 updates each, burn-in 5000
T = 2.8
t0 = time.time()
Us, jkerr, means = [], [], []
for k in range(40):
    M, _ = ising.wolff_chain(16, T, 500000, 5000, seed=7000000+k)
    U, Ujk, e = ising.binder_jk(M, 250)
    Us.append(U); jkerr.append(e)
Us = np.array(Us)
print(f"T=2.8 40 chains x 5e5: mean U={Us.mean():.6f}; std across chains={Us.std(ddof=1):.6f}; "
      f"std of mean of 40 chains (scatter, pooled-equivalent 2e7 samples)={Us.std(ddof=1)/np.sqrt(40):.6f}; "
      f"mean per-chain jackknife err={np.mean(jkerr):.6f} -> err for 4-chain pool={np.mean(jkerr)/2:.6f}; time {time.time()-t0:.1f}s", flush=True)
print("chain U:", np.round(Us, 5).tolist(), flush=True)
