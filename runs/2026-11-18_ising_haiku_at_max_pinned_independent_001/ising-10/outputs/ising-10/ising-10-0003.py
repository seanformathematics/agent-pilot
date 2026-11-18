
import time, sys, numpy as np
sys.path.insert(0, '/work')
import sw

def run_series(L, R, T, seed, burn, nsweep):
    eng = sw.SWEngine(L, R, T, seed)
    ts_m2 = []
    ts_m4 = []
    B2 = np.zeros(R); B4 = np.zeros(R)
    for t in range(burn + nsweep):
        eng.sweep()
        m = eng.spins.sum(axis=1)/eng.N
        m2 = m*m
        ts_m2.append(m2.mean()); ts_m4.append((m2*m2).mean())
        if t >= burn:
            B2 += m2; B4 += m2*m2
    return B2, B4, np.array(ts_m2), np.array(ts_m4)

def tau_int(x, maxlag=200):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    c0 = np.dot(x, x)/n
    taus = 0.5
    out = []
    for k in range(1, min(maxlag, n//4)):
        ck = np.dot(x[:-k], x[k:])/(n-k)/c0
        taus += ck
        out.append(taus)
        if k > 3*taus:
            break
    return taus

t0 = time.time()
for T in [1.5, 2.0, 2.269, 2.8]:
    seed = int(round(T*1000)) + 99
    B2, B4, s2, s4 = run_series(16, 1000, T, seed, burn=100, nsweep=1000)
    U, err = sw.binder_from_sums(B2, B4, 1000)
    tau = tau_int(s2[100:])
    print(f"T={T}: U={U:.5f} err(replica-jk)={err:.5f}  tau_int(m2 avg over replicas)={tau:.2f}")
    print("   m2 mean over sweeps windows (burn region):", np.round(s2[[0,5,10,20,40,60,80,99]], 4))
    print("   m2 mean  post-burn windows:", np.round([s2[100+i*200:300+i*200].mean() for i in range(4)], 5))
    print("   time so far", round(time.time()-t0, 1))
