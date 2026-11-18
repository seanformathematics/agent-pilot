import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import sw_setup, sw_sweep

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R = 512; N = 16
geo = sw_setup(R, N)
res = {}
t0 = time.time()
for k, T in enumerate(Ts):
    seed = 1000 + k
    rng = np.random.default_rng(seed)
    p = 1 - np.exp(-2.0/T)
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    for it in range(200):
        s = sw_sweep(s, p, rng, geo)
    nmeas = 800
    m2 = np.empty((nmeas, R)); m4 = np.empty((nmeas, R))
    for it in range(nmeas):
        s = sw_sweep(s, p, rng, geo)
        m = s.reshape(R, -1).mean(axis=1)
        m2[it] = m**2; m4[it] = m**4
    A = m2.mean(axis=0); B = m4.mean(axis=0)
    U = 1 - B.mean()/(3*A.mean()**2)
    # jackknife over replicas
    Ui = np.empty(R)
    Asum = A.sum(); Bsum = B.sum()
    for r in range(R):
        a = (Asum - A[r])/(R-1); b = (Bsum - B[r])/(R-1)
        Ui[r] = 1 - b/(3*a*a)
    Uerr = np.sqrt((R-1)/R * np.sum((Ui - Ui.mean())**2))
    # autocorrelation of per-sweep replica-averaged m2 and m4 (sweep series averaged over replicas)
    x = m2.mean(axis=1); 
    xc = x - x.mean()
    ac = np.correlate(xc, xc, 'full')[len(xc)-1:]/ (xc.var()*len(xc))
    tau = 0.5 + np.sum(ac[1:30])
    res[T] = dict(U=U, Uerr=Uerr, tau_m2=tau)
    np.savez(f'/work/pilot_T{T:.1f}.npz', m2=m2, m4=m4)
    print(f"T={T} U={U:.5f} Uerr={Uerr:.5f} tau_int(m2)~{tau:.2f} t={time.time()-t0:.1f}s", flush=True)
