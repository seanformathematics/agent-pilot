
prod = r'''
import sys, time
import numpy as np
sys.path.insert(0, '/work')
from sw import SWReplicas
from ana import jackknife_U

N = 16; nsite = N * N
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 1000; Rh = 500; nburn = 200; nmeas = 2000; blk = 100
SEED_BASE = 880000

def acf_mean(X, maxlag=300):
    Rr, n = X.shape
    Xc = X - X.mean()
    F = np.fft.rfft(Xc, n=2 * n, axis=1)
    ac = np.fft.irfft((F * np.conj(F)).real, n=2 * n, axis=1)[:, :maxlag + 1]
    c = ac.sum(axis=0) / (Rr * (n - np.arange(maxlag + 1)))
    return c / c[0]

def tau_int(rho):
    tau = 0.5
    for k in range(1, len(rho)):
        tau += rho[k]
        if k >= 6 * tau:
            return tau, k
    return tau, len(rho) - 1

def run(iT):
    T = Ts[iT]
    t0 = time.time()
    rng = np.random.default_rng(SEED_BASE + iT)
    sim = SWReplicas(N, R, T, rng, "hot")
    sim.s[Rh:] = 1                                   # replicas Rh.. : cold start; 0..Rh-1: hot start
    burn = np.zeros((nburn, 2))
    for t in range(nburn):
        sim.sweep()
        mm = sim.magnetization_sum() / nsite
        burn[t] = (np.mean(mm[:Rh] ** 2), np.mean(mm[Rh:] ** 2))
    m2ts = np.zeros((R, nmeas), dtype=np.float64)
    m4ts = np.zeros((R, nmeas), dtype=np.float64)
    SE = np.zeros(R); SE2 = np.zeros(R); nE = 0
    for t in range(nmeas):
        sim.sweep()
        mm = sim.magnetization_sum() / nsite
        m2ts[:, t] = mm ** 2
        m4ts[:, t] = mm ** 4
        if t % 5 == 0:
            ee = sim.energy().astype(float); SE += ee; SE2 += ee ** 2; nE += 1
    S2 = m2ts.sum(axis=1); S4 = m4ts.sum(axis=1)
    B2 = m2ts.reshape(R, nmeas // blk, blk).sum(axis=2)
    B4 = m4ts.reshape(R, nmeas // blk, blk).sum(axis=2)
    acf2 = acf_mean(m2ts); acf4 = acf_mean(m4ts)
    m2mean_t = m2ts.mean(axis=0)
    np.savez(f"/work/res_{iT:02d}.npz", T=T, seed=SEED_BASE + iT, R=R, Rh=Rh, nburn=nburn, nmeas=nmeas, blk=blk,
             S2=S2, S4=S4, B2=B2, B4=B4, SE=SE, SE2=SE2, nE=nE, burn=burn, acf2=acf2, acf4=acf4, m2mean_t=m2mean_t)
    U, err, Ubc = jackknife_U(S2, S4, nmeas)
    tau2, w2 = tau_int(acf2)
    print(f"T={T}: U={U:.5f} +- {err:.5f} (bias-corr {Ubc:.5f}); tau_int(m2)={tau2:.2f}; [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(int(a))
'''
open('/work/prod.py','w').write(prod)
import subprocess, sys
r = subprocess.run([sys.executable, '/work/prod.py', '0','1','2','3','4','5','6'], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
