
add = r'''

def run_T(N, R, T, seed, n_equil, n_meas, B=100, start="mixed", e_every=1, n_series=4):
    """Run R independent replicas at temperature T with Swendsen-Wang.
    Returns dict with per-replica block averages (blocks of B sweeps) of plain and improved m^2, m^4,
    burn-in trajectories (ensemble mean of m^2 for the hot-start half and cold-start half),
    energies and a short raw time series for autocorrelation analysis."""
    S = IsingSW(N, R, T, seed, start)
    half = R // 2
    traj_hot = np.zeros(n_equil); traj_cold = np.zeros(n_equil)
    for t in range(n_equil):
        S.sweep()
        m2 = S.magnetization() ** 2
        traj_hot[t] = m2[:half].mean(); traj_cold[t] = m2[half:].mean()
    nb = n_meas // B
    blk = np.zeros((4, R, nb))
    eblk = np.zeros((R, nb)); ecount = 0
    series = np.zeros((n_series, n_meas))   # raw m^2 series, replicas 0..n_series-1
    for b in range(nb):
        acc = np.zeros((4, R)); eacc = np.zeros(R); ne = 0
        for t in range(B):
            a, bb = S.sweep(improved=True)
            m = S.magnetization()
            m2 = m * m
            acc[0] += m2; acc[1] += m2 * m2; acc[2] += a; acc[3] += bb
            series[:, b * B + t] = m2[:n_series]
            if (t % e_every) == 0:
                eacc += S.energy_per_site(); ne += 1
        blk[:, :, b] = acc / B
        eblk[:, b] = eacc / max(ne, 1)
    return dict(blk=blk, eblk=eblk, traj_hot=traj_hot, traj_cold=traj_cold, series=series, N=N, R=R, T=T, seed=seed,
                n_equil=n_equil, n_meas=nb * B, B=B)


def U_jack(S2, S4):
    """Jackknife (delete-one replica) estimate of U = 1 - <m4>/(3<m2>^2) from per-replica means S2, S4
    (every replica contributes the same number of samples). Returns U, err."""
    R = len(S2)
    A = S2.sum(); Bs = S4.sum()
    U0 = 1.0 - (Bs / R) / (3.0 * (A / R) ** 2)
    a = (A - S2) / (R - 1); b = (Bs - S4) / (R - 1)
    Ul = 1.0 - b / (3.0 * a ** 2)
    err = np.sqrt((R - 1) / R * ((Ul - Ul.mean()) ** 2).sum())
    return U0, err
'''
open('/work/sw.py','a').write(add)

import sys, time
sys.path.insert(0,'/work')
import numpy as np, importlib
import sw, exact
importlib.reload(sw)

Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
t0=time.time()
print(" T    e_exact     e_SW      z_e   | U_plain +- err  | U_impr +- err | hot/cold m2 (last 100 equil) | tau_int(m2)")
zs=[]
for k,T in enumerate(Ts):
    r=sw.run_T(16, 400, T, seed=5000+7*k, n_equil=200, n_meas=1000, B=100)
    res[T]=r
    blk=r['blk']; R=r['R']
    s2=blk[0].mean(axis=1); s4=blk[1].mean(axis=1); i2=blk[2].mean(axis=1); i4=blk[3].mean(axis=1)
    Up,ep=sw.U_jack(s2,s4); Ui,ei=sw.U_jack(i2,i4)
    e_rep=r['eblk'].mean(axis=1); e_sw=e_rep.mean(); e_err=e_rep.std(ddof=1)/np.sqrt(R)
    e_ex=exact.energy_per_site_exact(T,16)
    z=(e_sw-e_ex)/e_err; zs.append(z)
    # tau_int from raw series (replicas 0..3), m^2
    x=r['series']-r['series'].mean(axis=1,keepdims=True)
    n=x.shape[1]
    f=np.fft.rfft(x,2*n,axis=1); ac=np.fft.irfft(f*np.conj(f),axis=1)[:,:n]; ac=ac.mean(axis=0)/ac.mean(axis=0)[0]
    W=1; tau=0.5
    for w in range(1,n//2):
        tau=0.5+ac[1:w+1].sum()
        if w>=6*tau: break
    print(f"{T:4.2f} {e_ex:10.6f} {e_sw:10.6f} {z:6.2f} | {Up:.5f} +- {ep:.5f} | {Ui:.5f} +- {ei:.5f} | {r['traj_hot'][-100:].mean():.5f}/{r['traj_cold'][-100:].mean():.5f} | {tau:.2f}")
print("time",time.time()-t0)
zs=np.array(zs); print("chi2 energy z-scores:", (zs**2).sum(), "for", len(zs), "dof")
import pickle
pickle.dump(res,open('/work/pilot16.pkl','wb'))
