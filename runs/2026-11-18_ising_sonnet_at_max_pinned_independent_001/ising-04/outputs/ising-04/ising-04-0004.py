
import sys, time, json
import numpy as np
sys.path.insert(0, "/work")
from sw import BatchSW
from ana import jackknife_U

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16
R, G = 1000, 20
neq, nmeas = 100, 100

def tau_int(x):
    """x: (T, R) series; ACF averaged over replicas with common mean. returns tau_int using self-consistent window."""
    T, Rr = x.shape
    y = x - x.mean()
    f = np.fft.rfft(y, n=2 * T, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:T].mean(axis=1)
    acf = acf / (T - np.arange(T))     # unbiased normalisation by number of pairs
    acf = acf / acf[0]
    tau = 0.5
    for t in range(1, T):
        tau += acf[t]
        if t >= 6 * tau:
            break
    return tau, acf

res = {}
t00 = time.time()
print(f"{'T':>4} | {'U_plain':>8} {'err':>7} | {'U_impr':>8} {'err':>7} | tau(m2) tau(m4) tau(im2) tau(im4) | <m2>   ")
for i, T in enumerate(Ts):
    sim = BatchSW(N, R, 20000 + i)
    # half hot, half cold
    s = sim.init_hot()
    s[R // 2:] = 1
    p = 1 - np.exp(-2 / T)
    # equilibration trace: record plain m2 vs time for hot and cold halves
    tr = np.empty((neq, R))
    for t in range(neq):
        m2, m4 = sim.measure(s)
        tr[t] = m2
        s, _, _ = sim.step(s, p)
    hot_tr, cold_tr = tr[:, :R // 2].mean(axis=1), tr[:, R // 2:].mean(axis=1)
    P2 = np.empty((nmeas, R)); P4 = np.empty((nmeas, R)); I2 = np.empty((nmeas, R)); I4 = np.empty((nmeas, R))
    for t in range(nmeas):
        P2[t], P4[t] = sim.measure(s)
        s, I2[t], I4[t] = sim.step(s, p)
    up, ujp, ep, n = jackknife_U(P2, P4, G)
    ui, uji, ei, n = jackknife_U(I2, I4, G)
    taus = [tau_int(x)[0] for x in (P2, P4, I2, I4)]
    res[T] = dict(Up=up, ep=ep, Ui=ui, ei=ei, taus=taus, m2=P2.mean())
    print(f"{T:4.1f} | {up:8.5f} {ep:7.5f} | {ui:8.5f} {ei:7.5f} | " + " ".join(f"{x:6.2f}" for x in taus) + f" | {P2.mean():.4f}")
    if T in (1.5, 2.0, 2.3, 2.8):
        ts = [0, 1, 2, 3, 5, 10, 20, 50, 99]
        print("     equil <m2>(t) hot :", " ".join(f"{hot_tr[t]:.4f}" for t in ts))
        print("     equil <m2>(t) cold:", " ".join(f"{cold_tr[t]:.4f}" for t in ts))
print("total time", time.time() - t00)
json.dump({str(k): v for k, v in res.items()}, open("/work/pilot.json", "w"))
