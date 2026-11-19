
import sys, time
import numpy as np
sys.path.insert(0, "/work")

ana = r'''
import numpy as np

def U_from_sums(S2, S4, n):
    M2 = S2 / n
    M4 = S4 / n
    return 1.0 - M4 / (3.0 * M2 ** 2)

def jackknife_U(m2, m4, G):
    """m2, m4: arrays (T, R) of per-sample estimates (R independent replicas, T times).
    Replicas are split into G contiguous groups (independent chains) -> delete-one-group jackknife.
    returns (U_pooled, U_jack_biascorrected, err_jack, n_samples)"""
    T, R = m2.shape
    assert R % G == 0
    k = R // G
    s2 = m2.reshape(T, G, k).sum(axis=(0, 2))
    s4 = m4.reshape(T, G, k).sum(axis=(0, 2))
    ng = T * k
    n = ng * G
    U_all = U_from_sums(s2.sum(), s4.sum(), n)
    Uloo = U_from_sums(s2.sum() - s2, s4.sum() - s4, n - ng)
    Um = Uloo.mean()
    var = (G - 1) / G * np.sum((Uloo - Um) ** 2)
    Ujack = G * U_all - (G - 1) * Um
    return U_all, Ujack, np.sqrt(var), n
'''
open("/work/ana.py", "w").write(ana)

from sw import BatchSW
from ana import jackknife_U

# ---------- exact enumeration for N=4 ----------
N = 4
nsp = N * N
confs = ((np.arange(1 << nsp)[:, None] >> np.arange(nsp)[None, :]) & 1) * 2 - 1
confs = confs.reshape(-1, N, N).astype(np.int64)
E = -(confs * np.roll(confs, -1, axis=2)).sum(axis=(1, 2)) - (confs * np.roll(confs, -1, axis=1)).sum(axis=(1, 2))
M = confs.sum(axis=(1, 2)) / nsp
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
exact = {}
for T in Ts:
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    m2 = (w * M ** 2).sum() / Z
    m4 = (w * M ** 4).sum() / Z
    exact[T] = 1 - m4 / (3 * m2 ** 2)

# ---------- MC with SW ----------
R, G = 4000, 20
neq, nmeas = 50, 200
print(f"{'T':>4} {'exact':>9} | {'plain U':>9} {'err':>7} {'z':>5} | {'impr U':>9} {'err':>7} {'z':>5}")
t0 = time.time()
for i, T in enumerate(Ts):
    sim = BatchSW(N, R, 1000 + i)
    s = sim.init_hot()
    p = 1 - np.exp(-2 / T)
    for _ in range(neq):
        s, _, _ = sim.step(s, p)
    P2 = np.empty((nmeas, R)); P4 = np.empty((nmeas, R)); I2 = np.empty((nmeas, R)); I4 = np.empty((nmeas, R))
    for t in range(nmeas):
        P2[t], P4[t] = sim.measure(s)
        s, I2[t], I4[t] = sim.step(s, p)
    up, ujp, ep, n = jackknife_U(P2, P4, G)
    ui, uji, ei, n = jackknife_U(I2, I4, G)
    print(f"{T:4.1f} {exact[T]:9.5f} | {up:9.5f} {ep:7.5f} {(up-exact[T])/ep:5.1f} | {ui:9.5f} {ei:7.5f} {(ui-exact[T])/ei:5.1f}")
print("time", time.time() - t0)
