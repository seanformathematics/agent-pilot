
import sys, time
import numpy as np
sys.path.insert(0, "/work")
from sw import BatchSW
from ana import jackknife_U

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

# ---------- exact N=4 ----------
N = 4; nsp = N * N
confs = ((np.arange(1 << nsp)[:, None] >> np.arange(nsp)[None, :]) & 1) * 2 - 1
confs = confs.reshape(-1, N, N).astype(np.int64)
E4 = -(confs * np.roll(confs, -1, axis=2)).sum(axis=(1, 2)) - (confs * np.roll(confs, -1, axis=1)).sum(axis=(1, 2))
M4_ = confs.sum(axis=(1, 2)) / nsp
exact = {}
for T in Ts:
    w = np.exp(-(E4 - E4.min()) / T); Z = w.sum()
    m2 = (w * M4_ ** 2).sum() / Z; m4 = (w * M4_ ** 4).sum() / Z
    exact[T] = 1 - m4 / (3 * m2 ** 2)

# ---------- Kaufman exact lnZ for N x N torus ----------
def lnZ_kaufman(N, K):
    s2k, c2k = np.sinh(2 * K), np.cosh(2 * K)
    l = np.arange(2 * N)
    cg = c2k ** 2 / s2k - np.cos(np.pi * l / N)
    gam = np.arccosh(np.maximum(cg, 1.0))
    gam[0] = 2 * K + np.log(np.tanh(K))   # signed
    odd = gam[1::2]; even = gam[0::2]
    def lncosh2(x): x = np.abs(x); return x + np.log1p(np.exp(-2 * x))
    def lnsinh2(x): x = np.abs(x); return x + np.log1p(-np.exp(-2 * x))
    a1 = lncosh2(N * odd / 2).sum()
    a2 = lnsinh2(N * odd / 2).sum()
    a3 = lncosh2(N * even / 2).sum()
    a4 = lnsinh2(N * even / 2).sum(); sgn4 = np.sign(even[0])
    terms = np.array([a1, a2, a3, a4]); signs = np.array([1, 1, 1, sgn4])
    mx = terms.max()
    tot = (signs * np.exp(terms - mx)).sum()
    return np.log(0.5) + 0.5 * N * N * np.log(2 * s2k) + mx + np.log(tot)

def e_exact(N, T, h=1e-5):
    K = 1.0 / T
    d = (lnZ_kaufman(N, K + h) - lnZ_kaufman(N, K - h)) / (2 * h)   # dlnZ/dK = <sum s s> = -<E>
    return -d / (N * N)       # <E>/N^2

# check against N=4 enumeration
for T in (1.5, 2.0, 2.27, 2.8):
    w = np.exp(-(E4 - E4.min()) / T); Z = w.sum()
    lnZ_enum = np.log(Z) - E4.min() / T
    print(f"N=4 T={T}: lnZ enum {lnZ_enum:.10f} kaufman {lnZ_kaufman(4, 1/T):.10f}   <e> enum {(w*E4).sum()/Z/16:.8f} kaufman {e_exact(4, T):.8f}")

# ---------- (1) high statistics N=4 SW test ----------
R, G = 4000, 20
neq, nmeas = 50, 2000
zp, zi = [], []
t0 = time.time()
print(f"{'T':>4} {'exact':>9} | {'plain U':>9} {'err':>7} {'z':>5} | {'impr U':>9} {'err':>7} {'z':>5}")
for i, T in enumerate(Ts):
    sim = BatchSW(N, R, 5000 + i)
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
    zp.append((up - exact[T]) / ep); zi.append((ui - exact[T]) / ei)
    print(f"{T:4.1f} {exact[T]:9.5f} | {up:9.5f} {ep:7.5f} {zp[-1]:5.1f} | {ui:9.5f} {ei:7.5f} {zi[-1]:5.1f}")
print("mean z plain %.2f  improved %.2f  (se ~ %.2f)  chi2/14 plain %.2f impr %.2f" % (np.mean(zp), np.mean(zi), 1/np.sqrt(14), np.mean(np.square(zp)), np.mean(np.square(zi))))
print("time", time.time() - t0)

# ---------- (2) N=16 energy: SW vs exact (Kaufman) ----------
N = 16
print("N=16 energy check  T, exact <e>, SW <e> +- err")
for i, T in enumerate(Ts):
    R, G = 500, 20
    sim = BatchSW(N, R, 7000 + i)
    s = sim.init_hot(); p = 1 - np.exp(-2 / T)
    for _ in range(50):
        s, _, _ = sim.step(s, p)
    nm = 100
    ee = np.empty((nm, R))
    for t in range(nm):
        ee[t] = (-(s * np.roll(s, -1, axis=2)).sum(axis=(1, 2), dtype=np.int64) - (s * np.roll(s, -1, axis=1)).sum(axis=(1, 2), dtype=np.int64)) / (N * N)
        s, _, _ = sim.step(s, p)
    gm = ee.reshape(nm, G, R // G).mean(axis=(0, 2))
    est, err = gm.mean(), gm.std(ddof=1) / np.sqrt(G)
    ex = e_exact(N, T)
    print(f"{T:4.1f} {ex:10.6f} {est:10.6f} +- {err:8.6f}  z={(est-ex)/err:5.2f}")
