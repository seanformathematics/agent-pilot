
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import sw

# ---------- Kaufman exact partition function on an N x N torus ----------
from scipy.special import logsumexp
def logZ_kaufman(K, N):
    # gamma_l for l = 0..2N-1
    l = np.arange(2 * N)
    cosh_g = np.cosh(2 * K) ** 2 / np.sinh(2 * K) - np.cos(np.pi * l / N)
    g = np.arccosh(cosh_g)
    # l=0: signed gamma_0 = 2K + ln tanh K
    g[0] = 2 * K + np.log(np.tanh(K))
    godd = g[1::2]; geven = g[0::2]
    def lc(x):  # log of prod 2cosh(N x/2)
        return np.sum(np.log(2 * np.cosh(N * x / 2)))
    def ls(x):  # log|prod 2 sinh(N x/2)| and sign
        v = 2 * np.sinh(N * x / 2)
        return np.sum(np.log(np.abs(v))), np.prod(np.sign(v))
    z1 = lc(godd)
    z2, s2 = ls(godd)
    z3 = lc(geven)
    z4, s4 = ls(geven)
    terms = np.array([z1, z2, z3, z4]); signs = np.array([1.0, s2, 1.0, s4])
    m = terms.max()
    tot = np.sum(signs * np.exp(terms - m))
    return N * N / 2 * np.log(2 * np.sinh(2 * K)) + np.log(0.5) + m + np.log(tot)

def e_exact(T, N, h=1e-5):
    K = 1.0 / T
    return -(logZ_kaufman(K + h, N) - logZ_kaufman(K - h, N)) / (2 * h) / (N * N)

# verify with 4x4 enumeration
N = 4; n = 16
configs = np.arange(2 ** n)
bits = (((configs[:, None] >> np.arange(n)) & 1).astype(np.int8) * 2 - 1)
spins = bits.reshape(-1, N, N)
E = -((spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2)) + (spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2)))
for T in [1.5, 2.3, 2.8, 3.5]:
    K = 1 / T
    lz_enum = logsumexp(-E * K)
    print(f"4x4 T={T}: logZ enum={lz_enum:.10f} kaufman={logZ_kaufman(K,4):.10f}  e_enum={(np.exp(-E*K - lz_enum)*E).sum()/16:.8f} e_kauf={e_exact(T,4):.8f}")

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
print("exact e(T), N=16:", {T: round(e_exact(T, 16), 6) for T in Ts})

# ---------- equilibration: hot vs cold at N=16 ----------
R = 1000
for T in [1.5, 1.9, 2.269, 2.8]:
    hot = sw.SwendsenWang(16, R, T, 31, start="hot")
    cold = sw.SwendsenWang(16, R, T, 32, start="cold")
    rec = []
    for it in range(1, 61):
        oh = hot.sweep(measure=True, energy=True)
        oc = cold.sweep(measure=True, energy=True)
        if it in (1, 2, 3, 5, 8, 12, 20, 40, 60):
            d2 = oh["m2"].mean() - oc["m2"].mean()
            er2 = np.sqrt(oh["m2"].var() / R + oc["m2"].var() / R)
            de = oh["e"].mean() - oc["e"].mean()
            ere = np.sqrt(oh["e"].var() / R + oc["e"].var() / R)
            rec.append(f"it{it}: dm2={d2:+.4f}({d2/er2:+.1f}s) de={de:+.4f}({de/ere:+.1f}s)")
    print(f"T={T}: " + " | ".join(rec))

# ---------- pilot at T=2.3 ----------
T = 2.3; R = 1000; neq = 100; nm = 1000
sim = sw.SwendsenWang(16, R, T, 41, start="hot")
for _ in range(neq):
    sim.sweep(measure=False)
ts = {k: np.zeros((nm, R)) for k in ["m2", "m4", "m2i", "m4i", "e"]}
t0 = time.time()
for it in range(nm):
    o = sim.sweep(measure=True, energy=True)
    for k in ts:
        ts[k][it] = o[k]
print("pilot time", time.time() - t0)
a = ts["m2"].mean(axis=0); b = ts["m4"].mean(axis=0)
U1, e1 = sw.jackknife_U(a, b, 100)
ai = ts["m2i"].mean(axis=0); bi = ts["m4i"].mean(axis=0)
U2, e2 = sw.jackknife_U(ai, bi, 100)
print(f"T=2.3 pilot: U plain={U1:.5f}+-{e1:.5f}, improved={U2:.5f}+-{e2:.5f}, e={ts['e'].mean():.5f}")
# autocorrelation of m2 within replicas (time-series)
x = ts["m2"] - ts["m2"].mean()
var = (x * x).mean()
tau = 0.5
for lag in range(1, 40):
    c = (x[:-lag] * x[lag:]).mean() / var
    tau += c
    if lag in (1, 2, 3, 5, 10, 20, 39):
        print(f" lag {lag}: C={c:.4f}  tau_int(running)={tau:.3f}")
