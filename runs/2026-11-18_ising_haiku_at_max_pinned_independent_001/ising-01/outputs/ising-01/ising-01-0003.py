code = r'''
import numpy as np

def run_sw(L, T, R, n_therm, n_meas, seed, start="random", max_iter_report=False):
    """Vectorized Swendsen-Wang over R independent LxL periodic Ising lattices.
    Returns int64 arrays M2s[R], M4s[R] = sums over n_meas measurements of M^2 and M^4,
    where M = sum of spins (m = M/L^2), after n_therm burn-in sweeps.
    Also returns total cluster-labelling iterations (for timing)."""
    rng = np.random.default_rng(seed)
    NS = L * L
    NT = R * NS
    p = -np.expm1(-2.0 / T)  # 1 - exp(-2/T)
    if start == "random":
        S = (rng.integers(0, 2, size=(R, L, L), dtype=np.int8) * 2 - 1).astype(np.int8)
    else:
        S = np.ones((R, L, L), dtype=np.int8)
    base = np.arange(NT, dtype=np.int32).reshape(R, L, L)
    BIG = np.int32(NT)
    n_iter_total = 0

    def step(S):
        nonlocal n_iter_total
        eq_r = S == np.roll(S, -1, axis=2)
        act_r = eq_r & (rng.random((R, L, L), dtype=np.float32) < p)
        eq_d = S == np.roll(S, -1, axis=1)
        act_d = eq_d & (rng.random((R, L, L), dtype=np.float32) < p)
        m_r = act_r
        m_l = np.roll(act_r, 1, axis=2)
        m_d = act_d
        m_u = np.roll(act_d, 1, axis=1)
        lab = base
        while True:
            n_iter_total += 1
            A = np.minimum(lab, np.where(m_r, np.roll(lab, -1, axis=2), BIG))
            A = np.minimum(A, np.where(m_l, np.roll(lab, 1, axis=2), BIG))
            A = np.minimum(A, np.where(m_d, np.roll(lab, -1, axis=1), BIG))
            A = np.minimum(A, np.where(m_u, np.roll(lab, 1, axis=1), BIG))
            flatA = A.reshape(-1)
            new = np.minimum(A, flatA[flatA].reshape(R, L, L))  # pointer jump A[A]
            if np.array_equal(new, lab):
                break
            lab = new
        bits = rng.random(NT) < 0.5          # one fair coin per cluster (label = root id)
        flip = bits[lab.reshape(-1)].reshape(R, L, L)
        return np.where(flip, -S, S).astype(np.int8)

    for _ in range(n_therm):
        S = step(S)
    M2s = np.zeros(R, dtype=np.int64)
    M4s = np.zeros(R, dtype=np.int64)
    for _ in range(n_meas):
        S = step(S)
        M = S.sum(axis=(1, 2), dtype=np.int64)
        M2 = M * M
        M2s += M2
        M4s += M2 * M2
    return M2s, M4s, n_iter_total


def binder_pooled(M2s, M4s, n):
    """Pooled Binder cumulant from per-replica sums (n measurements per replica)."""
    R = len(M2s)
    a2 = M2s.sum() / (R * n)
    a4 = M4s.sum() / (R * n)
    return 1.0 - a4 / (3.0 * a2 * a2)


def binder_jackknife(M2s, M4s, n):
    """Jackknife over independent replicas. Returns (U_pooled, U_err, U_jk)."""
    R = len(M2s)
    S2 = M2s.sum(); S4 = M4s.sum()
    U = 1.0 - (S4 / (R * n)) / (3.0 * (S2 / (R * n)) ** 2)
    U_minus = np.empty(R)
    for r in range(R):
        s2 = (S2 - M2s[r]) / ((R - 1) * n)
        s4 = (S4 - M4s[r]) / ((R - 1) * n)
        U_minus[r] = 1.0 - s4 / (3.0 * s2 * s2)
    Ubar = U_minus.mean()
    U_jk = R * U - (R - 1) * Ubar
    err = np.sqrt((R - 1) / R * np.sum((U_minus - Ubar) ** 2))
    return U, err, U_jk
'''
with open("/work/sw.py", "w") as f:
    f.write(code)

test = r'''
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from sw import run_sw, binder_pooled, binder_jackknife

# exact enumeration for L=4
def exact_U(L, T):
    NS = L*L
    states = np.arange(2**NS, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(NS)) & 1).astype(np.int64)
    s = (2*bits - 1).reshape(-1, L, L)
    E = np.zeros(len(states))
    for axis in (1, 2):
        E -= (s * np.roll(s, -1, axis=axis)).sum(axis=(1, 2))
    M = s.sum(axis=(1, 2)).astype(float)
    logw = -(E - E.min()) / T
    w = np.exp(logw); w /= w.sum()
    m2 = (w * M**2).sum(); m4 = (w * M**4).sum()
    return 1 - m4 / (3 * m2 * m2), m2 / NS**2

for T in [1.5, 2.0, 2.27, 3.0]:
    Ue, m2e = exact_U(4, T)
    t0 = time.time()
    M2s, M4s, nit = run_sw(4, T, R=512, n_therm=100, n_meas=2000, seed=12345)
    U, err, Ujk = binder_jackknife(M2s, M4s, 2000)
    print(f"L=4 T={T}: exact U={Ue:.5f}  SW U={U:.5f} +- {err:.5f}  (jk {Ujk:.5f})  z={(U-Ue)/err:.2f}  time={time.time()-t0:.1f}s  avg iters/step={nit/(2100):.1f}")
'''
with open("/work/test_exact.py", "w") as f:
    f.write(test)
print("written")
