import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np

def make_masks(N):
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    black = ((ii + jj) % 2 == 0)
    return black, ~black

def run_chains(N, T, R, n_equil, n_meas, seed, init='random', thin=1):
    """Run R independent Metropolis chains in parallel; return per-sweep m, m2, m4, e arrays (n_meas/thin, R)."""
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    black, white = make_masks(N)
    if init == 'random':
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
    elif init == 'cold':
        s = np.ones((R, N, N), dtype=np.int8)
    else:
        raise ValueError
    # acceptance table indexed by x = s*nn in {-4,-2,0,2,4}; dE = 2*x; accept prob min(1, exp(-2 beta x))
    xs = np.arange(-4, 5, 2)
    ptab = np.minimum(1.0, np.exp(-2.0 * beta * xs)).astype(np.float64)
    # index = x + 4 in {0,2,4,6,8}
    ptab_full = np.zeros(9)
    ptab_full[xs + 4] = ptab
    mask_b = black[None]
    mask_w = white[None]
    def half(s, cmask):
        nn = (np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) +
              np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2))
        x = s * nn  # int8
        p = ptab_full[(x + 4).astype(np.intp)]
        u = rng.random(s.shape)
        flip = (u < p) & cmask
        return np.where(flip, -s, s).astype(np.int8)
    def sweep(s):
        s = half(s, mask_b)
        s = half(s, mask_w)
        return s
    for _ in range(n_equil):
        s = sweep(s)
    nkeep = n_meas // thin
    M = np.zeros((nkeep, R)); M2 = np.zeros((nkeep, R)); M4 = np.zeros((nkeep, R)); E = np.zeros((nkeep, R))
    k = 0
    Nsq = N * N
    for t in range(n_meas):
        s = sweep(s)
        if (t + 1) % thin == 0:
            sf = s.reshape(R, Nsq).astype(np.float64)
            m = sf.mean(axis=1)
            M[k] = m; M2[k] = m * m; M4[k] = m ** 4
            bonds = (s * np.roll(s, -1, axis=1)).astype(np.int32).sum(axis=(1, 2)) + \
                    (s * np.roll(s, -1, axis=2)).astype(np.int32).sum(axis=(1, 2))
            E[k] = -bonds / Nsq
            k += 1
    return M, M2, M4, E, s

def binder_jack(M2, M4):
    """M2, M4: (n_samples, R) arrays. Returns U, U_err via jackknife over chains."""
    R = M2.shape[1]
    a2 = M2.mean(axis=0); a4 = M4.mean(axis=0)  # per chain
    A2 = a2.sum(); A4 = a4.sum()
    U_full = 1 - (A4 / R) / (3 * (A2 / R) ** 2)
    # leave-one-out
    A2_i = A2 - a2; A4_i = A4 - a4
    U_i = 1 - (A4_i / (R - 1)) / (3 * (A2_i / (R - 1)) ** 2)
    U_jack = U_i.mean()
    err = np.sqrt((R - 1) / R * np.sum((U_i - U_i.mean()) ** 2))
    return U_full, err
'''
with open('/work/ising.py', 'w') as f:
    f.write(code)
import sys
sys.path.insert(0, '/work')
import importlib, ising
importlib.reload(ising)

# Onsager exact energy per spin and magnetization
from scipy.special import ellipk
def onsager_e(T):
    b = 1/T
    K = 2*b
    k = 2*np.sinh(K)/np.cosh(K)**2
    return -(1/np.tanh(K))*(1 + (2/np.pi)*(2*np.tanh(K)**2 - 1)*ellipk(k*k))
def onsager_m(T):
    b = 1/T
    s = np.sinh(2*b)
    return (1 - s**-4)**(1/8) if T < 2.269185 else 0.0

N = 16
for T in [1.5, 2.0, 2.269, 2.8]:
    t0 = time.time()
    M, M2, M4, E, s = ising.run_chains(N, T, R=200, n_equil=300, n_meas=300, seed=1, init='random')
    dt = time.time() - t0
    print(f"T={T}: time {dt:.2f}s for 600 sweeps x 200 chains; E={E.mean():.4f} (Onsager inf {onsager_e(T):.4f}); |m|={np.abs(M).mean():.4f} (Onsager {onsager_m(T):.4f})")
