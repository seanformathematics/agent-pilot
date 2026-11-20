
import time, numpy as np, os, json
os.makedirs('/work', exist_ok=True)
N, R = 16, 1000
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
masks = [((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, n_eq=3000, n_meas=15000):
    rng = np.random.default_rng(seed)
    beta = 1.0/T
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    # precompute acceptance table for dE = 4,8 (dE in {4,8} when positive)
    acc_tab = {4: np.exp(-4*beta), 8: np.exp(-8*beta)}
    def sweep(s):
        for m in masks:
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nb
            p = np.where(dE<=0, 1.0, np.where(dE==4, acc_tab[4], np.where(dE==8, acc_tab[8], 0.0)))
            flip = m[None] & (rng.random(s.shape) < p)
            s[flip] *= -1
        return s
    for _ in range(n_eq): sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R)
    for _ in range(n_meas):
        sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        S2 += m2; S4 += m2*m2
    return S2/n_meas, S4/n_meas
def jk_U(a2, a4):
    # a2,a4: per-replica averages; U = 1 - <m4>/(3<m2>^2), jackknife over replicas
    M2 = a2.mean(); M4 = a4.mean()
    U = 1 - M4/(3*M2*M2)
    n = len(a2)
    tot2 = a2.sum(); tot4 = a4.sum()
    Uj = np.empty(n)
    for k in range(n):
        q2 = (tot2-a2[k])/(n-1); q4 = (tot4-a4[k])/(n-1)
        Uj[k] = 1 - q4/(3*q2*q2)
    err = np.sqrt((n-1)/n * np.sum((Uj-Uj.mean())**2))
    return U, err
res = {}
for T, seed in [(1.9, 101), (2.0, 102), (2.1, 103)]:
    t=time.time()
    a2, a4 = run(T, seed)
    np.save(f'/work/a2_T{T}.npy', a2); np.save(f'/work/a4_T{T}.npy', a4)
    U, err = jk_U(a2, a4)
    res[T] = (U, err)
    print(T, "U=%.5f err=%.5f  <m2>=%.4f time=%.0fs"%(U, err, a2.mean(), time.time()-t), flush=True)
