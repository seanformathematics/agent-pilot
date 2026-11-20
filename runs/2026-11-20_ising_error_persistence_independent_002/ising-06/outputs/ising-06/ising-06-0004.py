
import time, numpy as np
N, R = 16, 1000
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
masks = [((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, n_eq=4000, n_meas=15000):
    rng = np.random.default_rng(seed)
    beta = 1.0/T
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    acc = {4: np.exp(-4*beta), 8: np.exp(-8*beta)}
    def sweep(s):
        for m in masks:
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nb
            p = np.where(dE<=0, 1.0, np.where(dE==4, acc[4], np.where(dE==8, acc[8], 0.0)))
            flip = m[None] & (rng.random(s.shape) < p)
            s[flip] *= -1
        return s
    for _ in range(n_eq): sweep(s)
    S2 = np.zeros((2,R)); S4 = np.zeros((2,R))
    h = n_meas//2
    for k in range(n_meas):
        sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        S2[k//h] += m2; S4[k//h] += m2*m2
    return S2/h, S4/h
def jk_U(a2, a4):
    M2 = a2.mean(); M4 = a4.mean(); U = 1 - M4/(3*M2*M2)
    n = len(a2); t2=a2.sum(); t4=a4.sum()
    q2=(t2-a2)/(n-1); q4=(t4-a4)/(n-1)
    Uj = 1 - q4/(3*q2*q2)
    return U, np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
for T, seed in [(2.2, 104), (2.3, 105), (2.4, 106)]:
    t=time.time()
    A2, A4 = run(T, seed)
    a2 = A2.mean(0); a4 = A4.mean(0)
    np.save(f'/work/a2_T{T}.npy', a2); np.save(f'/work/a4_T{T}.npy', a4)
    np.save(f'/work/h2_T{T}.npy', A2); np.save(f'/work/h4_T{T}.npy', A4)
    U, err = jk_U(a2, a4)
    U1,_ = jk_U(A2[0],A4[0]); U2,_ = jk_U(A2[1],A4[1])
    print(T, "U=%.5f err=%.5f halves: %.5f %.5f <m2>=%.4f time=%.0fs"%(U, err, U1, U2, a2.mean(), time.time()-t), flush=True)
