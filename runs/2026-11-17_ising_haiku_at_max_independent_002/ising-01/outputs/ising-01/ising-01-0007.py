
import numpy as np, time, sys, json, os
sys.path.insert(0, '/work')
from ising import wolff_run

N = 16
R = 512
NSTEPS = 6000
BURN = 400

def analyze_sums(Mt, N=16):
    x = Mt.astype(np.float64) / (N * N)
    m2 = x ** 2; m4 = x ** 4
    nt, R_ = x.shape
    S2 = m2.sum(axis=0); S4 = m4.sum(axis=0)
    tot2 = S2.sum(); tot4 = S4.sum(); tot = nt * R_
    B = tot2 / tot; A = tot4 / tot
    U = 1 - A / (3 * B * B)
    B_r = (tot2 - S2) / ((R_ - 1) * nt)
    A_r = (tot4 - S4) / ((R_ - 1) * nt)
    Ujk = 1 - A_r / (3 * B_r ** 2)
    err = np.sqrt((R_ - 1) / R_ * np.sum((Ujk - Ujk.mean()) ** 2))
    return U, err, B, A

os.makedirs('/work/results', exist_ok=True)
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
t_start = time.time()
for T in Ts:
    t0 = time.time()
    seed = 900000 + int(round(T * 100))
    Mt, _ = wolff_run(N, T, R, NSTEPS, seed=seed)
    Mt = Mt[BURN:]
    U, err, B, A = analyze_sums(Mt)
    res = dict(T=T, U=float(U), U_err=float(err), n_samples=int(Mt.shape[0] * Mt.shape[1]),
               m2=float(B), m4=float(A), seed=seed, time_s=round(time.time() - t0, 1))
    with open(f'/work/results/rep_T_{T:.2f}.json', 'w') as f:
        json.dump(res, f)
    print(res, flush=True)
print("replicate part1 total time", round(time.time() - t_start, 1))
