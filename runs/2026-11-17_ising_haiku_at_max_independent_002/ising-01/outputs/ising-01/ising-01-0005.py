
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

def tau_int_m2(Mt, N=16, nrep=64, maxlag=100):
    x = Mt.astype(np.float64) / (N * N)
    y_all = x ** 2
    nt = y_all.shape[0]
    acs = []
    for r in range(0, min(nrep, y_all.shape[1])):
        y = y_all[:, r] - y_all[:, r].mean()
        f = np.fft.rfft(y, 2 * nt)
        ac = np.fft.irfft(f * np.conj(f))[:nt]
        acs.append(ac[:maxlag] / ac[0])
    ac = np.mean(acs, axis=0)
    t_int = 0.5
    for t in range(1, maxlag):
        t_int += ac[t]
        if t >= 6 * t_int:
            break
    return t_int

os.makedirs('/work/results', exist_ok=True)
Ts = [1.5, 1.6, 1.7, 1.8, 1.9]
t_start = time.time()
for T in Ts:
    t0 = time.time()
    seed = 500000 + int(round(T * 100))
    Mt, _ = wolff_run(N, T, R, NSTEPS, seed=seed)
    Mt = Mt[BURN:]
    U, err, B, A = analyze_sums(Mt)
    # sub-batch consistency check: 4 groups of 128 replicas
    subU = [analyze_sums(Mt[:, g * 128:(g + 1) * 128])[0] for g in range(4)]
    sd_sub = np.std(subU, ddof=1)
    tau = tau_int_m2(Mt)
    res = dict(T=T, U=float(U), U_err=float(err), n_samples=int(Mt.shape[0] * Mt.shape[1]),
               m2=float(B), m4=float(A), seed=seed, sd_subbatch_over2=float(sd_sub / 2), tau_int_m2=float(tau),
               time_s=round(time.time() - t0, 1))
    with open(f'/work/results/T_{T:.2f}.json', 'w') as f:
        json.dump(res, f)
    print(res, flush=True)
print("part1 total time", round(time.time() - t_start, 1))
