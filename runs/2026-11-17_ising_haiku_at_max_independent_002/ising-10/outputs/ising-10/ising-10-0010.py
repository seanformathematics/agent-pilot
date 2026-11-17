import numpy as np, time, sys, json
sys.path.insert(0, '/work')
import swnd

def jack_replicas(A, B, n):
    R = A.shape[0]
    At, Bt = A.sum(), B.sum()
    tot = n * R
    U_full = 1 - (Bt / tot) / (3 * (At / tot) ** 2)
    Ak = (At - A) / (n * (R - 1)); Bk = (Bt - B) / (n * (R - 1))
    Uk = 1 - Bk / (3 * Ak ** 2)
    Ubar = Uk.mean()
    se = np.sqrt((R - 1) / R * np.sum((Uk - Ubar) ** 2))
    return U_full, se

out = {}
for T in [2.3, 2.4]:
    Us, ses = [], []
    t0 = time.time()
    for k in range(8):
        seed = 777000 + 1000 * k + int(round(T * 100))
        M, _ = swnd.sw_run_nd(T, R=128, n_steps=3000, n_burn=300, seed=seed, N=16)
        m = M.astype(np.float64) / 256.0
        A = (m ** 2).sum(axis=0); B = (m ** 4).sum(axis=0)
        U, se = jack_replicas(A, B, 3000)
        Us.append(U); ses.append(se)
    Us = np.array(Us); ses = np.array(ses)
    print(f"T={T}: repeats U = {np.round(Us,5)}")
    print(f"   mean U={Us.mean():.5f}, scatter std(U)={Us.std(ddof=1):.5f} (+- {Us.std(ddof=1)/np.sqrt(2*(len(Us)-1)):.5f}), mean jackknife SE={ses.mean():.5f}, time={time.time()-t0:.0f}s")
    out[T] = dict(Us=Us.tolist(), ses=ses.tolist())
json.dump(out, open('/work/results/calibration.json', 'w'), indent=1)
