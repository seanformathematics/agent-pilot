import sys, time, json, numpy as np
sys.path.insert(0, '/work')
import ising
N = 16
Ts = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
out = {}
t0 = time.time()
for idx, T in enumerate(Ts):
    seed = 30000 + idx
    M, M2, M4, E, s = ising.run_chains(N, T, R=1000, n_equil=2000, n_meas=4000, seed=seed, init='random')
    U, err = ising.binder_jack(M2, M4)
    U_a, _ = ising.binder_jack(M2[:, :500], M4[:, :500])
    U_b, _ = ising.binder_jack(M2[:, 500:], M4[:, 500:])
    out[T] = dict(seed=seed, U=U, err=err, U_half1=U_a, U_half2=U_b, E=float(E.mean()))
    print(f"T={T:.2f} seed={seed} U={U:.5f} err={err:.5f} halves=({U_a:.5f},{U_b:.5f}) E={E.mean():.4f} t={time.time()-t0:.1f}s", flush=True)
    with open('/work/prod_part2.json', 'w') as f:
        json.dump({str(k): v for k, v in out.items()}, f, indent=1)
