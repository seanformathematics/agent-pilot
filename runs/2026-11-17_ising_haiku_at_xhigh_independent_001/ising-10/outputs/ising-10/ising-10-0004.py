import sys, time, json, numpy as np
sys.path.insert(0, '/work')
import ising
N = 16
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
out = {}
t0 = time.time()
for idx, T in enumerate(Ts):
    seed = 20000 + idx
    M, M2, M4, E, s = ising.run_chains(N, T, R=1000, n_equil=2000, n_meas=4000, seed=seed, init='random')
    U, err = ising.binder_jack(M2, M4)
    # block-check: split chains into halves to see consistency
    U_a, _ = ising.binder_jack(M2[:, :500], M4[:, :500])
    U_b, _ = ising.binder_jack(M2[:, 500:], M4[:, 500:])
    out[T] = dict(seed=seed, U=U, err=err, U_half1=U_a, U_half2=U_b, E=float(E.mean()))
    print(f"T={T:.2f} seed={seed} U={U:.5f} err={err:.5f} halves=({U_a:.5f},{U_b:.5f}) E={E.mean():.4f} t={time.time()-t0:.1f}s", flush=True)
    with open('/work/prod_part1.json', 'w') as f:
        json.dump({str(k): v for k, v in out.items()}, f, indent=1)
