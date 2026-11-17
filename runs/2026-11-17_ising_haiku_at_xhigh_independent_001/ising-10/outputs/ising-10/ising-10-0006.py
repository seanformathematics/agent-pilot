import sys, time, json, numpy as np
sys.path.insert(0, '/work')
import ising
N = 16
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
prod = {}
for fn in ['/work/prod_part1.json', '/work/prod_part2.json']:
    d = json.load(open(fn))
    prod.update({float(k): v for k, v in d.items()})
out = {}
t0 = time.time()
for idx, T in enumerate(Ts):
    seed = 40000 + idx
    init = 'cold' if idx % 2 == 0 else 'random'
    M, M2, M4, E, s = ising.run_chains(N, T, R=500, n_equil=3000, n_meas=3000, seed=seed, init=init)
    U, err = ising.binder_jack(M2, M4)
    p = prod[T]
    z = (U - p['U']) / np.sqrt(err**2 + p['err']**2) if (err + p['err']) > 0 else 0.0
    out[T] = dict(seed=seed, init=init, U=U, err=err, z_vs_prod=float(z))
    print(f"T={T:.2f} init={init} seed={seed} U={U:.5f} err={err:.5f} | prod U={p['U']:.5f} err={p['err']:.5f} | z={z:+.2f} t={time.time()-t0:.1f}s", flush=True)
with open('/work/verify.json', 'w') as f:
    json.dump({str(k): v for k, v in out.items()}, f, indent=1)
