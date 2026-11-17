import json, math, glob
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
points = []
seeds = []
for T in Ts:
    d = json.load(open(f'/work/prod_T{T:.1f}.json'))
    assert abs(d['T'] - T) < 1e-12
    U, e, eb = d['U'], d['err_rep'], d['err_blk']
    n = d['R'] * d['nmeas']
    seeds.append(d['seed'])
    ok = all(math.isfinite(x) for x in (U, e, eb)) and e > 0 and U <= 2/3 + 1e-12
    print(f"T={T}: U={U:.7f} err_rep={e:.3e} err_blk={eb:.3e} n={n} seed={d['seed']} tau2={d['tau_m2']:.2f} tau4={d['tau_m4']:.2f} ok={ok}")
    points.append(dict(T=T, status="ok", U=U, U_err=e, n_samples=n))
print(json.dumps(points))
print("seeds", seeds)
