import sys
sys.path.insert(0, '/work')
import ising_lib as L
import numpy as np, json, time

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16; R = 2000; nburn = 500; nmeas = 2000
t_all = time.time()
for i, T in enumerate(Ts):
    if i < 7:
        continue
    seed = 5000 + i
    m2, m4, hist, el, itm = L.run_chain(N, R, T, nburn, nmeas, seed)
    U, e_rep = L.jack_U_rep(m2.sum(1), m4.sum(1), nmeas)
    U_b, e_blk = L.block_jk(m2, m4, 100)
    tau2 = L.tau_int(m2); tau4 = L.tau_int(m4)
    blk = [round(float(hist[a:b].mean()), 5) for a, b in [(0, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2500)]]
    res = dict(T=T, N=N, R=R, nburn=nburn, nmeas=nmeas, seed=seed, U=float(U), err_rep=float(e_rep),
               U_blockjk=float(U_b), err_blk=float(e_blk), tau_m2=tau2, tau_m4=tau4,
               m2=float(m2.mean()), m4=float(m4.mean()), iters=itm, time_s=el, hist_blocks=blk)
    json.dump(res, open(f'/work/prod_T{T:.1f}.json', 'w'), indent=1)
    print(f"T={T}: U={U:.5f} err_rep={e_rep:.5f} U_blk={U_b:.5f} err_blk={e_blk:.5f} tau2={tau2:.2f} tau4={tau4:.2f} <m2>={m2.mean():.4f} <m4>={m4.mean():.4f} iters={itm:.1f} time={el:.0f}s burn-check={blk}", flush=True)
print("total time", time.time() - t_all, flush=True)
