import time, sys
import numpy as np
sys.path.insert(0, "/work")
from sw_ising import run_chain, U_from_series, U_jack

L = 16; R = 2000; NSW = 1000; BURN = 200; B = 20
Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
base_seed = 20250000
sel = [7, 8, 9, 10, 11, 12, 13]
log = []
t_start = time.time()
for i in sel:
    T = Tlist[i]
    seed = base_seed + i
    t0 = time.time()
    Ms, Es = run_chain(L, R, T, NSW, seed)
    dt = time.time() - t0
    np.savez_compressed(f"/work/prod_T{T:.1f}_s{seed}.npz", Ms=Ms, Es=Es, T=T, seed=seed, R=R, NSW=NSW)
    U, Ue = U_jack(Ms, B, BURN)
    U1 = U_from_series(Ms[BURN:600]); U2 = U_from_series(Ms[600:])
    G = 10
    gU = [U_from_series(Ms[BURN:, g*R//G:(g+1)*R//G]) for g in range(G)]
    gerr = np.std(gU, ddof=1)/np.sqrt(G)
    line = (f"T={T:.2f} seed={seed} time={dt:.1f}s U={U:.5f} Uerr_jack(B={B})={Ue:.5f} "
            f"Uerr_groups={gerr:.5f} U[200:600]={U1:.5f} U[600:]={U2:.5f} E/N={Es[BURN:].mean()/L**2:.4f}")
    print(line, flush=True)
    log.append(line)
    if time.time() - t_start > 480:
        print("stopping early for time budget", flush=True)
        break
with open("/work/log_batch2.txt", "w") as f:
    f.write("\n".join(log) + "\n")
print("total", time.time()-t_start)
