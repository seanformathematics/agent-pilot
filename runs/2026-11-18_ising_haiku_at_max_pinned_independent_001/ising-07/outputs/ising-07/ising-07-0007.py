import time, sys
import numpy as np
sys.path.insert(0, "/work")
from sw_ising import run_chain, U_from_series, U_jack

L = 16; R = 2000; NSW = 1000; BURN = 200; B = 20
Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
sel = [0, 1, 2, 3, 4, 5, 6]
log = []
t_start = time.time()
for i in sel:
    T = Tlist[i]
    seed = 30000100 + i
    t0 = time.time()
    Ms, Es = run_chain(L, R, T, NSW, seed, init="ordered")
    dt = time.time() - t0
    np.savez_compressed(f"/work/setB_T{T:.1f}_s{seed}.npz", Ms=Ms, Es=Es, T=T, seed=seed, R=R, NSW=NSW, init="ordered")
    U, Ue = U_jack(Ms, B, BURN)
    line = f"T={T:.2f} seed={seed} init=ordered time={dt:.1f}s U={U:.5f} Uerr_jack(B={B})={Ue:.5f} E/N={Es[BURN:].mean()/L**2:.4f}"
    print(line, flush=True)
    log.append(line)
with open("/work/log_setB_batch1.txt", "w") as f:
    f.write("\n".join(log) + "\n")
print("total", time.time()-t_start)
