
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from run import *

t0 = time.time()
K = 40
for T in [2.3, 2.8]:
    Us = {"plain": [], "improved": []}; errs = {"plain": [], "improved": []}
    for k in range(K):
        acc = run_T(16, T, R=100, E=200, S=200, L=20, seed=900000 + 1000*int(round(T*10)) + k)
        r = summarize(acc)
        for name in Us:
            Us[name].append(r[name]['U']); errs[name].append(r[name]['err_rep'])
    for name in Us:
        u = np.array(Us[name]); e = np.array(errs[name])
        z = (u - u.mean())/e
        print(f"T={T} {name:8s}: mean U={u.mean():.5f}  std over repeats={u.std(ddof=1):.5f}  rms reported err={np.sqrt(np.mean(e**2)):.5f}  mean err={e.mean():.5f} "
              f"ratio std/rms_err={u.std(ddof=1)/np.sqrt(np.mean(e**2)):.3f}  (expected 1 +- {1/np.sqrt(2*(K-1)):.2f});  std of z={z.std(ddof=1):.3f}")
    # combined mean of all repeats and its error
    print(f"      -> pooled mean (plain) = {np.mean(Us['plain']):.5f} +- {np.std(Us['plain'],ddof=1)/np.sqrt(K):.5f};  (improved) = {np.mean(Us['improved']):.5f} +- {np.std(Us['improved'],ddof=1)/np.sqrt(K):.5f}")
print("time", time.time()-t0)
