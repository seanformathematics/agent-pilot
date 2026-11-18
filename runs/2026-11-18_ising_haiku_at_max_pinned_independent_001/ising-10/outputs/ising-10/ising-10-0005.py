
import sys, time, numpy as np
sys.path.insert(0, '/work')
import sw

prod = r'''
import numpy as np, sys
sys.path.insert(0, "/work")
import sw

def run_T(T, seed, R=1000, burn=200, S=2500, L=16):
    eng = sw.SWEngine(L, R, T, seed)
    for _ in range(burn):
        eng.sweep()
    B2 = np.zeros(R); B4 = np.zeros(R); NN = np.zeros(R); ncnn = 0
    bi, bj = eng.bi, eng.bj
    for t in range(S):
        eng.sweep()
        sp = eng.spins
        M = sp.sum(axis=1, dtype=np.int64)
        m = M/eng.N
        m2 = m*m
        B2 += m2; B4 += m2*m2
        if t % 5 == 0:
            NN += (sp[:, bi].astype(np.int64)*sp[:, bj]).sum(axis=1)/bi.shape[0]
            ncnn += 1
    return B2, B4, NN/ncnn, S
'''
open('/work/prod.py', 'w').write(prod)
import prod
importlib_ok = True

Tlist = [1.5, 1.6, 1.7, 1.8]
t0 = time.time()
for T in Tlist:
    seed = 50000 + int(round(T*10))
    B2, B4, NN, S = prod.run_T(T, seed)
    np.savez(f'/work/res_T{T:.2f}.npz', B2=B2, B4=B4, NN=NN, S=S, seed=seed, R=1000, burn=200)
    U, err = sw.binder_from_sums(B2, B4, S)
    print(f"T={T:.2f} seed={seed} U={U:.6f} err={err:.6f} nn_corr={NN.mean():.5f} elapsed={time.time()-t0:.1f}", flush=True)
