
import sys, time, numpy as np
sys.path.insert(0, '/work')
import sw, prod
from scipy.special import ellipk

def onsager_nn(T):
    K = 1.0/T
    s2 = np.sinh(2*K); c2 = np.cosh(2*K); t2 = np.tanh(2*K)
    k = 2*s2/c2**2
    Kk = ellipk(k**2)
    return 0.5/t2*(1 + (2/np.pi)*(2*t2**2 - 1)*Kk)

Tlist = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for T in Tlist:
    seed = 50000 + int(round(T*10))
    B2, B4, NN, S = prod.run_T(T, seed)
    np.savez(f'/work/res_T{T:.2f}.npz', B2=B2, B4=B4, NN=NN, S=S, seed=seed, R=1000, burn=200)
    U, err = sw.binder_from_sums(B2, B4, S)
    print(f"T={T:.2f} seed={seed} U={U:.6f} err={err:.6f} nn_corr={NN.mean():.5f} (+-{NN.std()/np.sqrt(len(NN)):.5f}) onsager_nn={onsager_nn(T):.5f} elapsed={time.time()-t0:.1f}", flush=True)
