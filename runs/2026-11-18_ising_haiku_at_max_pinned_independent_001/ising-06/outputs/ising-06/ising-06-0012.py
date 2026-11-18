import sys, math
sys.path.insert(0, "/work")
import numpy as np, ising
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
for T in Ts:
    d = np.load(f"/work/prod_T{T:.2f}.npz")
    cU = d['chainU']; cE = d['chainErr']
    sd = cU.std(ddof=1)
    cold = cU[0::2].mean(); hot = cU[1::2].mean()
    print(f"T={T}: chain U={np.array2string(cU, precision=8)}  sd={sd:.2e}  mean jk chain err={cE.mean():.2e}  "
          f"sd/err={sd/max(cE.mean(),1e-12):.2f}  cold-mean={cold:.7f} hot-mean={hot:.7f}")
