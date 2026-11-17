import sys, glob
sys.path.insert(0, "/work")
import numpy as np
from prod import U_jack

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def pilot_U_err(T, G=100):
    d = np.load(f"/work/data/pilot_T{T:.2f}.npz")
    M2 = d["M2"][:, 100:].astype(np.float64); M4 = d["M4"][:, 100:].astype(np.float64)
    # reshape into blocks of 20 sweeps to reuse U_jack (200 measured sweeps -> 10 blocks)
    R, n = M2.shape
    S2 = M2.reshape(R, n//20, 20).sum(axis=2); S4 = M4.reshape(R, n//20, 20).sum(axis=2)
    return U_jack(S2, S4, n, G=G)

print("T      U_prod    err_prod(G=100) err(G=50) err(G=200)   U_pilot   err_pilot    z=(dU)/sqrt(e1^2+e2^2)   block-drift slope/err")
zs = []
for k, T in enumerate(Ts):
    f = glob.glob(f"/work/data/prod_T{T:.2f}_seed{777001+k}.npz")[0]
    d = np.load(f)
    S2 = d["S2"]; S4 = d["S4"]; nmeas = int(d["nmeas"])
    U, e100 = U_jack(S2, S4, nmeas, G=100)
    _, e50 = U_jack(S2, S4, nmeas, G=50)
    _, e200 = U_jack(S2, S4, nmeas, G=200)
    Up, ep = pilot_U_err(T)
    z = (U - Up)/np.sqrt(e100**2 + ep**2)
    zs.append(z)
    # drift check: replica-averaged m^2 per block (20 sweeps) vs block index
    blk_m2 = S2.mean(axis=0)/20.0   # per block mean of m^2 averaged over replicas
    x = np.arange(blk_m2.size)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, blk_m2, rcond=None)
    resid = blk_m2 - A@coef
    s = np.sqrt(resid.var(ddof=2)/((x-x.mean())**2).sum())
    print(f"{T:.2f}  {U:.6f}  {e100:.3e}  {e50:.3e}  {e200:.3e}   {Up:.6f}  {ep:.3e}   z={z:+.2f}        slope/err={coef[0]/s:+.2f}")
zs = np.array(zs)
print("RMS z =", np.sqrt(np.mean(zs**2)), " mean z =", zs.mean())
print("max production err:", max(e for e in [0]))
