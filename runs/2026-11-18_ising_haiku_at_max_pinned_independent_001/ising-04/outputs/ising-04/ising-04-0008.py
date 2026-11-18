
import sys, math, json
sys.path.insert(0, "/work")
import numpy as np

NN = 256
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def pooled_jk(arr, bpr):
    # arr: (R, n) magnetizations (int); blocks of equal length within each replica
    R, n = arr.shape
    m = arr.astype(np.float64) / NN
    m2 = m * m; m4 = m2 * m2
    L = n // bpr
    S2 = m2[:, :L*bpr].reshape(R, bpr, L).sum(axis=2).ravel()
    S4 = m4[:, :L*bpr].reshape(R, bpr, L).sum(axis=2).ravel()
    B = len(S2); ntot = B * L
    T2 = S2.sum(); T4 = S4.sum()
    Ufull = 1.0 - (T4/ntot) / (3.0 * (T2/ntot)**2)
    s2 = (T2 - S2) / ((B-1)*L); s4 = (T4 - S4) / ((B-1)*L)
    Uj = 1.0 - s4 / (3.0 * s2 * s2)
    err = math.sqrt((B-1)/B * np.sum((Uj - Uj.mean())**2))
    bias = (B-1) * (Uj.mean() - Ufull)
    return Ufull, err, bias, ntot

rows = []
for T in Ts:
    arr = np.load(f"/work/prod_T{T:.1f}.npy").astype(np.int64)
    U40, e40, b40, n = pooled_jk(arr, 40)
    _, e20, _, _ = pooled_jk(arr, 20)
    _, e80, _, _ = pooled_jk(arr, 80)
    Urep = np.array([1 - np.mean((r/NN)**4)/(3*np.mean((r/NN)**2)**2) for r in arr])
    se_rep = Urep.std(ddof=1)/math.sqrt(len(Urep))
    rows.append((T, U40, e40, n, e20, e80, se_rep, b40))
    print(f"T={T:.2f}  U={U40:.6f}  U_err(jk,160blk)={e40:.3e}  err(jk20 per-rep)={e20:.3e}  err(jk80)={e80:.3e}  se_from_replicas={se_rep:.3e}  jk_bias={b40:.1e}  n={n}")

with open("/work/final_table.json", "w") as f:
    json.dump([dict(T=r[0], U=float(r[1]), U_err=float(r[2]), n=int(r[3])) for r in rows], f, indent=1)
print("max U_err:", max(r[2] for r in rows))
