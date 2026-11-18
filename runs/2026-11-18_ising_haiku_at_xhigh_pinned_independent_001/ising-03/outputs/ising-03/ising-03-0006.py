import numpy as np, json
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def jk(A, B):
    n = len(A)
    Asum, Bsum = A.sum(), B.sum()
    a = (Asum - A)/(n-1); b = (Bsum - B)/(n-1)
    Ui = 1 - b/(3*a*a)
    U = 1 - B.mean()/(3*A.mean()**2)
    err = np.sqrt((n-1)/n*np.sum((Ui-Ui.mean())**2))
    return U, err
rows = []
for T in Ts:
    pil = np.load(f'/work/pilot_T{T:.1f}.npz')
    pr = np.load(f'/work/prod_T{T:.1f}.npz')
    A1 = pil['m2'].mean(axis=0); B1 = pil['m4'].mean(axis=0)
    A2 = pr['bm2'].sum(axis=0)/pr['bm2'].shape[0]; B2 = pr['bm4'].sum(axis=0)/pr['bm4'].shape[0]
    n1 = pil['m2'].shape[0]; n2 = pr['bm2'].shape[0]*10
    Ac = (A1*n1 + A2*n2)/(n1+n2); Bc = (B1*n1 + B2*n2)/(n1+n2)
    U, e = jk(Ac, Bc)
    rows.append(dict(T=T, U=float(U), U_err=float(e), n_sweeps=int(n1+n2)))
    print(f"T={T} U={U:.7f} U_err={e:.3e} sweeps_per_replica={n1+n2}")
json.dump(rows, open('/work/final.json','w'), indent=1)
