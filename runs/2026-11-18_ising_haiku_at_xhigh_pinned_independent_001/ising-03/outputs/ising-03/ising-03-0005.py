import numpy as np
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R = 512

def jk(A, B):
    # A,B: per-replica means of m^2, m^4 (length R)
    Asum, Bsum = A.sum(), B.sum()
    n = len(A)
    a = (Asum - A)/(n-1); b = (Bsum - B)/(n-1)
    Ui = 1 - b/(3*a*a)
    U = 1 - B.mean()/(3*A.mean()**2)
    err = np.sqrt((n-1)/n*np.sum((Ui-Ui.mean())**2))
    return U, err

out = {}
for T in Ts:
    pil = np.load(f'/work/pilot_T{T:.1f}.npz')
    A1 = pil['m2'].mean(axis=0); B1 = pil['m4'].mean(axis=0)
    U1, e1 = jk(A1, B1)
    pr = np.load(f'/work/prod_T{T:.1f}.npz')
    A2 = pr['bm2'].sum(axis=0)/pr['bm2'].shape[0]
    B2 = pr['bm4'].sum(axis=0)/pr['bm4'].shape[0]
    U2, e2 = jk(A2, B2)
    # combined: per-replica weighted by sweeps (800 pilot, 1500 production)
    n1 = pil['m2'].shape[0]; n2 = 1500
    Ac = (A1*n1 + A2*n2)/(n1+n2); Bc = (B1*n1 + B2*n2)/(n1+n2)
    Uc, ec = jk(Ac, Bc)
    pull = (U1-U2)/np.sqrt(e1**2+e2**2)
    out[T] = (U1,e1,U2,e2,Uc,ec)
    print(f"T={T}: pilot U={U1:.5f}±{e1:.5f} | prod U={U2:.5f}±{e2:.5f} | comb U={Uc:.5f}±{ec:.5f} | pull={pull:+.2f}")
pulls = [(out[T][0]-out[T][2])/np.sqrt(out[T][1]**2+out[T][3]**2) for T in Ts]
print("pull rms:", np.sqrt(np.mean(np.square(pulls))))
print("max comb err:", max(out[T][5] for T in Ts))
