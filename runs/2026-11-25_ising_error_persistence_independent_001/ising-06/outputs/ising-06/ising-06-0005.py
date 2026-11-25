import numpy as np, glob
files = {1.9:1000,2.0:1001,2.1:1002,2.2:1003,2.3:1004,2.4:1005}
def U_of(M):  # M shape (time, R)
    m2=(M**2).mean(); m4=(M**4).mean()
    return 1-m4/(3*m2**2)
for T,seed in files.items():
    M=np.load(f'/work/M_T{T}_s{seed}.npy')
    nt,R=M.shape
    U=U_of(M)
    # jackknife over replicas
    m2_tot=(M**2).sum(); m4_tot=(M**4).sum(); cnt=M.size
    m2_r=(M**2).sum(axis=0); m4_r=(M**4).sum(axis=0); c=nt
    loo=np.empty(R)
    for r in range(R):
        a2=(m2_tot-m2_r[r])/(cnt-c); a4=(m4_tot-m4_r[r])/(cnt-c)
        loo[r]=1-a4/(3*a2**2)
    jk_err=np.sqrt((R-1)/R*((loo-loo.mean())**2).sum())
    # halves check
    h=nt//2
    U1=U_of(M[:h]); U2=U_of(M[h:])
    # per-time-block (replica-averaged) check: block of 300 sweeps
    print(f"T={T} U={U:.5f} jk_err={jk_err:.5f}  U(first half)={U1:.5f} U(second half)={U2:.5f} <|m|>={np.abs(M).mean():.4f}")
