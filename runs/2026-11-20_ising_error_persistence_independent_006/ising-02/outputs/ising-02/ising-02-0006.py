
import numpy as np, glob
rows=[]
for f in sorted(glob.glob('/work/sums_*.npz'), key=lambda x: float(x.split('_T')[1].split('_')[0])):
    d=np.load(f); S=d['S']; T=float(d['T']); R=S.shape[-1]; n=5000.0
    # per-replica totals (both halves)
    A2=S[0,0]+S[1,0]; A4=S[0,1]+S[1,1]
    def U_of(a2,a4):
        M2=a2.sum()/(n*2*len(a2)); M4=a4.sum()/(n*2*len(a4))
        return 1-M4/(3*M2**2)
    U=U_of(A2,A4)
    # jackknife over replicas
    Uj=np.array([U_of(np.delete(A2,i),np.delete(A4,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # halves
    U1=U_of(S[0,0],S[0,1]); U2=U_of(S[1,0],S[1,1])
    # bootstrap-style check with 20 blocks of replicas
    rows.append((T,U,err,U1,U2))
    print(f"T={T} U={U:.5f} jackknife_err={err:.5f}  U(first half)={U1:.5f} U(second half)={U2:.5f}")
